using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Kjarni;
using Microsoft.Extensions.AI;

namespace Kjarni.Extensions.AI
{
    /// <summary>
    /// An <see cref="IChatClient"/> backed by a local Kjarni <see cref="Chat"/>. The model
    /// is loaded directly by your application, so prompts and responses stay on the machine
    /// and there is nothing listening on a port to keep running alongside it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="IChatClient"/> is stateless — the caller replays the whole transcript on
    /// every call — so this maps onto <see cref="Chat.SendWithHistory(IEnumerable{ChatTurn}, string)"/>
    /// rather than <see cref="ChatConversation"/>. Nothing is retained between calls, which
    /// is what a shared singleton in a web application needs.
    /// </para>
    /// <para>
    /// The underlying native chat is not re-entrant, so concurrent calls are serialized
    /// internally. Because generation is compute-bound rather than I/O-bound, a request
    /// that waits on the gate is waiting for a real CPU or GPU, not for a socket: throughput
    /// is one conversation at a time per instance.
    /// </para>
    /// </remarks>
    public sealed class KjarniChatClient : IChatClient
    {
        private const string ProviderNameConstant = "kjarni";

        private readonly Chat _chat;
        private readonly bool _ownsChat;
        private readonly ChatClientMetadata _metadata;
        private readonly SemaphoreSlim _gate = new SemaphoreSlim(1, 1);
        private readonly string _modelId;

        private bool _disposed;

        /// <summary>
        /// Creates a client that owns its own <see cref="Chat"/>.
        /// </summary>
        /// <param name="model">Kjarni model name, e.g. <c>llama3.2-3b-instruct</c>.</param>
        /// <param name="systemPrompt">Default system prompt, or <see langword="null"/> for none.</param>
        /// <param name="mode">Chat mode preset.</param>
        /// <param name="device"><c>"cpu"</c> or <c>"gpu"</c>.</param>
        /// <param name="quiet">Suppress model download/progress output.</param>
        public KjarniChatClient(
            string model = "llama3.2-3b-instruct",
            string? systemPrompt = null,
            ChatMode mode = ChatMode.Default,
            string device = "cpu",
            bool quiet = true)
            : this(
                new Chat(model: model, systemPrompt: systemPrompt, mode: mode, quiet: quiet, device: device),
                modelId: model,
                ownsChat: true)
        {
        }

        /// <summary>
        /// Wraps an existing <see cref="Chat"/>.
        /// </summary>
        /// <param name="chat">The chat instance to delegate to.</param>
        /// <param name="modelId">Model identifier reported through metadata.</param>
        /// <param name="ownsChat">
        /// When <see langword="true"/>, disposing this client also disposes <paramref name="chat"/>.
        /// </param>
        public KjarniChatClient(Chat chat, string? modelId = null, bool ownsChat = false)
        {
            _chat = chat ?? throw new ArgumentNullException(nameof(chat));
            _ownsChat = ownsChat;
            _modelId = modelId ?? SafeModelName(chat);
            _metadata = new ChatClientMetadata(
                providerName: ProviderNameConstant,
                providerUri: null,
                defaultModelId: _modelId);
        }

        /// <summary>The context window of the underlying model, in tokens.</summary>
        public int ContextSize => _chat.ContextSize;

        /// <inheritdoc />
        public async Task<ChatResponse> GetResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            if (messages is null) throw new ArgumentNullException(nameof(messages));
            ThrowIfDisposed();
            ValidateOptions(options);

            var (history, prompt) = Split(messages, options);
            var config = ToGenerationConfig(options);

            cancellationToken.ThrowIfCancellationRequested();
            await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            string text;
            try
            {
                ThrowIfDisposed();
                cancellationToken.ThrowIfCancellationRequested();
                text = _chat.SendWithHistory(history, prompt, config);
            }
            finally
            {
                _gate.Release();
            }

            return new ChatResponse(new ChatMessage(ChatRole.Assistant, text))
            {
                ModelId = _modelId,
                ResponseId = Guid.NewGuid().ToString("N"),
                CreatedAt = DateTimeOffset.UtcNow,
                FinishReason = ChatFinishReason.Stop,
            };
        }

        /// <inheritdoc />
        /// <remarks>
        /// Native generation is a blocking callback loop, so it runs on a background thread
        /// and feeds an unbounded channel that this method drains. Cancelling stops
        /// generation at the next token rather than after the full response, because the
        /// token callback reports the cancellation back to the native side.
        /// </remarks>
        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (messages is null) throw new ArgumentNullException(nameof(messages));
            ThrowIfDisposed();
            ValidateOptions(options);

            var (history, prompt) = Split(messages, options);
            var config = ToGenerationConfig(options);

            var responseId = Guid.NewGuid().ToString("N");
            var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = true,
            });

            cancellationToken.ThrowIfCancellationRequested();
            await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);

            var producer = Task.Run(() =>
            {
                try
                {
                    _chat.StreamWithHistory(history, prompt, config, token =>
                    {
                        if (cancellationToken.IsCancellationRequested) return false;
                        channel.Writer.TryWrite(token);
                        return true;
                    });
                    channel.Writer.TryComplete();
                }
                catch (Exception ex)
                {
                    channel.Writer.TryComplete(ex);
                }
            }, CancellationToken.None);

            try
            {
                while (await channel.Reader.WaitToReadAsync(cancellationToken).ConfigureAwait(false))
                {
                    while (channel.Reader.TryRead(out var token))
                    {
                        yield return new ChatResponseUpdate(ChatRole.Assistant, token)
                        {
                            ModelId = _modelId,
                            ResponseId = responseId,
                            MessageId = responseId,
                            CreatedAt = DateTimeOffset.UtcNow,
                        };
                    }
                }

                // Surfaces a generation failure, and lets the final update carry a finish reason.
                await producer.ConfigureAwait(false);

                yield return new ChatResponseUpdate(ChatRole.Assistant, string.Empty)
                {
                    ModelId = _modelId,
                    ResponseId = responseId,
                    MessageId = responseId,
                    CreatedAt = DateTimeOffset.UtcNow,
                    FinishReason = ChatFinishReason.Stop,
                };
            }
            finally
            {
                // The native call holds the handle until it returns, so the gate cannot be
                // released while the producer is still running — including when the consumer
                // abandons the enumerator early.
                await producer.ConfigureAwait(ConfigureAwaitOptions.SuppressThrowing);
                _gate.Release();
            }
        }

        /// <inheritdoc />
        public object? GetService(Type serviceType, object? serviceKey = null)
        {
            if (serviceType is null) throw new ArgumentNullException(nameof(serviceType));
            if (serviceKey is not null) return null;

            if (serviceType == typeof(ChatClientMetadata)) return _metadata;
            if (serviceType == typeof(Chat)) return _chat;
            if (serviceType.IsInstanceOfType(this)) return this;

            return null;
        }

        /// <inheritdoc />
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _gate.Dispose();
            if (_ownsChat) _chat.Dispose();
        }

        /// <summary>
        /// Splits the transcript into prior turns and the message to respond to.
        /// </summary>
        /// <remarks>
        /// <see cref="ChatOptions.Instructions"/> is prepended as a system turn when present,
        /// which overrides the system prompt supplied at construction — the same precedence
        /// the native layer applies.
        /// </remarks>
        private static (List<ChatTurn> History, string Prompt) Split(
            IEnumerable<ChatMessage> messages, ChatOptions? options)
        {
            var list = messages as IList<ChatMessage> ?? messages.ToList();

            var turns = new List<ChatTurn>(list.Count + 1);
            if (!string.IsNullOrEmpty(options?.Instructions))
                turns.Add(ChatTurn.System(options!.Instructions!));

            if (list.Count == 0)
                throw new ArgumentException("At least one message is required.", nameof(messages));

            for (int i = 0; i < list.Count - 1; i++)
            {
                var mapped = Map(list[i]);
                if (mapped.HasValue) turns.Add(mapped.Value);
            }

            // The final message is what the model responds to. Its role is not required to be
            // User — an assistant-final transcript is a continuation request — but Kjarni
            // templates the trailing turn as the user side either way.
            var prompt = list[list.Count - 1].Text ?? string.Empty;
            return (turns, prompt);
        }

        private static ChatTurn? Map(ChatMessage message)
        {
            var text = message.Text;
            if (string.IsNullOrEmpty(text)) return null;

            if (message.Role == ChatRole.System) return ChatTurn.System(text);
            if (message.Role == ChatRole.Assistant) return ChatTurn.Assistant(text);
            if (message.Role == ChatRole.User) return ChatTurn.User(text);

            // ChatRole.Tool and any custom role: Kjarni has no tool-calling surface, so
            // folding these in as user text would silently misrepresent the transcript.
            throw new NotSupportedException(
                $"Kjarni does not support the '{message.Role}' role. Supported roles are " +
                "System, User and Assistant; tool calling is not yet available.");
        }

        private static void ValidateOptions(ChatOptions? options)
        {
            if (options is null) return;

            if (options.Tools is { Count: > 0 })
            {
                throw new NotSupportedException(
                    "Kjarni does not support tool calling yet. Remove ChatOptions.Tools, or use " +
                    "a client that provides function invocation.");
            }

            if (options.ResponseFormat is ChatResponseFormatJson)
            {
                throw new NotSupportedException(
                    "Kjarni does not support structured output enforcement. Ask for JSON in the " +
                    "prompt and validate the result, or use a client with native JSON mode.");
            }
        }

        /// <summary>
        /// Maps the subset of <see cref="ChatOptions"/> that Kjarni's sampler implements.
        /// </summary>
        /// <remarks>
        /// Returns <see langword="null"/> when nothing is set, so the model's own defaults
        /// apply rather than a synthesised config. Options with no Kjarni equivalent
        /// (<c>Seed</c>, <c>StopSequences</c>, <c>FrequencyPenalty</c>, <c>PresencePenalty</c>)
        /// are ignored rather than approximated: mapping a frequency penalty onto a repetition
        /// penalty would change output in ways the caller did not ask for.
        /// </remarks>
        private static GenerationConfig? ToGenerationConfig(ChatOptions? options)
        {
            if (options is null) return null;

            bool any = options.Temperature.HasValue
                    || options.TopP.HasValue
                    || options.TopK.HasValue
                    || options.MaxOutputTokens.HasValue;
            if (!any) return null;

            var config = GenerationConfig.Default();

            if (options.Temperature is float t)
            {
                config.Temperature = t;
                // Temperature 0 means greedy, which is a sampling mode rather than a value.
                config.DoSample = t > 0f ? 1 : 0;
            }
            if (options.TopP is float p) config.TopP = p;
            if (options.TopK is int k) config.TopK = k;
            if (options.MaxOutputTokens is int max) config.MaxNewTokens = max;

            return config;
        }

        private static string SafeModelName(Chat chat)
        {
            try
            {
                var name = chat.ModelName;
                return string.IsNullOrEmpty(name) ? "unknown" : name;
            }
            catch
            {
                return "unknown";
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(KjarniChatClient));
        }
    }
}
