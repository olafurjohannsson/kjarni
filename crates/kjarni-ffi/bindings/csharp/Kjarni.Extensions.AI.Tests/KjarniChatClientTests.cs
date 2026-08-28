using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Kjarni.Extensions.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace Kjarni.Extensions.AI.Tests
{
    /// <summary>
    /// Loads the chat model once for the whole class.
    /// </summary>
    /// <remarks>
    /// Without a fixture every test re-loads the weights, which is slow and turns one
    /// download problem into a wall of identical stack traces. Here a failure is captured
    /// once and reported as a skip reason instead.
    /// </remarks>
    public sealed class ChatClientFixture : IDisposable
    {
        /// <summary>
        /// The model under test.
        /// </summary>
        /// <remarks>
        /// Qwen rather than Llama on purpose. `meta-llama/*` repositories on Hugging Face are
        /// gated: fetching them needs an accepted licence and a token, so CI gets
        /// HTTP 401 even though a developer with the weights already cached sees green.
        /// `Qwen/Qwen2.5-0.5B-Instruct` is openly downloadable, and being the smallest
        /// instruct model in the registry it is also the fastest to load.
        /// </remarks>
        public const string Model = "qwen2.5-0.5b-instruct";

        public KjarniChatClient? Client { get; }

        /// <summary>Non-null when the model could not be obtained, in which case tests skip.</summary>
        public string? SkipReason { get; }

        public ChatClientFixture()
        {
            try
            {
                Client = new KjarniChatClient(model: Model, quiet: true);
            }
            catch (KjarniException ex) when (IsUnavailable(ex))
            {
                SkipReason = $"Model '{Model}' unavailable: {ex.Message}";
            }
        }

        /// <summary>
        /// True when the model could not be fetched, as opposed to the library misbehaving.
        /// A network outage or a gated repository should skip; anything else should fail.
        /// </summary>
        public static bool IsUnavailable(KjarniException ex) =>
            ex.Message.Contains("download", StringComparison.OrdinalIgnoreCase) ||
            ex.ErrorCode == KjarniErrorCode.ModelNotFound;

        public void Dispose() => Client?.Dispose();
    }

    /// <summary>
    /// Tests for the <see cref="IChatClient"/> adapter.
    /// </summary>
    /// <remarks>
    /// These assert the adapter's plumbing — history reaching the model, options being
    /// honoured, unsupported surfaces failing loudly — not generation quality. Assertions
    /// are phrased to survive a model that words its answers differently.
    /// </remarks>
    public class KjarniChatClientTests : IClassFixture<ChatClientFixture>
    {
        private const string Model = ChatClientFixture.Model;

        private readonly ChatClientFixture _fixture;

        public KjarniChatClientTests(ChatClientFixture fixture) => _fixture = fixture;

        /// <summary>Skips when the model is unavailable, otherwise yields the shared client.</summary>
        private KjarniChatClient Client
        {
            get
            {
                Skip.If(_fixture.SkipReason is not null, _fixture.SkipReason);
                return _fixture.Client!;
            }
        }

        [SkippableFact]
        public void Metadata_ReportsProviderAndModel()
        {
            var meta = Client.GetService(typeof(ChatClientMetadata)) as ChatClientMetadata;

            Assert.NotNull(meta);
            Assert.Equal("kjarni", meta!.ProviderName);
            Assert.Equal(Model, meta.DefaultModelId);
        }

        [SkippableFact]
        public void GetService_UnwrapsUnderlyingChat()
        {
            var client = Client;

            Assert.IsType<Chat>(client.GetService(typeof(Chat)));
            Assert.Same(client, client.GetService(typeof(IChatClient)));
            Assert.Null(client.GetService(typeof(string)));
        }

        [SkippableFact]
        public void GetService_WithServiceKey_ReturnsNull()
        {
            // A keyed lookup is a request for a specific named service, which this client
            // does not provide; returning the unkeyed instance would be wrong.
            Assert.Null(Client.GetService(typeof(ChatClientMetadata), "some-key"));
        }

        [SkippableFact]
        public async Task GetResponseAsync_SingleTurn_ReturnsAssistantMessage()
        {
            var response = await Client.GetResponseAsync(
                new[] { new ChatMessage(ChatRole.User, "Reply with exactly the word: pong") });

            Assert.False(string.IsNullOrWhiteSpace(response.Text));
            Assert.Equal(ChatRole.Assistant, response.Messages[0].Role);
            Assert.Equal(Model, response.ModelId);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        }

        [SkippableFact]
        public async Task GetResponseAsync_UsesPriorTurns()
        {
            // The point of the adapter: IChatClient replays the whole transcript, so the
            // history must reach the model rather than being silently dropped.
            var response = await Client.GetResponseAsync(Conversation());

            Assert.Contains("lafur", response.Text, StringComparison.OrdinalIgnoreCase);
        }

        [SkippableFact]
        public async Task GetResponseAsync_WithoutHistory_DoesNotKnowTheAnswer()
        {
            // Control for GetResponseAsync_UsesPriorTurns. Without this, that test would
            // also pass if the model simply guessed the name from nowhere.
            var response = await Client.GetResponseAsync(
                new[] { new ChatMessage(ChatRole.User, "What is my name?") });

            Assert.DoesNotContain("lafur", response.Text, StringComparison.OrdinalIgnoreCase);
        }

        [SkippableFact]
        public async Task GetResponseAsync_EmptyMessages_Throws()
        {
            var client = Client;

            await Assert.ThrowsAsync<ArgumentException>(
                () => client.GetResponseAsync(Array.Empty<ChatMessage>()));
        }

        [SkippableFact]
        public async Task GetStreamingResponseAsync_YieldsIncrementallyAndUsesHistory()
        {
            var chunks = new List<string>();
            await foreach (var update in Client.GetStreamingResponseAsync(Conversation()))
            {
                Assert.Equal(ChatRole.Assistant, update.Role);
                if (update.Text.Length > 0) chunks.Add(update.Text);
            }

            Assert.True(chunks.Count > 1, $"expected multiple chunks, got {chunks.Count}");
            Assert.Contains("lafur", string.Concat(chunks), StringComparison.OrdinalIgnoreCase);
        }

        [SkippableFact]
        public async Task GetStreamingResponseAsync_UpdatesShareOneResponseId()
        {
            var ids = new HashSet<string?>();
            await foreach (var update in Client.GetStreamingResponseAsync(
                new[] { new ChatMessage(ChatRole.User, "Count to three.") }))
            {
                ids.Add(update.ResponseId);
            }

            Assert.Single(ids);
            Assert.NotNull(ids.Single());
        }

        [SkippableFact]
        public async Task GetStreamingResponseAsync_LastUpdateCarriesFinishReason()
        {
            ChatResponseUpdate? last = null;
            await foreach (var update in Client.GetStreamingResponseAsync(
                new[] { new ChatMessage(ChatRole.User, "Say hello.") }))
            {
                last = update;
            }

            Assert.NotNull(last);
            Assert.Equal(ChatFinishReason.Stop, last!.FinishReason);
        }

        [SkippableFact]
        public async Task GetResponseAsync_ZeroTemperature_IsDeterministic()
        {
            var client = Client;
            var options = new ChatOptions { Temperature = 0f, MaxOutputTokens = 24 };
            var prompt = new[] { new ChatMessage(ChatRole.User, "Name one color.") };

            var first = await client.GetResponseAsync(prompt, options);
            var second = await client.GetResponseAsync(prompt, options);

            Assert.Equal(first.Text.Trim(), second.Text.Trim());
        }

        [SkippableFact]
        public async Task GetResponseAsync_Tools_ThrowNotSupported()
        {
            // Tool calling is not implemented. Accepting the option and ignoring it would
            // leave the caller believing their functions were offered to the model.
            var client = Client;
            var options = new ChatOptions
            {
                Tools = new List<AITool> { AIFunctionFactory.Create(() => 1, "one") },
            };

            await Assert.ThrowsAsync<NotSupportedException>(
                () => client.GetResponseAsync(
                    new[] { new ChatMessage(ChatRole.User, "hi") }, options));
        }

        [SkippableFact]
        public async Task GetResponseAsync_ToolRole_ThrowsNotSupported()
        {
            var client = Client;
            var messages = new[]
            {
                new ChatMessage(ChatRole.Tool, "some tool output"),
                new ChatMessage(ChatRole.User, "and?"),
            };

            await Assert.ThrowsAsync<NotSupportedException>(
                () => client.GetResponseAsync(messages));
        }

        [SkippableFact]
        public async Task GetResponseAsync_AfterDispose_Throws()
        {
            Skip.If(_fixture.SkipReason is not null, _fixture.SkipReason);

            var client = new KjarniChatClient(model: Model, quiet: true);
            client.Dispose();

            await Assert.ThrowsAsync<ObjectDisposedException>(
                () => client.GetResponseAsync(new[] { new ChatMessage(ChatRole.User, "hi") }));
        }

        [SkippableFact]
        public void AsChatClient_WrapsWithoutTakingOwnership()
        {
            Skip.If(_fixture.SkipReason is not null, _fixture.SkipReason);

            using var chat = new Chat(model: Model, quiet: true);

            var client = chat.AsChatClient(modelId: Model);
            client.Dispose();

            // ownsChat defaults to false, so the caller's Chat must still be usable.
            Assert.False(string.IsNullOrEmpty(chat.ModelName));
        }

        [SkippableFact]
        public void AddKjarniChatClient_RegistersSingleton()
        {
            Skip.If(_fixture.SkipReason is not null, _fixture.SkipReason);

            var provider = new ServiceCollection()
                .AddKjarniChatClient(model: Model, quiet: true)
                .BuildServiceProvider();

            var first = provider.GetRequiredService<IChatClient>();
            var second = provider.GetRequiredService<IChatClient>();

            Assert.IsType<KjarniChatClient>(first);
            Assert.Same(first, second);
        }

        /// <summary>
        /// A transcript whose answer is only derivable from the history.
        /// </summary>
        private static ChatMessage[] Conversation() => new[]
        {
            new ChatMessage(ChatRole.System, "You answer in as few words as possible."),
            new ChatMessage(ChatRole.User, "My name is Olafur and I live in Reykjavik."),
            new ChatMessage(ChatRole.Assistant, "Noted."),
            new ChatMessage(ChatRole.User, "What is my name?"),
        };
    }
}
