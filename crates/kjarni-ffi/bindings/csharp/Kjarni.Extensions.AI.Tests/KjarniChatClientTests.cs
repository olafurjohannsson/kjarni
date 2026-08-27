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
    /// Tests for the <see cref="IChatClient"/> adapter.
    /// </summary>
    /// <remarks>
    /// The smallest instruct model in the registry is used deliberately: these assert the
    /// adapter's plumbing — history reaching the model, options being honoured, unsupported
    /// surfaces failing loudly — not generation quality. Assertions are phrased to survive
    /// a model that words its answers differently.
    /// </remarks>
    public class KjarniChatClientTests : IDisposable
    {
        private const string Model = "llama3.2-1b-instruct";

        private readonly KjarniChatClient _client;

        public KjarniChatClientTests()
        {
            _client = new KjarniChatClient(model: Model, quiet: true);
        }

        public void Dispose() => _client.Dispose();

        [Fact]
        public void Metadata_ReportsProviderAndModel()
        {
            var meta = _client.GetService(typeof(ChatClientMetadata)) as ChatClientMetadata;

            Assert.NotNull(meta);
            Assert.Equal("kjarni", meta!.ProviderName);
            Assert.Equal(Model, meta.DefaultModelId);
        }

        [Fact]
        public void GetService_UnwrapsUnderlyingChat()
        {
            Assert.IsType<Chat>(_client.GetService(typeof(Chat)));
            Assert.Same(_client, _client.GetService(typeof(IChatClient)));
            Assert.Null(_client.GetService(typeof(string)));
        }

        [Fact]
        public void GetService_WithServiceKey_ReturnsNull()
        {
            // A keyed lookup is a request for a specific named service, which this client
            // does not provide; returning the unkeyed instance would be wrong.
            Assert.Null(_client.GetService(typeof(ChatClientMetadata), "some-key"));
        }

        [Fact]
        public async Task GetResponseAsync_SingleTurn_ReturnsAssistantMessage()
        {
            var response = await _client.GetResponseAsync(
                new[] { new ChatMessage(ChatRole.User, "Reply with exactly the word: pong") });

            Assert.False(string.IsNullOrWhiteSpace(response.Text));
            Assert.Equal(ChatRole.Assistant, response.Messages[0].Role);
            Assert.Equal(Model, response.ModelId);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        }

        [Fact]
        public async Task GetResponseAsync_UsesPriorTurns()
        {
            // The point of the adapter: IChatClient replays the whole transcript, so the
            // history must reach the model rather than being silently dropped.
            var response = await _client.GetResponseAsync(Conversation());

            Assert.Contains("lafur", response.Text, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public async Task GetResponseAsync_WithoutHistory_DoesNotKnowTheAnswer()
        {
            // Control for GetResponseAsync_UsesPriorTurns. Without this, that test would
            // also pass if the model simply guessed the name from nowhere.
            var response = await _client.GetResponseAsync(
                new[] { new ChatMessage(ChatRole.User, "What is my name?") });

            Assert.DoesNotContain("lafur", response.Text, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public async Task GetResponseAsync_EmptyMessages_Throws()
        {
            await Assert.ThrowsAsync<ArgumentException>(
                () => _client.GetResponseAsync(Array.Empty<ChatMessage>()));
        }

        [Fact]
        public async Task GetStreamingResponseAsync_YieldsIncrementallyAndUsesHistory()
        {
            var chunks = new List<string>();
            await foreach (var update in _client.GetStreamingResponseAsync(Conversation()))
            {
                Assert.Equal(ChatRole.Assistant, update.Role);
                if (update.Text.Length > 0) chunks.Add(update.Text);
            }

            Assert.True(chunks.Count > 1, $"expected multiple chunks, got {chunks.Count}");
            Assert.Contains("lafur", string.Concat(chunks), StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public async Task GetStreamingResponseAsync_UpdatesShareOneResponseId()
        {
            var ids = new HashSet<string?>();
            await foreach (var update in _client.GetStreamingResponseAsync(
                new[] { new ChatMessage(ChatRole.User, "Count to three.") }))
            {
                ids.Add(update.ResponseId);
            }

            Assert.Single(ids);
            Assert.NotNull(ids.Single());
        }

        [Fact]
        public async Task GetStreamingResponseAsync_LastUpdateCarriesFinishReason()
        {
            ChatResponseUpdate? last = null;
            await foreach (var update in _client.GetStreamingResponseAsync(
                new[] { new ChatMessage(ChatRole.User, "Say hello.") }))
            {
                last = update;
            }

            Assert.NotNull(last);
            Assert.Equal(ChatFinishReason.Stop, last!.FinishReason);
        }

        [Fact]
        public async Task GetResponseAsync_ZeroTemperature_IsDeterministic()
        {
            var options = new ChatOptions { Temperature = 0f, MaxOutputTokens = 24 };
            var prompt = new[] { new ChatMessage(ChatRole.User, "Name one color.") };

            var first = await _client.GetResponseAsync(prompt, options);
            var second = await _client.GetResponseAsync(prompt, options);

            Assert.Equal(first.Text.Trim(), second.Text.Trim());
        }

        [Fact]
        public async Task GetResponseAsync_Tools_ThrowNotSupported()
        {
            // Tool calling is not implemented. Accepting the option and ignoring it would
            // leave the caller believing their functions were offered to the model.
            var options = new ChatOptions
            {
                Tools = new List<AITool> { AIFunctionFactory.Create(() => 1, "one") },
            };

            await Assert.ThrowsAsync<NotSupportedException>(
                () => _client.GetResponseAsync(
                    new[] { new ChatMessage(ChatRole.User, "hi") }, options));
        }

        [Fact]
        public async Task GetResponseAsync_ToolRole_ThrowsNotSupported()
        {
            var messages = new[]
            {
                new ChatMessage(ChatRole.Tool, "some tool output"),
                new ChatMessage(ChatRole.User, "and?"),
            };

            await Assert.ThrowsAsync<NotSupportedException>(
                () => _client.GetResponseAsync(messages));
        }

        [Fact]
        public async Task GetResponseAsync_AfterDispose_Throws()
        {
            var client = new KjarniChatClient(model: Model, quiet: true);
            client.Dispose();

            await Assert.ThrowsAsync<ObjectDisposedException>(
                () => client.GetResponseAsync(new[] { new ChatMessage(ChatRole.User, "hi") }));
        }

        [Fact]
        public void AsChatClient_WrapsWithoutTakingOwnership()
        {
            using var chat = new Chat(model: Model, quiet: true);

            var client = chat.AsChatClient(modelId: Model);
            client.Dispose();

            // ownsChat defaults to false, so the caller's Chat must still be usable.
            Assert.False(string.IsNullOrEmpty(chat.ModelName));
        }

        [Fact]
        public void AddKjarniChatClient_RegistersSingleton()
        {
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
