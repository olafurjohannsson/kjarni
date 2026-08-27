using System;
using Kjarni;
using Kjarni.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

// Same reasoning as KjarniEmbeddingGeneratorExtensions: this lives in the
// Microsoft.Extensions.AI namespace so that AsChatClient/AddKjarniChatClient light up
// for a consumer who already wrote `using Kjarni;` and `using Microsoft.Extensions.AI;`.
namespace Microsoft.Extensions.AI
{
    /// <summary>
    /// Extension methods for exposing a Kjarni <see cref="Chat"/> as an <see cref="IChatClient"/>.
    /// </summary>
    public static class KjarniChatClientExtensions
    {
        /// <summary>
        /// Wraps this <see cref="Chat"/> in an <see cref="IChatClient"/>.
        /// </summary>
        /// <param name="chat">The chat instance to wrap.</param>
        /// <param name="modelId">Model identifier reported through metadata.</param>
        /// <param name="ownsChat">
        /// When <see langword="true"/>, disposing the returned client also disposes
        /// <paramref name="chat"/>. Defaults to <see langword="false"/> so the caller keeps
        /// ownership of a chat it created.
        /// </param>
        public static IChatClient AsChatClient(
            this Chat chat,
            string? modelId = null,
            bool ownsChat = false)
            => new KjarniChatClient(chat, modelId, ownsChat);

        /// <summary>
        /// Registers a local Kjarni chat model as a singleton <see cref="IChatClient"/>.
        /// </summary>
        /// <remarks>
        /// Registered as a singleton because model weights are loaded once and are expensive
        /// to re-initialize. The client serializes concurrent calls internally, so sharing one
        /// instance across requests is safe — but generation is compute-bound, so concurrent
        /// requests queue rather than overlap.
        /// </remarks>
        /// <param name="services">The service collection.</param>
        /// <param name="model">Kjarni model name, e.g. <c>llama3.2-3b-instruct</c>.</param>
        /// <param name="systemPrompt">Default system prompt, or <see langword="null"/> for none.</param>
        /// <param name="mode">Chat mode preset.</param>
        /// <param name="device"><c>"cpu"</c> or <c>"gpu"</c>.</param>
        /// <param name="quiet">Suppress model download/progress output.</param>
        public static IServiceCollection AddKjarniChatClient(
            this IServiceCollection services,
            string model = "llama3.2-3b-instruct",
            string? systemPrompt = null,
            ChatMode mode = ChatMode.Default,
            string device = "cpu",
            bool quiet = true)
        {
            if (services is null) throw new ArgumentNullException(nameof(services));

            services.AddSingleton<IChatClient>(
                _ => new KjarniChatClient(model, systemPrompt, mode, device, quiet));

            return services;
        }
    }
}
