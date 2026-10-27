using System;
using Xunit;

namespace Kjarni.Tests
{
    public class ChatTests
    {
        [SkippableFact]
        public void QuantizedChatReplies()
        {
            Heavy.SkipIfSet();
            using var chat = new Chat("qwen2.5-0.5b-instruct", quiet: true, quantized: true);
            var config = GenerationConfig.Greedy();
            config.MaxNewTokens = 16;
            var reply = chat.Send("What is the capital of France? Answer in one word.", config);
            Assert.Contains("paris", reply, StringComparison.OrdinalIgnoreCase);
        }
    }
}
