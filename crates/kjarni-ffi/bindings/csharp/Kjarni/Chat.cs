using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Kjarni
{
    public enum ChatMode
    {
        Default = 0,
        Creative = 1,
        Reasoning = 2,
    }

    /// <summary>
    /// Generation parameters for decoder models.
    /// Use -1 sentinel values to keep model defaults.
    /// </summary>
    public struct GenerationConfig
    {
        public float Temperature;
        public int TopK;
        public float TopP;
        public float MinP;
        public float RepetitionPenalty;
        public int MaxNewTokens;
        public int DoSample;

        /// <summary>All model defaults.</summary>
        public static GenerationConfig Default() => new GenerationConfig
        {
            Temperature = -1.0f,
            TopK = -1,
            TopP = -1.0f,
            MinP = -1.0f,
            RepetitionPenalty = -1.0f,
            MaxNewTokens = -1,
            DoSample = -1,
        };

        /// <summary>Greedy decoding (deterministic).</summary>
        public static GenerationConfig Greedy() => new GenerationConfig
        {
            Temperature = 0.0f,
            TopK = -1,
            TopP = -1.0f,
            MinP = -1.0f,
            RepetitionPenalty = -1.0f,
            MaxNewTokens = -1,
            DoSample = 0,
        };

        /// <summary>Creative generation.</summary>
        public static GenerationConfig Creative() => new GenerationConfig
        {
            Temperature = 0.9f,
            TopK = 50,
            TopP = 0.95f,
            MinP = -1.0f,
            RepetitionPenalty = -1.0f,
            MaxNewTokens = -1,
            DoSample = 1,
        };
    }

    /// <summary>
    /// High-level chat interface for conversational AI using local decoder models.
    /// </summary>
    public class Chat : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;
        private Native.KjarniStreamCallback? _activeCallback;

        /// <summary>
        /// Create a new Chat instance.
        /// </summary>
        /// <param name="model">Model name, e.g. "llama3.2-1b-instruct", "qwen2-0.5b-instruct"</param>
        /// <param name="systemPrompt">Optional system prompt</param>
        /// <param name="mode">Chat mode (Default, Creative, Reasoning)</param>
        /// <param name="quiet">Suppress loading output</param>
        /// <param name="device">Device: "cpu" or "gpu"</param>
        public Chat(string model, string? systemPrompt = null, ChatMode mode = ChatMode.Default,
                     bool quiet = false, string? device = null)
        {
            var config = Native.kjarni_chat_config_default();

            var modelBytes = Encoding.UTF8.GetBytes(model + '\0');
            var modelPin = GCHandle.Alloc(modelBytes, GCHandleType.Pinned);
            config.ModelName = modelPin.AddrOfPinnedObject();
            config.Device = device == "gpu" ? KjarniDevice.Gpu : KjarniDevice.Cpu;
            config.Mode = (int)mode;
            config.Quiet = quiet ? 1 : 0;

            GCHandle systemPin = default;
            if (systemPrompt != null)
            {
                var systemBytes = Encoding.UTF8.GetBytes(systemPrompt + '\0');
                systemPin = GCHandle.Alloc(systemBytes, GCHandleType.Pinned);
                config.SystemPrompt = systemPin.AddrOfPinnedObject();
            }

            try
            {
                Native.CheckError(Native.kjarni_chat_new(ref config, out _handle));
            }
            finally
            {
                modelPin.Free();
                if (systemPin.IsAllocated)
                    systemPin.Free();
            }
        }

        /// <summary>
        /// Send a message and get the full response (blocking).
        /// </summary>
        public string Send(string message)
        {
            ThrowIfDisposed();
            Native.CheckError(Native.kjarni_chat_send(
                _handle, message, IntPtr.Zero, out var resultPtr));
            return ConsumeString(resultPtr);
        }

        /// <summary>
        /// Send a message with custom generation config.
        /// </summary>
        public string Send(string message, GenerationConfig config)
        {
            ThrowIfDisposed();
            var nativeConfig = ToNativeConfig(config);
            Native.CheckError(Native.kjarni_chat_send_config(
                _handle, message, ref nativeConfig, out var resultPtr));
            return ConsumeString(resultPtr);
        }

        /// <summary>
        /// Stream a response token by token.
        /// Return false from the callback to stop generation.
        /// </summary>
        public void Stream(string message, Func<string, bool> onToken)
        {
            ThrowIfDisposed();
            _activeCallback = (textPtr, _) =>
            {
                var text = Marshal.PtrToStringUTF8(textPtr) ?? "";
                return onToken(text);
            };
            try
            {
                Native.CheckError(Native.kjarni_chat_stream(
                    _handle, message, IntPtr.Zero,
                    _activeCallback, IntPtr.Zero, IntPtr.Zero));
            }
            finally
            {
                _activeCallback = null;
            }
        }

        /// <summary>
        /// Stream with custom generation config.
        /// </summary>
        public void Stream(string message, GenerationConfig config, Func<string, bool> onToken)
        {
            ThrowIfDisposed();
            var nativeConfig = ToNativeConfig(config);
            _activeCallback = (textPtr, _) =>
            {
                var text = Marshal.PtrToStringUTF8(textPtr) ?? "";
                return onToken(text);
            };
            try
            {
                Native.CheckError(Native.kjarni_chat_stream_config(
                    _handle, message, ref nativeConfig,
                    _activeCallback, IntPtr.Zero, IntPtr.Zero));
            }
            finally
            {
                _activeCallback = null;
            }
        }

        /// <summary>
        /// Create a stateful conversation that maintains history.
        /// </summary>
        public ChatConversation Conversation()
        {
            ThrowIfDisposed();
            return new ChatConversation(this);
        }

        /// <summary>Get the model name.</summary>
        public string ModelName
        {
            get
            {
                ThrowIfDisposed();
                var needed = (int)Native.kjarni_chat_model_name(_handle, IntPtr.Zero, UIntPtr.Zero);
                if (needed == 0) return "";
                var buf = Marshal.AllocHGlobal(needed + 1);
                try
                {
                    Native.kjarni_chat_model_name(_handle, buf, (UIntPtr)(needed + 1));
                    return Marshal.PtrToStringUTF8(buf) ?? "";
                }
                finally
                {
                    Marshal.FreeHGlobal(buf);
                }
            }
        }

        /// <summary>Get the context window size.</summary>
        public int ContextSize
        {
            get
            {
                ThrowIfDisposed();
                return (int)Native.kjarni_chat_context_size(_handle);
            }
        }

        internal IntPtr Handle => _handle;

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Chat));
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                if (_handle != IntPtr.Zero)
                {
                    Native.kjarni_chat_free(_handle);
                    _handle = IntPtr.Zero;
                }
            }
        }

        // Helpers

        private static string ConsumeString(IntPtr ptr)
        {
            try
            {
                return ptr != IntPtr.Zero ? Marshal.PtrToStringUTF8(ptr) ?? "" : "";
            }
            finally
            {
                if (ptr != IntPtr.Zero)
                    Native.kjarni_string_free(ptr);
            }
        }

        internal static Native.KjarniGenerationConfig ToNativeConfig(GenerationConfig config)
        {
            return new Native.KjarniGenerationConfig
            {
                Temperature = config.Temperature,
                TopK = config.TopK,
                TopP = config.TopP,
                MinP = config.MinP,
                RepetitionPenalty = config.RepetitionPenalty,
                MaxNewTokens = config.MaxNewTokens,
                DoSample = config.DoSample,
            };
        }
    }

    /// <summary>
    /// Stateful conversation that maintains chat history automatically.
    /// The parent Chat must outlive this object.
    /// </summary>
    public class ChatConversation : IDisposable
    {
        private IntPtr _handle;
        private readonly Chat _chat;
        private bool _disposed;
        private Native.KjarniStreamCallback? _activeCallback;

        internal ChatConversation(Chat chat)
        {
            _chat = chat;
            Native.CheckError(Native.kjarni_chat_conversation_new(chat.Handle, out _handle));
        }

        /// <summary>
        /// Send a message and get the response. Both are added to history.
        /// </summary>
        public string Send(string message)
        {
            ThrowIfDisposed();
            Native.CheckError(Native.kjarni_chat_conversation_send(
                _handle, message, IntPtr.Zero, out var resultPtr));
            return ConsumeString(resultPtr);
        }

        /// <summary>
        /// Send with custom generation config.
        /// </summary>
        public string Send(string message, GenerationConfig config)
        {
            ThrowIfDisposed();
            var nativeConfig = Chat.ToNativeConfig(config);
            Native.CheckError(Native.kjarni_chat_conversation_send_config(
                _handle, message, ref nativeConfig, out var resultPtr));
            return ConsumeString(resultPtr);
        }

        /// <summary>
        /// Stream a response. Message and full response are added to history.
        /// </summary>
        public void Stream(string message, Func<string, bool> onToken)
        {
            ThrowIfDisposed();
            _activeCallback = (textPtr, _) =>
            {
                var text = Marshal.PtrToStringUTF8(textPtr) ?? "";
                return onToken(text);
            };
            try
            {
                Native.CheckError(Native.kjarni_chat_conversation_stream(
                    _handle, message, IntPtr.Zero,
                    _activeCallback, IntPtr.Zero, IntPtr.Zero));
            }
            finally
            {
                _activeCallback = null;
            }
        }

        /// <summary>Number of messages in history.</summary>
        public int Length
        {
            get
            {
                ThrowIfDisposed();
                return (int)Native.kjarni_chat_conversation_len(_handle);
            }
        }

        /// <summary>Clear conversation history.</summary>
        public void Clear(bool keepSystem = true)
        {
            ThrowIfDisposed();
            Native.kjarni_chat_conversation_clear(_handle, keepSystem ? 1 : 0);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(ChatConversation));
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                if (_handle != IntPtr.Zero)
                {
                    Native.kjarni_chat_conversation_free(_handle);
                    _handle = IntPtr.Zero;
                }
            }
        }

        private static string ConsumeString(IntPtr ptr)
        {
            try
            {
                return ptr != IntPtr.Zero ? Marshal.PtrToStringUTF8(ptr) ?? "" : "";
            }
            finally
            {
                if (ptr != IntPtr.Zero)
                    Native.kjarni_string_free(ptr);
            }
        }
    }
}