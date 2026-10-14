// Types the compiler needs that netstandard2.1 does not ship. Internal, so they never
// collide with a consumer's own copies; net8.0 has the real ones and skips this file's
// first half.

#if NETSTANDARD2_1
namespace System.Runtime.CompilerServices
{
    /// <summary>Lets <c>init</c> accessors and records compile against netstandard2.1.</summary>
    internal static class IsExternalInit { }
}
#endif

namespace AOT
{
    /// <summary>
    /// Marks a static method that native code calls back into. Unity's IL2CPP looks
    /// for this attribute by name and generates a native entry point for each method
    /// carrying it; without one, passing the delegate to native code throws. Defined
    /// here so the library needs no reference to UnityEngine. Ignored on CoreCLR.
    /// </summary>
    [System.AttributeUsage(System.AttributeTargets.Method)]
    internal sealed class MonoPInvokeCallbackAttribute : System.Attribute
    {
        public MonoPInvokeCallbackAttribute(System.Type type) { }
    }
}
