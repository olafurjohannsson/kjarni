using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Kjarni
{
    /// <summary>
    /// Parses the <c>metadata_json</c> string attached to each search result.
    ///
    /// The native side builds it with <c>serde_json::to_string</c> from a
    /// <c>HashMap&lt;String, String&gt;</c>, so it is always one flat object with string
    /// keys and string values. That is all this reads. It exists so the netstandard2.1
    /// build needs no System.Text.Json, which Unity does not ship and cannot pull from
    /// NuGet along with its dependency chain.
    /// </summary>
    internal static class MetadataJson
    {
        /// <summary>
        /// Returns the key/value pairs, or an empty dictionary if the input is not a flat
        /// object of strings. Never throws; a result without metadata is still a result.
        /// </summary>
        public static Dictionary<string, object> Parse(string json)
        {
            var result = new Dictionary<string, object>();
            int i = 0;

            SkipWhitespace(json, ref i);
            if (!Consume(json, ref i, '{')) return new Dictionary<string, object>();
            SkipWhitespace(json, ref i);
            if (Consume(json, ref i, '}')) return result;

            while (true)
            {
                SkipWhitespace(json, ref i);
                var key = ReadString(json, ref i);
                if (key == null) return new Dictionary<string, object>();

                SkipWhitespace(json, ref i);
                if (!Consume(json, ref i, ':')) return new Dictionary<string, object>();

                SkipWhitespace(json, ref i);
                var value = ReadString(json, ref i);
                if (value == null) return new Dictionary<string, object>();

                result[key] = value;

                SkipWhitespace(json, ref i);
                if (Consume(json, ref i, ',')) continue;
                if (Consume(json, ref i, '}')) return result;
                return new Dictionary<string, object>();
            }
        }

        private static void SkipWhitespace(string s, ref int i)
        {
            while (i < s.Length && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
                i++;
        }

        private static bool Consume(string s, ref int i, char c)
        {
            if (i < s.Length && s[i] == c)
            {
                i++;
                return true;
            }
            return false;
        }

        /// <summary>Reads a JSON string literal starting at <paramref name="i"/>, or returns null.</summary>
        private static string? ReadString(string s, ref int i)
        {
            if (!Consume(s, ref i, '"')) return null;

            var sb = new StringBuilder();
            while (i < s.Length)
            {
                char c = s[i++];
                if (c == '"') return sb.ToString();
                if (c != '\\')
                {
                    sb.Append(c);
                    continue;
                }

                if (i >= s.Length) return null;
                char esc = s[i++];
                switch (esc)
                {
                    case '"': sb.Append('"'); break;
                    case '\\': sb.Append('\\'); break;
                    case '/': sb.Append('/'); break;
                    case 'b': sb.Append('\b'); break;
                    case 'f': sb.Append('\f'); break;
                    case 'n': sb.Append('\n'); break;
                    case 'r': sb.Append('\r'); break;
                    case 't': sb.Append('\t'); break;
                    case 'u':
                        // A surrogate pair arrives as two consecutive \u escapes, and
                        // appending each UTF-16 unit in turn reassembles it.
                        if (i + 4 > s.Length) return null;
                        if (!ushort.TryParse(s.Substring(i, 4), NumberStyles.AllowHexSpecifier,
                                CultureInfo.InvariantCulture, out var unit))
                            return null;
                        sb.Append((char)unit);
                        i += 4;
                        break;
                    default:
                        return null;
                }
            }
            return null;
        }
    }
}
