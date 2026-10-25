using System.Collections.Generic;
using System.Text.Json;
using Xunit;

namespace Kjarni.Tests
{
    /// <summary>
    /// MetadataJson replaces System.Text.Json for search-result metadata. These pin it
    /// against what serde_json emits and against System.Text.Json's own reading.
    /// </summary>
    public class MetadataJsonTests
    {
        [Fact]
        public void Parses_serde_json_output()
        {
            // Verbatim serde_json::to_string of a HashMap<String, String>.
            var m = MetadataJson.Parse("{\"source\":\"docs/refunds.md\",\"chunk\":\"3\"}");

            Assert.Equal(2, m.Count);
            Assert.Equal("docs/refunds.md", m["source"]);
            Assert.Equal("3", m["chunk"]);
        }

        [Fact]
        public void Empty_object_is_empty()
        {
            Assert.Empty(MetadataJson.Parse("{}"));
        }

        [Theory]
        [InlineData("{\"k\":\"a\\\"b\"}", "a\"b")]
        [InlineData("{\"k\":\"a\\\\b\"}", "a\\b")]
        [InlineData("{\"k\":\"line\\nbreak\\ttab\\r\"}", "line\nbreak\ttab\r")]
        [InlineData("{\"k\":\"\\u0001\\u001f\"}", "\u0001\u001f")]
        [InlineData("{\"k\":\"\\b\\f\\/\"}", "\b\f/")]
        [InlineData("{\"k\":\"Reykjavík þú 東京\"}", "Reykjavík þú 東京")]
        [InlineData("{\"k\":\"\\ud83d\\ude00\"}", "\U0001F600")]
        public void Decodes_escapes(string json, string expected)
        {
            Assert.Equal(expected, MetadataJson.Parse(json)["k"]);
        }

        [Theory]
        [InlineData("")]
        [InlineData("not json")]
        [InlineData("[]")]
        [InlineData("{\"k\":1}")]
        [InlineData("{\"k\":\"unterminated}")]
        [InlineData("{\"k\":\"v\"")]
        [InlineData("{\"k\" \"v\"}")]
        [InlineData("{\"k\":\"\\x\"}")]
        [InlineData("{\"k\":\"\\u12\"}")]
        public void Malformed_input_yields_empty(string json)
        {
            Assert.Empty(MetadataJson.Parse(json));
        }

        [Fact]
        public void Agrees_with_System_Text_Json_on_awkward_strings()
        {
            var original = new Dictionary<string, string>
            {
                ["path"] = "C:\\docs\\\"quoted\".txt",
                ["control"] = "\u0000\u0007\u001b",
                ["unicode"] = "ísl \U0001F600 \u2028 \u2029",
                ["empty"] = "",
            };
            // System.Text.Json escapes more aggressively than serde_json (\u escapes for
            // non-ASCII), which exercises the \u path harder than the native side will.
            var json = JsonSerializer.Serialize(original);

            var parsed = MetadataJson.Parse(json);

            Assert.Equal(original.Count, parsed.Count);
            foreach (var kv in original)
                Assert.Equal(kv.Value, parsed[kv.Key]);
        }
    }
}
