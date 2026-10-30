// The C# binding. Prints one line of JSON.
using System.Diagnostics;
using System.Globalization;
using Kjarni;

const int Warmup = 200, Calls = 3000;
const string Text = "how do I get a refund for a damaged order?";

using var embedder = new Embedder("minilm-l6-v2", quiet: true);
var times = new double[Calls];
for (var i = 0; i < Warmup + Calls; i++)
{
    var start = Stopwatch.GetTimestamp();
    embedder.Encode(Text);
    if (i >= Warmup) times[i - Warmup] = (Stopwatch.GetTimestamp() - start) * 1e6 / Stopwatch.Frequency;
}
Array.Sort(times);
Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
    $"{{\"lang\": \"C#\", \"median_us\": {times[Calls / 2]:F1}, \"min_us\": {times[0]:F1}}}"));
