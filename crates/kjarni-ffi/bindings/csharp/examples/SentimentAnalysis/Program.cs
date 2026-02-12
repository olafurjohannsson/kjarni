using Kjarni;

using var classifier = new Classifier("distilbert-sentiment");

string[] reviews = [
    "This product exceeded all my expectations!",
    "Terrible quality, broke after one day.",
    "It's okay, nothing special.",
    "Absolutely love it, best purchase this year!",
    "Worst customer service I've ever experienced.",
];

foreach (var review in reviews)
{
    var result = classifier.Classify(review);
    Console.WriteLine($"{result.Label,8} ({result.Score:P0})  {review}");
}