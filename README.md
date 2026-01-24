# Kjarni

Run machine learning models from Rust, Python, C#, Go, and C++.

Kjarni is an inference engine designed to run locally without external dependencies. Single shared core written in Rust with multiple FFI language bindings, shipped as a single binary.

## Examples

### Classification

### CLI

```bash
kjarni classify --format json --model distilbert-sentiment "i love Kjarni" 
```
```bash
{
  "label": "POSITIVE",
  "label_index": 0,
  "predictions": [
    {
      "label": "POSITIVE",
      "score": 0.9998016
    },
    {
      "label": "NEGATIVE",
      "score": 0.0001984584
    }
  ],
  "score": 0.9998016,
  "text": "i love kjarni"
}
```
### Python

```python
from kjarni import Classifier

classifier = Classifier("distilbert-sentiment")
result = classifier.classify("i love kjarni")
print(result.all_scores)
```

```bash
[('POSITIVE', 0.9998015761375427), ('NEGATIVE', 0.0001984583941521123)]
```

### C#

```csharp
using Kjarni;

using var classifier = new Classifier("distilbert-sentiment");
var result = classifier.Classify("I love kjarni");

foreach (var (label, score) in result.AllScores)
{
    Console.WriteLine($"{label}: {score:F4}");
}
```
```bash
POSITIVE: 0.99980158
NEGATIVE: 0.00019846
```


### Golang


### C++


### Rust