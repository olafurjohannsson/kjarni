# Content API — AI-Powered Moderation & Search

A production-ready ASP.NET Minimal API with sentiment analysis, content moderation, and semantic search. No Python, no cloud APIs, no API keys.

## Run

```bash
dotnet run
```

Models download automatically on first run (~850MB total, cached after that).

## Endpoints

### Sentiment Analysis

```bash
curl -X POST http://localhost:5000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

```json
{
  "label": "POSITIVE",
  "score": 0.9998,
  "all_scores": { "POSITIVE": 0.9998, "NEGATIVE": 0.0002 }
}
```

### Content Moderation

```bash
curl -X POST http://localhost:5000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "You are an absolute idiot"}'
```

```json
{
  "flagged": true,
  "toxic_score": 0.85,
  "scores": { "toxic": 0.85, "insult": 0.72, "obscene": 0.12, ... },
  "recommendation": "block"
}
```

### Semantic Search

```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how do I get my money back?", "topK": 3}'
```

```json
{
  "query": "how do I get my money back?",
  "results": [
    { "text": "Refunds are processed within 5-7 business days...", "score": 0.72 },
    { "text": "Returns are accepted within 30 days...", "score": 0.68 },
    { "text": "We offer a price match guarantee...", "score": 0.45 }
  ]
}
```

### Batch Sentiment

```bash
curl -X POST http://localhost:5000/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible service.", "It was okay."]}'
```

### Add Documents

```bash
curl -X POST http://localhost:5000/index \
  -H "Content-Type: application/json" \
  -d '{"text": "Express shipping is available for an additional $9.99."}'
```

## What This Demonstrates

- **Three AI models** running simultaneously on CPU, no GPU required
- **Sub-10ms inference** per request after model warmup
- **Zero external dependencies** — no API keys, no cloud calls, no Python
- **Production patterns** — health checks, batch endpoints, semantic search with dynamic indexing

## Cost Comparison

| Approach | Cost per 1M requests |
|----------|---------------------|
| OpenAI API | ~$600 |
| Azure AI | ~$400 |
| AWS Comprehend | ~$300 |
| **Kjarni (self-hosted)** | **$0** |

All inference runs locally. Your data never leaves your server.