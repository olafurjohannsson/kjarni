"""Fixed inputs shared by every engine, so they all do identical work."""

QUERY = "how do I get a refund for a damaged order?"

_SUBJECTS = ["The refund policy", "Our delivery service", "The account settings page", "Customer support",
             "The warranty", "International shipping", "The mobile app", "Gift cards",
             "The loyalty programme", "Password recovery", "Order tracking", "Returns at the store"]
_CLAUSES = ["covers items that arrive damaged or are not as described",
            "applies within thirty days of the delivery date",
            "requires the original receipt or the order number",
            "is handled by the regional team during business hours",
            "can be changed at any time from your profile",
            "does not apply to perishable goods or opened software",
            "sends a confirmation email once the request is processed",
            "may take five to ten working days to reach your bank"]


def documents(n=64, sentences=5):
    """n distinct documents of roughly 100-130 tokens each: a typical RAG chunk."""
    docs = []
    for i in range(n):
        parts = []
        for j in range(sentences):
            s = _SUBJECTS[(i + j) % len(_SUBJECTS)]
            c = _CLAUSES[(i * 3 + j) % len(_CLAUSES)]
            parts.append(f"{s} {c}.")
        docs.append(f"Section {i}: " + " ".join(parts))
    return docs


REVIEWS = [
    "I absolutely love this product, it works perfectly.",
    "Terrible experience, the package arrived broken and support never answered.",
    "It's okay, does the job but nothing special.",
    "Best purchase I've made all year, highly recommend it to everyone.",
    "The battery dies within an hour, very disappointed.",
    "Fast delivery and the quality exceeded my expectations.",
    "Not worth the money at all.",
    "Pretty good overall, a few minor issues with the setup.",
]


def reviews(n=64):
    return [f"{REVIEWS[i % len(REVIEWS)]} (order {i})" for i in range(n)]
