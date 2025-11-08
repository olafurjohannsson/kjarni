from transformers import pipeline, AutoConfig

model_name = "sshleifer/distilbart-cnn-12-6"

# Load the configuration object from the Hub
# This fetches the config.json file and parses it.
print(f"Loading configuration for model: {model_name}...")
config = AutoConfig.from_pretrained(model_name)
print("✓ Configuration loaded.")

# Print the configuration object
# The object's __str__ method provides a clean, readable output.
print("\n--- Model Configuration ---")
print(config)
print("--------------------------")

# The article to be summarized
article = """Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, \
type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without \
using a garbage collector. To simultaneously enforce memory safety and prevent data races, its 'borrow checker' \
tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages \
like C++, Haskell, and Erlang."""

# Initialize the summarization pipeline with the specified model
# This will download the model on the first run
print("Loading DistilBART model...")
summarizer = pipeline("summarization", model=model_name, device="cpu")
print("✓ Model loaded.")

# Generate the summary with a max length of 60 tokens
print("\nGenerating summary...")
summary = summarizer(article)

print("\n--- ARTICLE ---")
print(article)

print("\n--- GENERATED SUMMARY (PyTorch) ---")
print(summary[0]['summary_text'])