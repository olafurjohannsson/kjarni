from transformers import BartForConditionalGeneration, BartTokenizer, LogitsProcessor, LogitsProcessorList
import torch

# --- 1. Define a Debug Processor to visualize steps ---
class BeamDebugProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.step_count = 0

    def __call__(self, input_ids, scores):
        # input_ids shape: [num_beams, current_sequence_length]
        print(f"\n--- Step {self.step_count} ---")
        
        # We assume batch size 1 for clarity
        for beam_idx, beam_tokens in enumerate(input_ids):
            # Decode to string
            decoded = self.tokenizer.decode(beam_tokens, skip_special_tokens=False)
            # Clean up newlines for cleaner printing
            decoded_clean = decoded.replace('\n', ' ')
            
            # Print raw IDs (last 5) and decoded text
            last_ids = beam_tokens[-5:].tolist()
            print(f"Beam {beam_idx}: ...{last_ids} | \"{decoded_clean}\"")
            
        self.step_count += 1
        return scores

# --- 2. Setup Model ---
model_name = "sshleifer/distilbart-cnn-12-6"
print(f"Loading {model_name}...")
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

text = """Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without using a garbage collector. To simultaneously enforce memory safety and prevent data races, its "borrow checker" tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages like C++, Haskell, and Erlang."""

input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

# --- 3. Run Generation with Debugger ---
print("\n=== GENERATING WITH DEBUG TRACE ===")

# Create the debug processor
debug_processor = BeamDebugProcessor(tokenizer)

# We must ensure we pass the same config as your Rust test
summary_ids = model.generate(
    input_ids,
    num_beams=4,
    length_penalty=2.0,
    max_length=142,
    min_length=56,
    no_repeat_ngram_size=3,
    early_stopping=True,
    logits_processor=LogitsProcessorList([debug_processor]) # Inject here
)

output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("\n=== FINAL OUTPUT ===")
print(output_text)