from transformers import T5ForConditionalGeneration, AutoTokenizer

model_path = "/home/olafurj/.cache/kjarni/google_flan-t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

text = "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair."

# T5 summarization prefix
input_text = f"summarize: {text}"
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    num_beams=1,  # greedy
    do_sample=False,
)

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Summary: {summary}")