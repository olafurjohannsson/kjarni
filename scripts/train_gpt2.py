from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load base model
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

# Your dataset format: {"text": "[PERSONALITY: grumpy_blacksmith] Player: I need a sword\nNPC: Bah! Another adventurer..."}
dataset = load_dataset("json", data_files="npc_dialogue.jsonl")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./npc-gpt2-medium",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    learning_rate=5e-5
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized["train"])
trainer.train()
trainer.save_model("./npc-dialogue-gpt2")