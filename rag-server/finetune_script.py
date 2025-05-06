import markdown
import os
import torch
from datasets import Dataset
from bs4 import BeautifulSoup
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# 1️⃣ Load Markdown Documents
docs_folder = "ulhpc_manuals"
documents = []

for file_name in os.listdir(docs_folder):
    if file_name.endswith(".md"):
        with open(os.path.join(docs_folder, file_name), "r", encoding="utf-8") as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            plain_text = BeautifulSoup(html_text, "html.parser").get_text()
            documents.append({"text": plain_text.strip().replace("\n", " ")})

dataset = Dataset.from_list(documents)
print(f"✅ Loaded {len(dataset)} documents")
print(dataset[0])

# 2️⃣ Load Tokenizer and Model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def format_dataset(batch):
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    return {
        "input_ids": [ids.tolist() for ids in tokenized["input_ids"]],
        "attention_mask": [mask.tolist() for mask in tokenized["attention_mask"]],
        "labels": [ids.tolist() for ids in tokenized["input_ids"]]
    }

train_dataset = dataset.map(format_dataset, batched=True)

# 3️⃣ Load Model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# 4️⃣ Apply LoRA (optional but compatible)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 5️⃣ Training Config
training_args = TrainingArguments(
    output_dir="./fine_tuned_deepseek",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    num_train_epochs=3,
    save_steps=100,
    logging_steps=50,
    eval_strategy="no",
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

print("🚀 Starting fine-tuning...")
trainer.train()
print("✅ Fine-tuning completed!")

# 6️⃣ Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_deepseek")
tokenizer.save_pretrained("./fine_tuned_deepseek")
print("✅ Model saved to './fine_tuned_deepseek'")

# 7️⃣ Test Inference
print("\n🚀 Testing Model...")
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_deepseek")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_deepseek")

prompt = "What is ULHPC?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)

print("\n🤖 Answer:", tokenizer.decode(outputs[0], skip_special_tokens=True))