print("hello 1")
import sys
print("hello 2")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AdapterConfig

print("hello 3")

# Load model and tokenizer
model_path = "./fine_tuned_deepseek"
adapter_config = AdapterConfig.load(model_path + "/adapter_config.json")
# Ensure support for custom/model-specific code
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("hello 4 - tokenizer loaded")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    attn_implementation="sdpa"  # safer default than "eager"
)
# print("hello 5 - model loaded")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()
# print("hello 6 - model moved to", device)

# # Get inputs
# question = sys.argv[1]
# context = sys.argv[2]

# prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
# print("📥 Prompt created")

# inputs = tokenizer(prompt, return_tensors="pt").to(device)
# print("📏 Token count:", len(inputs["input_ids"][0]))

# # Generate response
# with torch.no_grad():
#     outputs = model.generate(**inputs, max_new_tokens=150)

# response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Optional: Only print the answer (strip the prompt part if needed)
# print("✅ Response generated")
# print(response)
