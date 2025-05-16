import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure the inputs are read properly
try:
    question = sys.argv[1]
    context = sys.argv[2]
except Exception as e:
    print(json.dumps({"error": f"Error reading inputs: {str(e)}"}))
    sys.exit(1)

print(f"Question: {question}")
print(f"Context: {context}")

# Define the paths for the model and tokenizer
model_path = "model-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer_path = "tokenizer-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load the tokenizer and model from the local directory
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
except Exception as e:
    print(json.dumps({"error": f"Error loading model: {str(e)}"}))
    sys.exit(1)

# Create the prompt with context and the question
prompt = f"""You are a factual assistant. Only use the provided context. Do not show reasoning steps, internal thoughts, or anything before the answer. Just give a direct, final answer.

Context:
{context}

Question: {question}
Answer:"""

# Tokenize and generate the response
try:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=556000,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the answer from the generated text
    answer = decoded.split("Answer:")[-1].strip()

    # Output the answer as JSON
    print(json.dumps({"DeepSeek Answer": answer}))

except Exception as e:
    print(json.dumps({"error": f"Error during generation: {str(e)}"}))
    sys.exit(1)
