# embed_documents.py
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


all_texts = []


for root, dirs, files in os.walk("documents"):
    for filename in files:
        if filename.lower().endswith(".md"):
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    chunks = splitter.split_text(text)
                    all_texts.extend(chunks)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")


embeddings = model.encode(all_texts, normalize_embeddings=True)

with open("faiss_data.pkl", "wb") as f:
    pickle.dump({
        "texts": all_texts,
        "embeddings": np.array(embeddings)
    }, f)

print("✅ FAISS-compatible data saved.")
