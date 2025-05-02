# embed_documents.py
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# data cleaning before spliting
# def clean_text(text):
#     return " ".join(text.strip().split())


# Load and split documents
all_texts = []
# for filename in os.listdir("documents"):
#     with open(os.path.join("documents", filename), "r", encoding="utf-8", errors="ignore") as f:
#         text = f.read()
#         chunks = splitter.split_text(text)
#         all_texts.extend(chunks)

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

# Generate embeddings
embeddings = model.encode(all_texts, normalize_embeddings=True)
# Generate embeddings with batching

# embeddings = model.encode(all_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
# embeddings = np.array(embeddings, dtype=np.float32)


# Save as pickle for use in FAISS
with open("faiss_data.pkl", "wb") as f:
    pickle.dump({
        "texts": all_texts,
        "embeddings": np.array(embeddings)
    }, f)

print("✅ FAISS-compatible data saved.")
