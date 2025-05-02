# embed.py
import sys
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Read input
query = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 5  # Default top_k = 5 //optimze

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load precomputed vectors and texts
import pickle
with open("faiss_data.pkl", "rb") as f:
    data = pickle.load(f)

texts = data["texts"]
embeddings = data["embeddings"]
dimension = embeddings.shape[1]

# Build FAISS index
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, 20, faiss.METRIC_INNER_PRODUCT) # Optimized nlist
index.train(embeddings)
index.add(embeddings)

# TUNE nprobe here
index.nprobe = 5  # EXPERIMENT 2: Try different values e.g., 1, 5, 10

# Encode query
query_vector = model.encode([query], normalize_embeddings=True)
D, I = index.search(query_vector, k)

# Return top-k matching texts
results = [texts[i] for i in I[0]]
print(json.dumps(results))
