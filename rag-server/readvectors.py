import pickle

with open("faiss_data.pkl", "rb") as f:
    data = pickle.load(f)

print("First chunk of text:")
print(data["texts"][0])
print("\nCorresponding embedding vector:")
print(data["embeddings"][0])
