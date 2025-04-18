# embed.py
import sys
import json
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


query = sys.argv[1]

# Load embedding model
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=model)

# Search top 3 matching chunks
docs = vectorstore.similarity_search(query, k=3)

# Output to stdout for Node.js to capture
print(json.dumps([doc.page_content for doc in docs]))
