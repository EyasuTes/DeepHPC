# embed_documents.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import os

# Load embedding model
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

all_texts = []
for filename in os.listdir("documents"):
    with open(os.path.join("documents", filename), "r") as f:
        text = f.read()
        chunks = splitter.split_text(text)
        all_texts.extend(chunks)

db = Chroma.from_texts(texts=all_texts, embedding=model, persist_directory="./chroma_db")
db.persist()
print('done')
