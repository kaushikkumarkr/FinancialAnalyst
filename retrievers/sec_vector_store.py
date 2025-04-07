import os
# ✅ Updated imports
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Persistent DB path
CHROMA_DIR = "vector_db/sec_filings"

def load_and_split_sec_file(file_path: str, metadata: dict):
    loader = TextLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Add metadata like company, year, filing_type
    for chunk in chunks:
        chunk.metadata.update(metadata)

    return chunks

def build_vectorstore(docs):
    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=CHROMA_DIR
    )
    db.persist()
    return db

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding
    )

vectorstore = Chroma(
    persist_directory="sec_vectorstore/",
    embedding_function=embedding,
)
print("📁 Documents in vectorstore:", vectorstore._collection.count())



embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = "vector_db/sec_filings"

def get_sec_retriever_by_year(ticker: str, year: str):
    retriever = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_function
    ).as_retriever(search_kwargs={
        "filter": {
            "ticker": ticker.upper(),
            "year": year
        },
        "k": 5
    })
    return retriever
