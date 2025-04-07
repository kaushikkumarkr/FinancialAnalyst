# chains/sec_rag_chain.py
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

CHROMA_DIR = "vector_db/sec_filings"

# Create retriever with default parameters
retriever = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
).as_retriever(search_kwargs={"k": 6})

# LLM (Groq)
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0)

# Custom Prompt for synthesizing RAG answer
rag_prompt = PromptTemplate.from_template("""
You are a professional financial analyst assistant. Use the context below to answer the user's question accurately and concisely.

Context:
---------
{context}

Question:
---------
{question}

Answer in a clear, professional tone.
""")

# RAG Chain using RetrievalQA
sec_rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True
)

# ✅ RAG Query Handler with metadata filtering
def get_sec_summary(query: str, filing_type: str = None, year: int = None):
    filters = {}
    if filing_type:
        filters["filing_type"] = {"$eq": filing_type}
    if year:
        filters["filing_year"] = {"$eq": str(year)}

    return sec_rag_chain.invoke({
        "query": query,
        "search_kwargs": {
            "k": 6,
            "filter": filters
        }
    })
