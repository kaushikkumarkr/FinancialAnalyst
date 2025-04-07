from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

prompt = PromptTemplate.from_template(
    "The user asked to forecast the metric '{raw_metric}'. Based on common financial terms, return the exact metric name from Yahoo Finance financials that matches it. Example: 'income' → 'Net Income', 'revenue' → 'Total Revenue'. Only respond with the name."
)

normalize_chain = prompt | llm | RunnableLambda(lambda x: x.strip())

def normalize_metric_name(raw_metric: str) -> str:
    return normalize_chain.invoke({"raw_metric": raw_metric})
