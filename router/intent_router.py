from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b")

router_prompt = PromptTemplate.from_template(
    """You are a financial assistant router. Classify the user's query into one of the following categories ONLY:
- financial_statements
- sec_filings
- news
- forecasting

Respond ONLY with the category name on a single line.

Query: {input}"""
)

router_chain = router_prompt | llm

def route_intent(user_query: str) -> str:
    response = router_chain.invoke({"input": user_query})
    output = response.content.strip().lower()

    # ✅ Extract the last line as the label
    lines = output.splitlines()
    return lines[-1] if lines else output




# intent_router.py
import re

def classify_intent(user_input: str) -> str:
    user_input = user_input.lower()

    # Year-specific fact lookup
    if re.search(r"\b(in|for)\s+20\d{2}\b", user_input) or re.search(r"\bshow|what was|report|revenue|net income|earnings per share|eps|expenses|r&d|operating income\b", user_input):
        return "lookup"

    # Comparisons
    if "compare" in user_input or "vs" in user_input or "difference" in user_input:
        return "comparison"

    # Forecast-related
    if "forecast" in user_input or "predict" in user_input:
        return "forecast"

    # Charts or visualization
    if "chart" in user_input or "plot" in user_input or "graph" in user_input:
        return "visual"

    # Default to summary
    return "summary"


# Example usage:
# intent = classify_intent("What was Tesla's net income in 2022?")
# -> "lookup"
