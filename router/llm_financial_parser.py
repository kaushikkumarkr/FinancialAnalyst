# router/llm_financial_parser.py

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b")

parser_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a financial query interpreter for an AI analyst system.

Given this user query:
"{query}"

Extract the following:
- intent: summarize_statement, retrieve_metric, sec_filings, forecasting, or news
- company: The company name (e.g., Amazon)
- ticker: Stock ticker (e.g., AMZN)
- year: The year referenced (e.g., 2023), or null
- metric: e.g., "net income", "revenue", or null
- financial_type: "income_statement", "balance_sheet", or "cash_flow", or null

Respond in this JSON format:
{{
  "intent": "...",
  "company": "...",
  "ticker": "...",
  "year": 2023,
  "metric": "...",
  "financial_type": "..."
}}
"""
)

parser_chain = LLMChain(llm=llm, prompt=parser_prompt)

def parse_financial_query(query: str) -> dict:
    try:
        result = parser_chain.run({"query": query})
        parsed = eval(result.strip())
        return parsed
    except Exception as e:
        return {"error": f"❌ Could not parse query. Details: {str(e)}"}
