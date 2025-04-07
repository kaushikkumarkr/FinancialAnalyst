# router/llm_query_parser.py

import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

from tools.yahoo_api import resolve_company

llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b")

parser_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are an intelligent financial assistant. Based on the user query, extract the following as JSON:
- company (or best guess),
- financial type (income_statement, balance_sheet, cash_flow),
- intent (retrieve_metric, summarize_statement, forecasting, sec_filings, news),
- metric (like revenue, net income),
- year (if any)
- filing_type (either "10-K", "10-Q", "8-K" or null)
- periods (number of years to forecast, if applicable, otherwise null).

Return a valid JSON object with keys: company, financial_type, intent, metric, year.

User query: "{query}"
"""
)

parser_chain = LLMChain(llm=llm, prompt=parser_prompt)


import json
import re

def parse_user_query(query: str) -> dict:
    try:
        raw_output = parser_chain.run({"query": query})
        print("🧾 LLM Raw Output:", raw_output)

        # Extract JSON block using regex
        json_match = re.search(r"\{.*?\}", raw_output, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON block found.")

        json_str = json_match.group(0)
        parsed = json.loads(json_str)

        # Resolve company name and ticker
        company, ticker = resolve_company(parsed.get("company", ""))
        parsed["company"] = company
        parsed["ticker"] = ticker

        return parsed

    except Exception as e:
        return {"error": f"❌ Failed to parse query: {str(e)}"}

