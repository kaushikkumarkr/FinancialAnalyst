from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from tools.yahoo_api import get_balance_sheet

def balance_sheet_chain():
    llm = ChatGroq(temperature=0.3, model_name="deepseek-r1-distill-llama-70b")
    prompt = PromptTemplate.from_template(
        "Here is the actual balance sheet data for {company}:\n\n{data}\n\n"
        "Summarize key trends in assets, liabilities, and equity. Mention any notable changes over time."
    )
    return LLMChain(llm=llm, prompt=prompt)

def summarize_balance_sheet(company: str, ticker: str, period="annual"):
    data = get_balance_sheet(ticker, period)
    if "error" in data:
        return f"❌ {data['error']}"
    compact = {k: {yr: str(v) for yr, v in vals.items()} for k, vals in data.items()}
    chain = balance_sheet_chain()
    return chain.run({"company": company, "data": compact})
