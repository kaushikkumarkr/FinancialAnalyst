from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from tools.yahoo_api import get_cash_flow


def cash_flow_chain():
    llm = ChatGroq(temperature=0.3, model_name="deepseek-r1-distill-llama-70b")
    prompt = PromptTemplate.from_template(
        """
        You are a financial analyst.
        Here is the actual cash flow statement data for {company}:

        {data}

        Summarize the cash flow activities. Focus on:
        - Operating cash flow trends
        - Investing cash flow patterns
        - Financing cash flow shifts
        - Any notable anomalies or observations across years.
        """
    )
    return LLMChain(llm=llm, prompt=prompt)


def summarize_cash_flow(company: str, ticker: str, period="annual"):
    data = get_cash_flow(ticker, period)
    if "error" in data:
        return f"❌ {data['error']}"

    compact = {k: {yr: str(v) for yr, v in vals.items()} for k, vals in data.items()}
    chain = cash_flow_chain()
    return chain.run({"company": company, "data": compact})
