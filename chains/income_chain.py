from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from tools.yahoo_api import get_income_statement
from utils.formatters import format_income_statement
from utils.extractors import extract_company_name, resolve_company_to_ticker
from utils.metric_extractor import extract_metric_query
from utils.llm_metric_mapper import resolve_metric_name

# 🧠 Use Groq-powered LLM
llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b")

# 📄 Prompt Template for full summary
prompt = PromptTemplate(
    input_variables=["company", "data"],
    template="""
You are a financial analyst.
Here is the income statement data for {company}:

{data}

Generate a 5-8 sentence summary of this company's financial performance over time, mentioning key trends, notable changes in revenue, net income, expenses, and margins.
Use bullet points if needed.
"""
)

# 🔗 LangChain wrapper for summary
summary_chain = LLMChain(llm=llm, prompt=prompt)


def summarize_income_statement(company: str, ticker: str) -> str:
    raw_data = get_income_statement(ticker)
    if "error" in raw_data:
        return f"❌ Could not fetch income data for {company}: {raw_data['error']}"

    compact = format_income_statement(raw_data)
    if not compact:
        return f"⚠️ Could not format income data for {company}."

    result = summary_chain.run({"data": compact, "company": company})
    return result



from difflib import get_close_matches
from tools.yahoo_api import get_income_statement
from utils.llm_metric_mapper import resolve_metric_name

def fuzzy_resolve_metric_key(user_metric: str, available_keys: list) -> str:
    lower_map = {k.lower(): k for k in available_keys}
    matches = get_close_matches(user_metric.lower(), lower_map.keys(), n=1, cutoff=0.6)
    return lower_map[matches[0]] if matches else user_metric



def extract_income_metric(ticker: str, user_metric: str, year: str, company: str = "") -> str:
    raw_data = get_income_statement(ticker)
    if "error" in raw_data:
        return f"❌ Could not fetch income data for {company or ticker}: {raw_data['error']}"

    print("🛠️ Raw Data Keys (Dates):", list(raw_data.keys()))

    year = str(year)
    llm_metric = resolve_metric_name(user_metric).strip()
    print(f"🔍 Metric from LLM: {llm_metric}")

    for date, values in raw_data.items():
        print(f"📅 {date} → Available Metrics: {list(values.keys())}")  # Debug print
        if year in str(date):
            actual_metric = fuzzy_resolve_metric_key(llm_metric, list(values.keys()))
            print(f"✅ Matched '{llm_metric}' → '{actual_metric}'")
            if actual_metric in values:
                value = values[actual_metric]
                return f"📊 {company or ticker}'s **{actual_metric}** in {year} was **${value:,.0f}**"

    return f"⚠️ '{llm_metric}' not found in Yahoo Finance data for {company or ticker}."

