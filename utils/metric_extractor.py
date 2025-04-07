from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Use a Groq-hosted LLM to parse metric extraction requests
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0)

prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a financial query parser. Extract the company name, year (if any), and financial metric from the input query.

Return a JSON object with the following keys:
- company (string)
- year (int or null)
- metric (string or null)

Input query: "{query}"

Only output JSON.
"""
)

from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)

def extract_metric_query(query: str):
    """Extract metric-related info from natural language query using LLM."""
    try:
        result = chain.run(query)
        parsed = eval(result) if isinstance(result, str) else result
        return {
            "company": parsed.get("company"),
            "year": parsed.get("year"),
            "metric": parsed.get("metric"),
            "metric_key": parsed.get("metric")  # Assuming formatters will map this
        }
    except Exception as e:
        return None
