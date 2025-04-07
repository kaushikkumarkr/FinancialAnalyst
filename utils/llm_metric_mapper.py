# utils/llm_metric_mapper.py

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b")

prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a financial assistant. Extract the most likely financial metric name from this user query:
"{query}"

Respond with the **exact matching Yahoo Finance metric key** such as:
- Net Income
- Total Revenue
- Operating Income
- EBITDA
- Research Development
- Interest Expense
- Diluted EPS

Only return the metric key, nothing else.
"""
)

metric_mapper_chain = LLMChain(llm=llm, prompt=prompt)


def resolve_metric_name(user_query: str) -> str:
    result = metric_mapper_chain.run({"query": user_query})
    
    # Fix: Strip any think-aloud LLM output and isolate the actual metric
    if "```json" in result:
        # Guardrail: if the LLM returns a JSON block for some reason
        try:
            import re
            match = re.search(r'```json\n(.*?)\n```', result, flags=re.DOTALL)
            if match:
                result = match.group(1).strip()
        except:
            pass

    # Strip <think> block if returned by LLM
    if "<think>" in result:
        result = result.split("</think>")[-1].strip()

    # Final cleanup
    return result.strip().splitlines()[-1]

