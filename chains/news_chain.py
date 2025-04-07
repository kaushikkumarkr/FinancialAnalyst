from tools.news_api import fetch_company_news
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0)

news_prompt = PromptTemplate.from_template("""
You are a financial news analyst assistant. Given the following recent news articles related to {company}, summarize the most relevant and timely updates based on the user's query.

User Query:
{query}

News Articles:
{articles}

Provide a clear, professional summary, including dates and context if available.
""")


news_chain = LLMChain(llm=llm, prompt=news_prompt)

def get_news_summary(company: str):
    articles = fetch_company_news(company)
    articles_text = "\n\n".join([f"- {a['title']}: {a['description']}" for a in articles])
    return news_chain.run({"company": company, "articles": articles_text})
