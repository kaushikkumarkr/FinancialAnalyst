# router/chain_orchestrator.py

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

from chains.income_chain import summarize_income_statement, extract_income_metric
from chains.balance_sheet_chain import summarize_balance_sheet
from chains.cashflow_chain import summarize_cash_flow
from chains.sec_rag_chain import sec_rag_chain
from chains.news_chain import news_chain
from tools.news_api import fetch_company_news
import re
import json

from tools.news_api import fetch_company_news
from chains.news_chain import news_chain
from datetime import datetime, timedelta

# LLM #2 to decide which chain(s) to call
llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b")

router_prompt = PromptTemplate(
    input_variables=["parsed"],
    template="""
You are an expert financial analyst assistant.

A user query has already been parsed into structured data:
{parsed}

Based on this, decide which data source or chain to invoke:
- "income_chain"
- "balance_chain"
- "cashflow_chain"
- "forecast_chain"
- "sec_chain"
- "news_chain"

Also determine the correct sub-intent:
- "summarize_statement"
- "retrieve_metric"
- "forecast"
- "search_filings"
- "summarize_news"

Rules:
- If intent is "summarize_statement", then sub_intent must be "summarize_statement".
- If a metric is provided, use "retrieve_metric" unless intent is forecast or filings.
- If the intent is "forecasting", always include "forecast_chain" in the chains.

Return a JSON like:
{{
  "chains": ["forecast_chain"],
  "sub_intent": "forecast"
}}
Only return valid JSON.
"""
)


router_chain = LLMChain(llm=llm, prompt=router_prompt)



from chains.sec_rag_chain import sec_rag_chain

def summarize_sec_filing(company: str, year: int = None, topic: str = ""):
    query_parts = [company, topic, str(year) if year else ""]
    full_query = " ".join(filter(None, query_parts)).strip()
    print(f"🔍 SEC Summary Query: {full_query}")
    
    result = sec_rag_chain.invoke({"query": full_query})
    sources = "\n\n".join(
        f"📄 Source Preview:\n{doc.page_content[:300]}..."
        for doc in result.get("source_documents", [])
    )
    
    if not result.get("result"):
        return f"❌ No content found in SEC filings for '{full_query}'."
    
    return f"{result['result']}\n\n---\n{sources}" if sources else result["result"]

from chains.sec_rag_chain import sec_rag_chain

def retrieve_sec_insight(company: str, year: int = None, topic: str = ""):
    query_parts = [f"{company}", topic, str(year) if year else ""]
    final_query = " ".join(filter(None, query_parts)).strip()

    print(f"🔍 SEC Query via RAG: {final_query}")
    result = sec_rag_chain.invoke({"query": final_query})

    if not result.get("result"):
        return f"⚠️ No insights found from SEC filings for '{final_query}'."

    sources = "\n\n".join(
        f"📄 Source Preview:\n{doc.page_content[:300]}..." for doc in result.get("source_documents", [])
    )
    return f"{result['result']}\n\n---\n{sources}" if sources else result["result"]





def summarize_news(company: str, query: str):
    # Optional: parse for keywords like "today", "this morning", etc.
    now = datetime.now()
    if any(phrase in query.lower() for phrase in ["this morning", "today", "now"]):
        from_date = now.strftime("%Y-%m-%d")
    elif "yesterday" in query.lower():
        from_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        from_date = None

    articles = fetch_company_news(company_name=company, custom_from_date=from_date or None)
    if not articles:
        return "❌ No recent news found."

    # Format for LLM summarization
    article_texts = [
        f"- {a['title']} ({a['publishedAt'][:10]})\n  {a['description'] or 'No description.'}" for a in articles
    ]
    formatted_articles = "\n".join(article_texts)

    result = news_chain.invoke({
        "company": company,
        "articles": formatted_articles
    })

    return result.content




def orchestrate_chains(parsed: dict) -> str:

    router_decision = router_chain.invoke({"parsed": str(parsed)})

    if isinstance(router_decision, dict) and "text" in router_decision:
        from re import search
        print("📨 Router Raw Output:", router_decision)
        match = search(r"```json\n(.*?)\n```", router_decision["text"], flags=re.DOTALL)
        if match:
            try:
                decision = json.loads(match.group(1))
                print("✅ Parsed Decision:", decision)
            except Exception as e:
                return f"❌ Failed to parse routing decision JSON:\n{match.group(1)}\n\nError: {e}"
        else:
            return "❌ Could not extract valid JSON block from router LLM output."    
        
    elif isinstance(router_decision, dict):
        decision = router_decision
    else:
        return f"❌ Unexpected routing output type: {type(router_decision)}"    

    chains = decision.get("chains", [])
    sub_intent = decision.get("sub_intent", "")

    if parsed.get("intent") == "summarize_statement":
        sub_intent = "summarize_statement"

    company = parsed.get("company")
    ticker = parsed.get("ticker")
    year = parsed.get("year")
    metric = parsed.get("metric")

    # Single Chain Routing
    if sub_intent == "summarize_statement":
        if "income_chain" in chains:
            return summarize_income_statement(company=company, ticker=ticker)
        elif "balance_chain" in chains:
            return summarize_balance_sheet(company=company, ticker=ticker)
        elif "cashflow_chain" in chains:
            return summarize_cash_flow(company=company, ticker=ticker)
        elif "sec_chain" in chains:
            return summarize_sec_filing(company=company, year=year, topic="business strategy")

    elif sub_intent == "retrieve_metric":
        if "income_chain" in chains:
            return extract_income_metric(ticker, metric, year, company)
        elif "balance_chain" in chains:
            return extract_income_metric(ticker, metric, year, company)
        elif "cashflow_chain" in chains:
            return extract_income_metric(ticker, metric, year, company)
        elif "sec_chain" in chains:
            return retrieve_sec_insight(company=company, year=year, topic=metric or "")
        else:
            return f"❌ No matching chain found for sub_intent: {sub_intent}"
        

    if sub_intent == "forecast":
        if "forecast_chain" in chains:
            try:
                from chains.forecast_chain import ForecastChain
                forecast_chain = ForecastChain(llm)
                forecast_result = forecast_chain.forecast(
    company=parsed["company"],
    ticker=parsed["ticker"],
    metric=parsed["metric"],
    periods=int(parsed["periods"])
)
                return forecast_result
            except Exception as e:
                return f"❌ Forecasting failed: {e}"
        else:
            return "⚠️ Forecast chain not available."    

    elif sub_intent  in ["search_filings", "retrieve_metric"]:
        if "sec_chain" in chains:
            query_parts = [company or "", metric or ""]
            if year:
                query_parts.append(str(year))
            full_query = " ".join(query_parts).strip()
            print(f"🔍 SEC Query: {full_query}")
            result = sec_rag_chain.invoke({"query": full_query})    


            sources = "\n\n".join(
                f"📄 Source Preview:\n{doc.page_content[:300]}..."
                for doc in result.get("source_documents", [])
            )

            if not result.get("result"):
                return f"❌ No relevant information found in SEC filings for '{full_query}'."

            return f"{result['result']}\n\n---\n{sources}" if sources else result['result']

    elif sub_intent == "summarize_news":
        if "news_chain" in chains:
            user_query = parsed.get("raw_query", f"Latest news about {company}")
            now = datetime.now()

            if any(kw in user_query.lower() for kw in ["this morning", "today", "now", "recent"]):
                from_date = now.strftime("%Y-%m-%d")
            elif "yesterday" in user_query.lower():
                from_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                from_date = None


            articles = fetch_company_news(company_name=company, custom_from_date=from_date)
            if not articles:
                return "❌ Could not fetch news or no headlines available."
            formatted_articles = "\n".join(
            f"- {a['title']} ({a['publishedAt'][:10]})\n  {a['description'] or 'No description.'}"
            for a in articles
        )
            response = news_chain.invoke({
            "company": company,
            "articles": formatted_articles,
            "query": user_query
        })
            return response.content if hasattr(response, "content") else response

    return "⚠️ No appropriate chain could be executed."
