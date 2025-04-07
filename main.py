# main.py

from dotenv import load_dotenv
load_dotenv()

from chains.sec_rag_chain import sec_rag_chain
from router.llm_query_parser import parse_user_query


def handle_sec_query(user_input):
    response = sec_rag_chain.invoke({"query": user_input})
    answer = response["result"]

    sources = "\n\n".join(
        f"📄 Source Preview:\n{doc.page_content[:300]}..."
        for doc in response.get("source_documents", [])
    )

    return f"{answer}\n\n---\n{sources}" if sources else answer


from router.llm_query_parser import parse_user_query
from router.chain_orchestrator import orchestrate_chains

def run_assistant(user_input):
    parsed = parse_user_query(user_input)
    if "error" in parsed:
        return parsed["error"]

    print(f"🔍 Parsed: {parsed}")

    # Check if the query is related to SEC filings (e.g., 10-Q, 10-K)
    if parsed["intent"] == "search_filings" or "10-Q" in user_input or "10-K" in user_input:
        response = sec_rag_chain.invoke({"query": user_input})
        answer = response["result"]

        sources = "\n\n".join(
            f"📄 Source Preview:\n{doc.page_content[:300]}..."
            for doc in response.get("source_documents", [])
        )

        return f"{answer}\n\n---\n{sources}" if sources else answer

    # Otherwise, proceed with normal chain orchestration
    return orchestrate_chains(parsed)



if __name__ == "__main__":
    print("💼 Welcome to IntelliFin Analyst (Groq-Powered) 💼")
    print("Type 'exit' to quit.")

    while True:
        query = input("\n🧑‍💼 Ask something: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = run_assistant(query)
        print("\n🧠 Answer:\n", response)
