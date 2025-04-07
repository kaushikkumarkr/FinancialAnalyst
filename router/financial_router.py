def route_financial_type(user_query: str):
    query = user_query.lower()
    if "income" in query:
        return "income_chain"
    elif "balance" in query:
        return "balance_chain"
    elif "cash" in query or "cash flow" in query:
        return "cashflow_chain"
    else:
        return "default_chain"
