import requests
import re

YF_AUTOCOMPLETE_URL = "https://query2.finance.yahoo.com/v1/finance/search"

def extract_company_name(user_input: str) -> tuple[str, str]:
    """
    Resolve a fuzzy company name to its official name and ticker using Yahoo Finance's autocomplete API.
    Returns (company_name, ticker).
    """
    query = user_input.strip().split(" ")[-1]  # crude fallback
    params = {"q": query, "quotesCount": 1, "newsCount": 0}
    response = requests.get(YF_AUTOCOMPLETE_URL, params=params)

    if response.status_code != 200:
        return "Unknown", "UNKNOWN"

    data = response.json()
    if not data.get("quotes"):
        return "Unknown", "UNKNOWN"

    best_match = data["quotes"][0]
    name = best_match.get("shortname") or best_match.get("longname") or "Unknown"
    ticker = best_match.get("symbol", "UNKNOWN")

    print(f"🔎 Resolved '{query}' to {name} ({ticker})")
    return name, ticker


def extract_year_from_query(query: str) -> str | None:
    """
    Extract a 4-digit year (e.g., 2023) from a user query.
    Returns the year as a string, or None if not found.
    """
    match = re.search(r"\b(20\d{2})\b", query)
    if match:
        return match.group(1)
    return None

def resolve_company_to_ticker(query: str) -> tuple[str, str] | None:
    """
    Uses Yahoo Finance's autocomplete API to resolve user input to a ticker symbol.
    """
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=1&newsCount=0"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers).json()
        if response.get("quotes"):
            best = response["quotes"][0]
            return best["shortname"], best["symbol"]
    except Exception as e:
        print("Ticker resolution failed:", e)
    return None
