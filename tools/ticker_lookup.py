# tools/ticker_lookup.py
import requests

def extract_company_and_ticker(user_input):
    """
    Takes user query like 'Summarize income statement for Tesla' or 'Show revenue of AAPL'
    and returns a tuple (company_name, ticker_symbol) using Yahoo Finance autocomplete.
    """
    base_url = "https://query2.finance.yahoo.com/v1/finance/search"
    query = user_input.split(" for ")[-1].strip().upper()  # Grab last part as the potential name/ticker

    response = requests.get(base_url, params={"q": query, "quotesCount": 1, "newsCount": 0})
    if response.status_code == 200:
        data = response.json()
        if data["quotes"]:
            best_match = data["quotes"][0]
            return best_match.get("shortname"), best_match.get("symbol")
    return None, None



