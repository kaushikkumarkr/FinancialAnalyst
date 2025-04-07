import yfinance as yf

def get_income_statement(ticker: str, period: str = "annual") -> dict:
    stock = yf.Ticker(ticker)
    try:
        df = stock.quarterly_financials if period == "quarterly" else stock.financials
        if df.empty:
            return {"error": "No financial data found."}
        return df.to_dict()
    except Exception as e:
        return {"error": str(e)}

def get_balance_sheet(ticker: str, period: str = "annual") -> dict:
    stock = yf.Ticker(ticker)
    try:
        df = stock.balance_sheet if period == "annual" else stock.quarterly_balance_sheet
        if df.empty:
            return {"error": "No balance sheet data found."}
        return df.to_dict()
    except Exception as e:
        return {"error": str(e)}

def get_cash_flow(ticker: str, period: str = "annual") -> dict:
    stock = yf.Ticker(ticker)
    try:
        df = stock.cashflow if period == "annual" else stock.quarterly_cashflow
        if df.empty:
            return {"error": "No cash flow data found."}
        return df.to_dict()
    except Exception as e:
        return {"error": str(e)}






# tools/yahoo_api.py
import requests
import pandas as pd

def fetch_income_statement(ticker: str, frequency: str = "annual") -> pd.DataFrame:
    """
    Fetch the income statement for a given ticker from Yahoo Finance.
    frequency: "annual" or "quarterly"
    Returns a pandas DataFrame.
    """
    try:
        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=incomeStatementHistory{'' if frequency == 'annual' else 'Quarterly'}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        key = "incomeStatementHistory" if frequency == "annual" else "incomeStatementHistoryQuarterly"
        income_data = data["quoteSummary"]["result"][0][key]["incomeStatementHistory" if frequency == "annual" else "incomeStatementHistory"]

        records = []
        for entry in income_data:
            row = {k: v.get("raw") if isinstance(v, dict) else v for k, v in entry.items()}
            records.append(row)

        df = pd.DataFrame(records)
        df["endDate"] = pd.to_datetime(df["endDate"], unit='s')
        df.set_index("endDate", inplace=True)
        return df.sort_index(ascending=False)

    except Exception as e:
        print(f"[YahooAPI] Failed to fetch income statement for {ticker}: {e}")
        return pd.DataFrame()



import requests

def resolve_company(query: str) -> tuple:
    """
    Uses Yahoo Finance's autocomplete endpoint to map fuzzy user input to (company name, ticker).
    """
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 1, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        result = data.get("quotes", [])[0]

        if "symbol" in result and "shortname" in result:
            return result["shortname"], result["symbol"]
        else:
            return "Unknown", None
    except Exception as e:
        print(f"[YahooAPI] Error resolving company name: {e}")
        return "Unknown", None





import yfinance as yf
import pandas as pd

def get_financial_metric(ticker: str, metric_name: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    fin = stock.financials.transpose()

    if metric_name not in fin.columns:
        raise ValueError(f"❌ Metric '{metric_name}' not found for {ticker}")

    df = fin[[metric_name]].dropna().reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'].dt.year, format='%Y')  # Convert to yearly
    return df
