# fetchers/sec_filings.py
from sec_edgar_downloader import Downloader
import os

# Provide email, company name, and download folder
dl = Downloader(
    email_address="kk795@njit.edu",
    company_name="RAG Analyst Assistant",  # Or any project name
    download_folder="sec_filings/"
)


def get_latest_sec_filing(ticker, form_type="10-K"):
    try:
        dl.get(form_type, ticker, limit=5)  # Download up to 5 recent filings
        path = f"sec_filings/sec-edgar-filings/{ticker}/{form_type}"
        if not os.path.exists(path):
            return f"No {form_type} filings directory found for {ticker}."

        folders = sorted(os.listdir(path))
        for subfolder in reversed(folders):  # Check most recent first
            folder_path = os.path.join(path, subfolder)
            files = os.listdir(folder_path)

            for file in files:
                if file.endswith((".txt", ".htm", ".html")):
                    with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                        content = f.read()
                        return extract_main_filing_section(content)
        return f"No readable {form_type} filing found for {ticker}."
    except Exception as e:
        return f"No {form_type} filing found for {ticker}. Error: {str(e)}"


# sec_filings.py

def extract_main_filing_section(raw_data: str) -> str:
    """
    Extract a portion of the 10-K related to the income statement.
    Looks for the term 'Consolidated Statements of Operations' or similar.
    """
    keywords = [
        "CONSOLIDATED STATEMENTS OF OPERATIONS",
        "INCOME STATEMENT",
        "CONSOLIDATED INCOME STATEMENTS",
        "STATEMENT OF EARNINGS",
        "CONSOLIDATED RESULTS OF OPERATIONS",
        "RESULTS OF OPERATIONS"
    ]

    raw_upper = raw_data.upper()
    for keyword in keywords:
        idx = raw_upper.find(keyword)
        if idx != -1:
            return raw_data[idx:idx + 15000]  # pull next 15k characters from match

    # fallback if nothing matched
    return raw_data[:10000]





import os

def get_all_sec_filings(ticker, form_type="10-K"):
    path = f"sec_filings/sec-edgar-filings/{ticker}/{form_type}"
    filings = {}
    try:
        if not os.path.exists(path):
            return {}
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith((".txt", ".htm", ".html")):
                        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                            content = f.read()
                            for line in content.splitlines():
                                if "CONFORMED PERIOD OF REPORT" in line:
                                    year = line.strip().split(":")[-1].strip()[:4]
                                    filings[year] = folder_path
                                    break
        return filings
    except Exception as e:
        print(f"Error: {e}")
        return {}




def get_filing_text(folder_path):
    try:
        for file in os.listdir(folder_path):
            if file.endswith((".txt", ".htm", ".html")):
                with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                    from .sec_filings import extract_main_filing_section
                    return extract_main_filing_section(f.read())
    except Exception as e:
        return f"Could not load filing content: {e}"

def download_sec_filings(ticker, form_type="10-K", limit=5):
    """
    Downloads SEC filings of the given type for the specified ticker.
    """
    try:
        dl.get(form_type, ticker, limit=limit)
    except Exception as e:
        print(f"Failed to download {form_type} for {ticker}: {e}")



