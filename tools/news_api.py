import os
import requests
from datetime import datetime, timedelta

NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # Store this in your .env or shell

def fetch_company_news(company_name: str, from_days_ago: int = 7, max_articles: int = 5, custom_from_date: str = None):
    base_url = "https://newsapi.org/v2/everything"
    from_date = custom_from_date or (datetime.now() - timedelta(days=from_days_ago)).strftime('%Y-%m-%d')

    params = {
        "q": company_name,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"NewsAPI request failed: {response.text}")

    data = response.json()
    articles = [
        {
            "title": article["title"],
            "description": article["description"],
            "url": article["url"],
            "publishedAt": article["publishedAt"]
        }
        for article in data.get("articles", [])
    ]

    return articles


