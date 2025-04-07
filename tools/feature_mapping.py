from difflib import get_close_matches
from langchain_groq import ChatGroq  # Assuming you're using this LLM

# Initialize LLM
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.0)

available_features = [
    'Revenue', 'Revenue Growth', 'Cost of Revenue', 'Gross Profit', 'R&D Expenses', 'SG&A Expense',
    'Operating Expenses', 'Operating Income', 'Interest Expense', 'Earnings before Tax',
    'Income Tax Expense', 'Net Income - Non-Controlling int', 'Net Income - Discontinued ops',
    'Net Income', 'Preferred Dividends', 'Net Income Com', 'EPS', 'EPS Diluted',
    'Weighted Average Shs Out', 'Weighted Average Shs Out (Dil)', 'Dividend per Share',
    'Gross Margin', 'EBITDA Margin', 'EBIT Margin', 'Profit Margin', 'Free Cash Flow margin',
    'EBITDA', 'EBIT', 'Consolidated Income', 'Earnings Before Tax Margin', 'Net Profit Margin'
]

def map_metric_with_llm(user_metric: str, available_features: list):
    prompt = f"""
    You are an intelligent financial assistant. The user has provided the following metric: "{user_metric}". 

    From the list of available features:
    {available_features}

    Find the most relevant feature from the list that matches the user's metric. 
    If you cannot find a relevant match, say "No match found".
    """

    
    response = llm.invoke([{"role": "user", "content": prompt}])
    if "No match found" in response.content:
        return None
    else:
        # Extract the most relevant feature name from the LLM's response
        closest_match = response.content.strip()
        if closest_match in available_features:
            return closest_match
        return None


def get_closest_feature(target_feature, available_features):
    closest_match = get_close_matches(target_feature, available_features, n=1, cutoff=0.6)
    return closest_match[0] if closest_match else None


def validate_feature_with_llm(user_metric: str, available_features: list):
    llm_match = map_metric_with_llm(user_metric, available_features)
    if llm_match:
        return llm_match

    # If LLM fails, fall back to fuzzy matching
    return get_closest_feature(user_metric, available_features)
