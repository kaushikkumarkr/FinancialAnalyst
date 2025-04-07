# utils/formatters.py
def format_large_number(value):
    """Formats large integers with commas."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"

def format_currency(value, prefix="$"):
    """Formats number as currency."""
    try:
        return f"{prefix}{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"

def format_percentage(value):
    """Formats a float as a percentage string."""
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"
    
def format_income_statement(data):
    """Formats income statement dictionary for better LLM summarization."""
    if not data:
        return "No income statement data available."

    lines = []
    for year in sorted(data.keys(), reverse=True):
        entry = data[year]
        lines.append(f"\n📅 **{year}**:")
        for key, val in entry.items():
            if "margin" in key.lower() or "rate" in key.lower():
                val_fmt = format_percent(val)
            elif "eps" in key.lower():
                val_fmt = f"${val:.2f}" if val is not None else "N/A"
            else:
                val_fmt = format_large_number(val)
            lines.append(f"- {key.replace('_', ' ').title()}: {val_fmt}")
    return "\n".join(lines)    



def format_percent(value, precision=1):
    try:
        return f"{float(value)*100:.{precision}f}%"
    except:
        return "N/A"
