import re
import sqlite3

# --- SQL Cleaning Function ---
def clean_sql(final_sql):
    match = re.search(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', final_sql, re.DOTALL)
    if match:
        final_sql = match.group(1).strip()

    cleaned = final_sql.replace("\\n", " ")      # Remove newlines
    cleaned = cleaned.replace('\\"', '')         # Replace escaped quotes 
    cleaned = cleaned.replace('\\', '')          # Remove remaining backslashes
    cleaned = cleaned.replace("`", '')           # Remove remaining backticks
    cleaned = ' '.join(cleaned.split())          # Remove extra spaces
    return cleaned
