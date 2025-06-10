import re
import sqlite3
from typing_extensions import Annotated

# --- SQL Cleaning Function ---
def clean_sql(final_sql):
    """
    Cleans a generated SQL string by removing JSON/markdown wrappers, newlines,
    escaped quotes, backslashes, backticks, and extra spaces.
    Ensures the query ends with a semicolon.
    """
    # Regex to extract SQL from a JSON-like string (e.g., {"sql": "SELECT..."})
    match = re.search(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', final_sql, re.DOTALL)
    if match:
        final_sql = match.group(1).strip()

    # Ensures only the first statement is taken and ends with a semicolon
    final_sql = final_sql.split(";", 1)[0].strip() + ";"

    cleaned = final_sql.replace("\\n", " ")       # Remove newlines
    cleaned = cleaned.replace('\\"', '')          # Replace escaped quotes
    cleaned = cleaned.replace("`", '')            # Remove remaining backticks
    cleaned = ' '.join(cleaned.split())           # Remove extra spaces
    
    return cleaned


# --- SQL Executor Function ---
# This function executes a SQL query against a SQLite database.
async def execute_query(sql: str, dbname: str) -> Annotated[str, "query results"]:
    db_url = f"/home/simges/.cache/spider_data/test_database/{dbname}/{dbname}.sqlite"
    
    conn = None # Initialize conn to None for proper cleanup in finally block
    try:
        conn = sqlite3.connect(db_url)
        cursor = conn.cursor()
        cursor.execute(sql)
        return "passed"
    except Exception as e:
        return "failure: " + str(e)
    finally:
        if conn:
            conn.close() # Ensure connection is closed.
