# --- Agent Prompts ---
# These define the system messages and user prompts for each agent.
ANALYST_PROMPT = """
Analyse user question and state user goals and output data which user wants to obtain.

Reason about the conditions and filtering, grouping and ordering.

State if the target data must be a single-row or multi-rows.

State string literals to be used in condition checks.

Do not give more information apart from what asked above.

Explain explicitly your reasoning step by step.

###Natural Language Question
{question}

###Database Schema

{database_schema}

###Helpful Data (use string values exactly as they appear in the reference data)

{reference_data}

Do not return SQL.
"""


SCHEMA_LINKER_PROMPT = """
Given a natural language question and a database schema (list of tables and columns), \
    your task is to identify the corresponding list of 'exact tables and columns'.

Explain explicitly your reasoning.

#Input

###Database Schema

{database_schema}

###User Goals and Intention

{goals}

###Rules & Guidelines
Always use SQLite dialect.

** Only link to column or table names that exactly match the provided schema. **

"""


QUERY_BUILDER_PROMPT = """
You are tasked with building an SQL query based on the inputs given:

#Input
You are given:

#Natural Language Question

{question}

#Comments on User Goals

{goals}

###Schema Linker Output

{schemalink}


###Reference Database Schema

{database_schema}

###Your Task
**Do exactly what is decribed in `User Goals` and `Schema Linker Output` to construct the SQL query.**

###Rules & Guidelines
Always use SQLite dialect.

DO NOT PERFORM CAST.

"""


REFINER_PROMPT = """
###Task
Fix execution error based on `database schema columns` and `return corrected query`, \
    **do not return the same query**.

USE SQLITE SYNTAX ONLY.

###Database Schema

{schema}

#Query

{query}

#Error

`{error}`

#Stick to the User Goals when fixing the query execution error

`{goals}`
"""