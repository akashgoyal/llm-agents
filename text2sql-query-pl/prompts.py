instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

get_tablename_summary_str = (
    "Give me a summary of the table in the following JSON format.\n"
    "- The table name must be unique to the table and describe it while being concise.\n"
    "- Do NOT output a generic table name (e.g. table, my_table).\n"
    "Do NOT make the table name one of the following: {exclude_table_name_list}\n"
    "Table:\n{table_str}\n"
    "Summary: "
)