import os
import json
import requests
import zipfile
import pandas as pd
from pathlib import Path
from llama_index.core.program import LLMTextCompletionProgram, MultiModalLLMCompletionProgram
from llama_index.core.bridge.pydantic import BaseModel, Field
import prompts
import llms

#
my_llm = llms.llama3_70b_init()
tableinfo_dir = "WikiTableQuestions_TableInfo"
os.makedirs(tableinfo_dir, exist_ok=True)

#
def download_and_extract_data(url):
    response = requests.get(url)
    with open('data.zip', 'wb') as file:
        file.write(response.content)
    with zipfile.ZipFile('data.zip', 'r') as zip_ref:
        zip_ref.extractall()

def process_csv_files(data_dir):
    csv_files = sorted([f for f in data_dir.glob("*.csv")])
    dfs = []
    for csv_file in csv_files:
        print(f"processing file: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error parsing {csv_file}: {str(e)}")
    return dfs

# extract tablename and summary from each table. By prompting
class TableInfo(BaseModel):
    """Information regarding a structured table."""
    table_name: str = Field( ..., description="table name (must be underscores and NO spaces)" )
    table_summary: str = Field( ..., description="short, concise summary/caption of the table" )

def create_llmtextcompletion_program(llm, prompt_str):
    program = LLMTextCompletionProgram.from_defaults(output_cls=TableInfo, llm=llm, prompt_template_str=prompt_str,)
    return program

#
def _get_tableinfo_with_index(idx: int) -> str:
    results_gen = Path(tableinfo_dir).glob(f"{idx}_*")
    results_list = list(results_gen)
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        return TableInfo.parse_file(path)
    else:
        raise ValueError(f"More than one file matching index: {list(results_gen)}")

def process_table_infos(dfs, tableinfo_dir):
    table_names = set()
    table_infos = []
    for idx, df in enumerate(dfs):
        table_info = _get_tableinfo_with_index(idx)
        if table_info:
            table_infos.append(table_info)
        else:
            while True:
                df_str = df.head(10).to_csv()
                table_info = program(
                    table_str=df_str,
                    exclude_table_name_list=str(list(table_names)),
                )
                table_name = table_info.table_name
                print(f"Processed table: {table_name}")
                if table_name not in table_names:
                    table_names.add(table_name)
                    break
                else:
                    # try again
                    print(f"Table name {table_name} already exists, trying again.")
                    pass

            out_file = f"{tableinfo_dir}/{idx}_{table_name}.json"
            json.dump(table_info.dict(), open(out_file, "w"))
        table_infos.append(table_info)
    
    return table_infos

#
url = "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip"
download_and_extract_data(url)
print("Data downloaded and extracted")
#
data_dir = Path("./WikiTableQuestions/csv/200-csv")
dfs = process_csv_files(data_dir)
#
prompt_str = prompts.get_tablename_summary_str
program = create_llmtextcompletion_program(my_llm, prompt_str)

table_infos = process_table_infos(dfs, tableinfo_dir)

