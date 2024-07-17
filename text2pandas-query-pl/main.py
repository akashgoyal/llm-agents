import os
import requests
import pandas as pd
from llama_index.llms import Gemini #OpenAI
from pipeline import define_pipeline, visualise_pipeline
from dotenv import load_dotenv

# initialize llm
load_dotenv('.env')
api_key = os.getenv("GOOGLE_API_KEY")
llm = Gemini(api_key=api_key, model_name= "models/gemini-1.5-flash")
print("LLM initialized")

# Load the dataset from a URL - start
def download_file(url, filename):
    try:
        response = requests.get(url)
        with open(filename, 'wb') as file:
            file.write(response.content)
    except Exception as e:
        return f"Failed to download file: {e}"
    return 200
data_url = "https://raw.githubusercontent.com/akashgoyal/datasets-ref/master/titanic.csv"
status = download_file(data_url, 'titanic.csv')
if status == 200:
    print("File downloaded successfully")
else:
    print(status)
    exit()

df = pd.read_csv("./titanic.csv")
print(len(df))
print("Dataset loaded successfully")
# Load the dataset from a URL - end

# Define the pipeline 
qp = define_pipeline(df, llm1=llm, llm2=llm)
print("Pipeline defined")

# visualise pipeline 
visualise_pipeline(qp)
print("Pipeline visualized")

# execute 
while True:
    query_str = input("Enter your query: ")
    if query_str == "exit":
        break
    # "What is the correlation between survival and age?"
    response = qp.run(
        query_str=query_str,
    )
    print(response.message.content)
