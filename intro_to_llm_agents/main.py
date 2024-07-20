import os
import yaml

config = yaml.safe_load(open("config.yml"))

os.environ["OPENAI_API_KEY"] = config["OPEN_AI_API_KEY"]
os.environ["TAVILY_API_KEY"] = config["TAVILY_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = config["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_HUB_API_KEY"] = config["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = str(config["LANGCHAIN_TRACING_V2"]).lower()
os.environ["LANGCHAIN_ENDPOINT"] = config["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_HUB_API_URL"] = config["LANGCHAIN_HUB_API_URL"]
os.environ["LANGCHAIN_WANDB_TRACING"] = str(config["LANGCHAIN_WANDB_TRACING"]).lower()
os.environ["WANDB_PROJECT"] = config["WANDB_PROJECT"]


import planning as plan


from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv('../.env')
api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash", temperature=0.7)
