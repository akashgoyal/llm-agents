import os
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq

load_dotenv('.env')
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

def gemini_init():
    llm = Gemini(api_key=google_api_key, model_name= "models/gemini-1.5-flash")
    return llm

def llama3_8b_init():
    llm = Groq(api_key=groq_api_key, model="llama3-8b-8192")
    return llm

def llama3_70b_init():
    llm = Groq(api_key=groq_api_key, model="llama3-70b-8192")
    return llm
