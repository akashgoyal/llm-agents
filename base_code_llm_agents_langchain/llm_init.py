from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
#
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.tools import GoogleSearchRun
from langchain_community.tools import DuckDuckGoSearchRun

import os
from dotenv import load_dotenv
load_dotenv('../.env')

####### LLMs ########
def init_openai():
    """Initialize OpenAI LLM"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return ChatOpenAI(api_key=api_key)

def init_claude():
    """Initialize Anthropic Claude LLM"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    return ChatAnthropic(anthropic_api_key=api_key, model_name="claude-3")

def init_gemini():
    """Initialize Google Gemini LLM"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    return ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash", temperature=0.7)

def init_mistral():
    """Initialize Mistral LLM using Hugging Face Hub"""
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_key:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
    return HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", huggingfacehub_api_token=api_key)

# lite-llm


####### TOOLS ########
def init_duckduckgo():
    """Initialize DuckDuckGo Search Tool"""
    ddg_search = DuckDuckGoSearchRun()
    return ddg_search

def init_tavily():
    """Initialize Tavily Search Tool"""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    tavily_search = TavilySearchResults(tavily_api_key=tavily_api_key)
    return tavily_search

def init_googlesearch():
    """Initialize Google Search Tool"""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    if not google_api_key or not google_cse_id:
        raise ValueError("GOOGLE_API_KEY or GOOGLE_CSE_ID not found in environment variables")
    google_search = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper())
    return google_search


# if __name__ == "__main__":
#     try:
#         openai_llm = init_openai()
#         claude_llm = init_claude()
#         gemini_llm = init_gemini()
#         mistral_llm = init_mistral()
        
#         print("All LLMs initialized successfully!")
#     except ValueError as e:
#         print(f"Error: {e}")
