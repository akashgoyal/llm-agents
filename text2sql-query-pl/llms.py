import os
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv('.env')
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

def gemini_init():
    llm = Gemini(api_key=google_api_key, model_name= "models/gemini-1.5-flash")
    return llm

def llama3_8b_init():
    llm = Groq(api_key=groq_api_key, model="llama3-8b-8192")
    return llm

def llama3_70b_init():
    llm = Groq(api_key=groq_api_key, model="llama3-70b-8192")
    return llm

def together_llama_8b_init():
    llm = TogetherLLM(api_key=together_api_key, model="meta-llama/Llama-3-8b-chat-hf")
    return llm

def together_llama_70b_init():
    llm = TogetherLLM(api_key=together_api_key, model="meta-llama/Llama-3-70b-chat-hf")
    return llm

def together_m2bert_2k_init():
    llm = TogetherLLM(api_key=together_api_key, model="togethercomputer/m2-bert-80M-2k-retrieval")
    return llm

def together_baai_bge_base_init():
    llm = TogetherLLM(api_key=together_api_key, model="BAAI/bge-base-en-v1.5")
    return llm

def hf_baai_bge_small_init():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return embed_model

from llama_index.core import Settings
# Settings.llm = together_llama_70b_init()
Settings.embed_model = hf_baai_bge_small_init()
