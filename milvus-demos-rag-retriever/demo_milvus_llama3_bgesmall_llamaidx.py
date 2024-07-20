from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
import os
import urllib.request
from dotenv import load_dotenv
load_dotenv('.env')

##
together_api_key = os.getenv("TOGETHER_API_KEY")
#
model_name = "meta-llama/Llama-3-8b-chat-hf"
llm = TogetherLLM(model=model_name, api_key=together_api_key)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") # 384
# print("Embed len:", len(Settings.embed_model.get_text_embedding("some text for testing")))

# doing below to override default model settings while creating VectorStoreIndex
from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model = embed_model

##
def download_files():
    os.makedirs('data', exist_ok=True)
    url1 = 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt'
    url2 = 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf'
    urllib.request.urlretrieve(url1, 'data/paul_graham_essay.txt')
    urllib.request.urlretrieve(url2, 'data/uber_2021.pdf')
download_files()

##

# load documents
documents = SimpleDirectoryReader(input_files=["./data/paul_graham_essay.txt"]).load_data()
print("Document ID:", documents[0].doc_id)

# Create an index over the documents
vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=384, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

#
query_engine = index.as_query_engine()
res = query_engine.query("What did the author learn?")
print(res)

#
res = query_engine.query("What challenges did the disease pose for the author?")
print(res)

#