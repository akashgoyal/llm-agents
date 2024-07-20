import os, wget
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, ServiceContext
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.llms.together import TogetherLLM

from dotenv import load_dotenv
load_dotenv('.env')

# set models
llm = TogetherLLM(model="meta-llama/Llama-3-8b-chat-hf", api_key=os.getenv("TOGETHER_API_KEY"))
jina_embedding_model = HuggingFaceEmbedding(model_name="jinaai/jina-embeddings-v2-base-en") # 768
print("test : ", len(jina_embedding_model.get_text_embedding('This is a test'))) 


# download & view chunk data
def download_pdf():
    url = 'https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf'
    filename = 'wework.pdf'
    wget.download(url, filename)

def chunk_documents(input_files=["wework.pdf"], chunk_size=100, chunk_overlap=20):
    # chunking 
    docs = SimpleDirectoryReader(input_files=input_files).load_data()
    print(len(docs))

    base_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    base_nodes = base_splitter.get_nodes_from_documents(docs)

    for elt in base_nodes[5:10]:
        print(f'element is: {elt.get_content()}\n')

download_pdf()
chunk_documents()


#######  Load data in Milvus  ########
vector_store_jina = MilvusVectorStore(
    uri="milvus_rag_llama_index.db",
    collection_name="wework_data",
    dim=768,  # the value changes with embedding model
    overwrite=True  # drop table if exist and then create
    )
storage_context_jina = StorageContext.from_defaults(vector_store=vector_store_jina)

## ServiceContext has llm specification 
docs = SimpleDirectoryReader(input_files=["wework.pdf"]).load_data()
service_context_jina = ServiceContext.from_defaults(llm=llm, embed_model=jina_embedding_model, chunk_size=512, chunk_overlap=100)
vector_index_jina = VectorStoreIndex.from_documents(docs, storage_context=storage_context_jina, service_context=service_context_jina)

# optional tool
# milvus_tool_openai = RetrieverTool(
#     retriever=vector_index_jina.as_retriever(similarity_top_k=3),  # retrieve top_k results
#     metadata=ToolMetadata(
#         name="CustomRetriever",
#         description='Retrieve relevant information from provided documents.'
#     ),
# )

# query_engine = vector_index_jina.as_query_engine()
# response = query_engine.query("What are the Risk Factors for WeWork that could happen?")
# print(response)
print("####################")

# "Can you tell me about WeWork, what they do and who they are?"
# "What are the Risk Factors for WeWork that could happen?"
query_engine = vector_index_jina.as_query_engine()
while True:
    query_str = input("Enter your query: ")
    if query_str == "quit":
        break
    response = query_engine.query(query_str)
    print(response)
