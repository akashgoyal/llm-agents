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


# DEFINE TOOLS
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
#
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search)
#
@tool
def calculate_length_tool(a: str) -> int:
    """The function calculates the length of the input string."""
    return len(a)

@tool
def calculate_uppercase_tool(a: str) -> int:
    """The function calculates the number of uppercase characters in the input string."""
    return sum(1 for c in a if c.isupper())


# DEFINE MEMORY / VECTOR STORE / CHAT HISTORY
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# Long Term / VectorStore
loader = WebBaseLoader("https://lilianweng.github.io/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()
retriever_tool = create_retriever_tool( retriever, "lilianweng_search", "Search for information about Lilian Weng's blog!", )
# Short Term / Chat History
message_history = ChatMessageHistory()


# Sensory Memory 
from langchain_openai.templates import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai.placeholders import MessagesPlaceholder
from langchain_openai.templates import PromptTemplate
prompt = [
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]


# TOOLS & VectorStore Tool
tools = [retriever_tool, tavily_tool, calculate_length_tool, calculate_uppercase_tool]
for t in tools:
    print(t.name, t.description)


# AGENT
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# RUN
agent_with_chat_history.invoke(
    {"input": "Hello!"},
    config={"configurable": {"session_id": "<foo>"}},
)
agent_with_chat_history.invoke(
    {"input": "What are the topics and sub-topics covered by Lilian Weng's blogs. Calcluate the length of the result?"},
    config={"configurable": {"session_id": "<foo>"}},
)
agent_with_chat_history.invoke(
    {"input": "What is the recent blog written by Lilian Weng. How many uppercase characters are in that blog?"},
    config={"configurable": {"session_id": "<foo>"}},
)
agent_with_chat_history.invoke(
    {"input": "List some of the popular researchers in the area of Artificial Intelligence. Calculate the uppercase characters of this description"},
    config={"configurable": {"session_id": "<foo>"}},
)       
agent_with_chat_history.invoke(
    {"input": "What is the weather today in Madrid and calculate the uppercase characters of this description"},
    config={"configurable": {"session_id": "<foo>"}},
)
agent_with_chat_history.invoke(
    {"input": "Make a quick summary of what we have discussed today"},
    config={"configurable": {"session_id": "<foo>"}},
)


# STRUCTURE THE OUTPUTS 
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator

# Define your desired data structure.
class DesiredStructure(BaseModel):
    question: str = Field(description="the question asked")
    numerical_answer: int = Field(description="the number extracted from the answer, text excluded")
    text_answer: str = Field(description="the text part of the answer, numbers excluded")

parser = PydanticOutputParser(pydantic_object=DesiredStructure)

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
prompt_and_model = prompt | model
output = prompt_and_model.invoke({
    "query": "Find the description of Topic of focus for Researchers at OpenAI and calculate the length of this description"}
)

print(output)