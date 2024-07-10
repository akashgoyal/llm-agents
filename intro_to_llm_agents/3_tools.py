## TOOL 1
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

loader = WebBaseLoader("https://lilianweng.github.io/") # webpage or pdf or any other information source
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()
retriever_tool = create_retriever_tool( retriever, "lilianweng_search", "Search for information about Lilian Weng's blog!", )

## TOOL 2
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search)

# Initialize Agent with both tools
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
agent_chain = initialize_agent(
    [retriever_tool, tavily_tool],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
# run the agent, let's see how the agent decide which tool to use when
agent_chain.run("What are the topics covered in lilianweng's blogs.",)
agent_chain.run("List some of the popular researchers in the area of Artificial Intelligence.",)


#########################
# Custom tools and reasoning
# Custom Tools : https://python.langchain.com/docs/modules/agents/tools/custom_tools
# Reasoning: https://api.python.langchain.com/en/latest/agents/langchain.agents.agent_types.AgentType.html

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
#
@tool
def calculate_length_tool(a: str) -> int:
    """The function calculates the length of the input string."""
    return len(a)
print(calculate_length_tool.description)
print(calculate_length_tool.args)

#
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
agent_chain = initialize_agent(
    [retriever_tool, tavily_tool, calculate_length_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
#
agent_chain.run( "Find the research topics covered by lilian weng in her blogs, and calculate the length of that result", )


