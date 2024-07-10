
# ADD LONG TERM MEMORY
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

# webpage or pdf or any other information source
loader = WebBaseLoader("https://lilianweng.github.io/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

results = retriever.get_relevant_documents("give count of publications year wise")
print(results)

# create retriever tool. It will be linked to one particular agent. 
retriever_tool = create_retriever_tool( retriever, "lilianweng_search", "Search for information about Lilian Weng's blog!", )


# ADD SHORT TERM MEMORY
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_openai.templates import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai.placeholders import MessagesPlaceholder
from langchain_openai.templates import PromptTemplate

prompt = [
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
# agent_scratchpad - must be there. intermediate agent actions and tool output messages will be passed in here.
# prompt = ChatPromptTemplate.from_messages(
#     [ ("system", "You are a helpful assistant"), MessagesPlaceholder("chat_history", optional=True),
#     ("human", "{input}"), MessagesPlaceholder("agent_scratchpad"), ]
# )


# CREATE Agent with the retriever tool
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [retriever_tool]
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# MAKE USE OF SHORT MEMORY
# create message history
message_history = ChatMessageHistory()
# message_history is linked with the agent, through session_id
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# To not utilise short memory - don't create agent_with_chat_history, just use agent_executor.
# run agent without chat history/short memory
agent_executor.invoke({"input": "hi"})
agent_executor.invoke({"input": "what can you do for me"})
agent_executor.invoke({"input": "what is your tool about. Can you search for information about Lilian Weng's blog ?"})


# Try out the agent
agent_with_chat_history.invoke(
    {"input": "hi! I'm bob"},
    config={"configurable": {"session_id": "<foo>"}}, 
    # session_id is a unique identifier for the conversation. Must be present in real world scenarios.
)

agent_with_chat_history.invoke(
    {"input": "what's my name?"},
    config={"configurable": {"session_id": "<foo>"}},
)

agent_with_chat_history.invoke(
    {"input": "I want to know about Lilian Weng's blog. Give count of publications year wise. Also list important topics she has covered in previous year."},
    config={"configurable": {"session_id": "<foo>"}},
)

