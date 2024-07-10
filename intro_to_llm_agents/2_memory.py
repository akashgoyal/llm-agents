
# ADD LONG TERM MEMORY
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# webpage or pdf or any other information source
loader = WebBaseLoader("https://lilianweng.github.io/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

results = retriever.get_relevant_documents("give count of publications year wise")
print(results)


# ADD SHORT TERM MEMORY
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_openai.templates import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai.placeholders import MessagesPlaceholder
from langchain_openai.templates import PromptTemplate

retriever_tool = create_retriever_tool( retriever, "lilianweng_search", "Search for information about Lilian Weng's blog!", )

prompt = [
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
# prompt = ChatPromptTemplate.from_messages(
#     [ ("system", "You are a helpful assistant"), MessagesPlaceholder("chat_history", optional=True),
#     ("human", "{input}"), MessagesPlaceholder("agent_scratchpad"), ]
# )


# RUN AGENT USING MEMORY
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [retriever_tool]
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# create message history
message_history = ChatMessageHistory()
# message_history is linked with the agent
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Try out the agent
agent_with_chat_history.invoke(
    {"input": "hi! I'm bob"},
    config={"configurable": {"session_id": "<foo>"}}, 
    # session_id is a unique identifier for the conversation. Needed in real world scenarios.
    # here it can be ignored.
)

agent_with_chat_history.invoke(
    {"input": "what's my name?"},
    config={"configurable": {"session_id": "<foo>"}},
)

# 
