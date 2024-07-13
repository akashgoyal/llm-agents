
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub

# initialize llm
import os
from dotenv import load_dotenv
load_dotenv('llm_init.env')
model = "gemini-1.5-flash"
api_key = os.getenv("GOOGLE_API_KEY")
llm_gemini = ChatGoogleGenerativeAI(google_api_key=api_key, model=model, temperature=0.7)


# ADD LONG TERM MEMORY - vector store
# webpage or pdf or any other information source
loader = WebBaseLoader("https://lilianweng.github.io/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200).split_documents(docs)
vector = FAISS.from_documents(documents, HuggingFaceEmbeddings())
retriever = vector.as_retriever()

results = retriever.get_relevant_documents("give count of publications year wise")
print(results)

# CREATE Agent with the retriever tool
prompt = hub.pull("hwchase17/structured-chat-agent")
qa_chain = RetrievalQA.from_chain_type(llm_gemini, retriever=retriever)
retriever_tool = Tool( name="lilianweng_search", func=qa_chain.run, description="Search for information about Lilian Weng's blog. Input should be a fully formed question." )
tools = [retriever_tool]
agent = create_structured_chat_agent(llm_gemini, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# MAKE USE OF SHORT MEMORY - chat history
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
agent_executor.invoke({"input": "hi. what can you do for me. what is your tool about. Can you search for information about Lilian Weng's blog ?"})

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


## RESULTS
# [Document(metadata={'source': 'https://lilianweng.github.io/', 'title': "Lil'Log", 'description': 'Document my learning notes.', 'language': 'en'}, 
# page_content='Date: September 24, 2021  |  Estimated Reading Time: 21 min  |  Author: Lilian Weng\n\n\n\n\nWhat are Diffusion Models?'), 
# Document(metadata={'source': 'https://lilianweng.github.io/', 'title': "Lil'Log", 'description': 'Document my learning notes.', 'language': 'en'}, 
# page_content='Date: June 9, 2022  |  Estimated Reading Time: 25 min  |  Author: Lilian Weng\n\n\n\n\nLearning with not Enough Data Part 3: Data Generation'), 
# Document(metadata={'source': 'https://lilianweng.github.io/', 'title': "Lil'Log", 'description': 'Document my learning notes.', 'language': 'en'}, 
# page_content='Date: January 2, 2021  |  Estimated Reading Time: 42 min  |  Author: Lilian Weng\n\n\n\n\nHow to Build an Open-Domain Question Answering System?'), 
# Document(metadata={'source': 'https://lilianweng.github.io/', 'title': "Lil'Log", 'description': 'Document my learning notes.', 'language': 'en'}, 
# page_content='Date: October 29, 2020  |  Estimated Reading Time: 33 min  |  Author: Lilian Weng\n\n\n\n\nNeural Architecture Search')]


# > Entering new AgentExecutor chain...
# ```json
# {
#  "action": "Final Answer",
#  "action_input": "I am a large language model, trained to be informative and comprehensive. I can provide summaries of factual topics or create stories. I can also search for information about Lilian Weng's blog using a specialized tool.  What would you like me to do?"
# }
# ```

# > Finished chain.


# > Entering new AgentExecutor chain...
# ```json
# {
#  "action": "Final Answer",
#  "action_input": "Hi Bob! 👋  What can I do for you today?"
# }
# ```

# > Finished chain.


# > Entering new AgentExecutor chain...
# ```json
# {
#   "action": "Final Answer",
#   "action_input": "Your name is Bob! 😄"
# }
# ```

# > Finished chain.


# > Entering new AgentExecutor chain...
# ```json
# {
#   "action": "lilianweng_search",
#   "action_input": "What are the publication counts for Lilian Weng's blog by year and what are some of the important topics she covered in the previous year?"
# }
# ```I can't give you the exact publication counts for Lilian Weng's blog by year. The provided context only lists the dates for a few articles and doesn't provide information on the total number of posts for each year.

# However, based on the provided context, here are some topics Lilian Weng covered in the previous year (2023):

# * **Prompt Engineering:** This topic was discussed in a post dated June 23, 2023. 
# * **Data Generation:**  While the provided context only mentions a post on this topic dated June 9, 2022, it's likely that Lilian Weng continued to explore this area in 2023.  

# To find the publication counts and a more comprehensive list of topics for each year, you would need to visit Lilian Weng's blog directly. 
# ```json
# {
#   "action": "Final Answer",
#   "action_input": "I can't give you the exact publication counts for Lilian Weng's blog by year. The provided context only lists the dates for a few articles and doesn't provide information on the total number of posts for each year. However, based on the provided context, some topics Lilian Weng covered in the previous year (2023) include Prompt Engineering and Data Generation. To find the publication counts and a more comprehensive list of topics for each year, you would need to visit Lilian Weng's blog directly."
# }
# ```

# > Finished chain.