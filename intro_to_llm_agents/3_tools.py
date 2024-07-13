from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# llm init 
load_dotenv('llm_init.env')
model = "gemini-1.5-flash"
api_key = os.getenv("GOOGLE_API_KEY")
llm_gemini = ChatGoogleGenerativeAI(google_api_key=api_key, model=model, temperature=0.7)


## TOOL 1
loader = WebBaseLoader("https://lilianweng.github.io/") # webpage or pdf or any other information source
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200).split_documents(docs)
vector = FAISS.from_documents(documents, HuggingFaceEmbeddings())
retriever = vector.as_retriever()
# 
qa_chain = RetrievalQA.from_chain_type(llm_gemini, retriever=retriever)
retriever_tool = Tool( name="lilianweng_search", func=qa_chain.run, description="Search for information about Lilian Weng's blog. Input should be a fully formed question." )


## TOOL 2
tavily_ak = os.getenv("TAVILY_API_KEY")
api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_ak)
tavily_tool = Tool( name="Tavily Search", description="Useful for searching the internet for current information. Input should be a search query.", func=api_wrapper.results )

## AGENT - Initialize Agent with both tools
agent_chain = initialize_agent(
    [retriever_tool, tavily_tool],
    llm_gemini,
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
agent_chain = initialize_agent(
    [retriever_tool, tavily_tool, calculate_length_tool],
    llm_gemini,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
#
agent_chain.run( "Find the research topics covered by lilian weng in her blogs, and calculate the length of that result", )



### RESULTS ###
# > Entering new AgentExecutor chain...
# Thought: Lilian Weng's blog covers a variety of topics related to machine learning and deep learning. I need to find more information about the specific topics she covers.
# Action:
# ```json
# {
#   "action": "lilianweng_search",
#   "action_input": "What topics does Lilian Weng's blog cover?"
# }
# ```
# Observation: Based on the provided context, Lilian Weng's blog covers the following topics:

# * **Extrinsic Hallucinations in LLMs** 
# * **Prompt Engineering**
# * **Learning with not Enough Data Part 3: Data Generation** 

# Thought:Action:
# ```json
# {
#   "action": "lilianweng_search",
#   "action_input": "What are the topics covered in lilianweng's blogs?"
# }
# ```
# Observation: Based on the provided context, Lilian Weng's blog, Lil'Log, covers topics such as:

# * **Extrinsic Hallucinations in LLMs**
# * **Learning with not Enough Data Part 3: Data Generation**
# * **Neural Architecture Search** 

# Thought:Action:
# ```json
# {
#   "action": "lilianweng_search",
#   "action_input": "What topics are covered in lilianweng's blog?"
# }
# ```
# Observation: Based on the provided context, Lilian Weng's blog, Lil'Log, covers topics such as:

# * **Extrinsic Hallucinations in LLMs** 
# * **Learning with not Enough Data Part 3: Data Generation**
# * **How to Build an Open-Domain Question Answering System?** 

# It appears that the blog focuses on AI and machine learning. 

# Thought:Action:
# ```json
# {
#   "action": "Final Answer",
#   "action_input": "Lilian Weng's blog, Lil'Log, covers a variety of topics related to AI and machine learning, including: Extrinsic Hallucinations in LLMs, Learning with not Enough Data Part 3: Data Generation, How to Build an Open-Domain Question Answering System?, and Prompt Engineering.  It is a great resource for anyone looking to learn more about these topics."
# }
# ```

# > Finished chain.


# > Entering new AgentExecutor chain...
# Thought: I need to find a list of popular researchers in AI.  I can use Tavily Search to find a list.
# Action:
# ```json
# {
#  "action": "Tavily Search",
#  "action_input": "List of popular AI researchers"
# }
# ```
# Observation: [{'url': 'https://time.com/collection/time100-ai/', 'content': 'Daniela Amodei\nSam Altman\nDemis Hassabis\nRobin Li\nClément Delangue\nLila Ibrahim\nElon Musk\nRaquel Urtasun\nAlex Karp\nReid Hoffman\nGreg Brockman\nMarc Andreessen\nSandra Rivera\nAidan Gomez\nDaniel Gross\nKai-Fu Lee\nJaime Teevan\nAndrew Ng\nKevin Scott\nJensen Huang\nClara Shih\nAlexandr Wang\nMustafa Suleyman\nMarc Raibert\nTed Chiang\nTed Chiang\nCharlie Brooker\nHolly Herndon\nPelonomi Moiloa\nGrimes\nNeal Khosla\nStephanie Dinkins\nSougwen Chung\nCristóbal Valenzuela\nLilly Wachowski\nManu Chopra\nKate Kallot\nZiad Obermeyer\nNoam Shazeer\nAlison Darcy\nNathaniel Manning\nTushita Gupta\nAndrew Hopkins\nLinda Dounia Rebeiz\nRichard Socher\nKeith Dreyer\nNancy Xu\nRootport\nAlondra Nelson\nAlondra Nelson\nIan Hogarth\nAudrey Tang\nMeredith Whittaker\nJames Manyika\nJack Clark\nAnna Makanju\nOmar Al Olama\nKelly McKernan\nEric Schmidt\nMargrethe Vestager\nAnna Eshoo\nRichard Mathenge\nSneha Revanur\nTristan Harris\nJoy Buolamwini\nEliezer Yudkowsky\nVerity Harding\nTed Lieu\nSarah Chander\nNina Jankowicz\nRomesh and Sunil Wadhwani\nElham Tabassi\nDan Hendrycks\nJess Whittlestone\nJohn Honovich\nGeoffrey Hinton\nGeoffrey Hinton\nFei-Fei Li\nAbeba Birhane\nShane Legg\nRumman Chowdhury\nYi Zeng\nTimnit Gebru\nKate Crawford\nPushmeet Kohli\nIlya Sutskever\nYejin Choi\nYann LeCun\nInioluwa Deborah Raji\nMax Tegmark\nYoshua Bengio\nEmily M. Bender\nMargaret Mitchell\nPaul Scharre\nJan Leike\nPaul Christiano\nKalika Bali\nStuart Russell\nArvind Narayanan & Sayash Kapoor\nShakir Mohamed\nLead Editor: Naina Bajekal\nProject Team: Andrew R. Chow, Will Henshall, Ayesha Javed and Billy Perrigo\nEditors: Ayesha Javed, Kyla Mandel, Lily Rothman, Oliver Staley, and Karl Vick\nCopy Editors: Mark Hokoda and Jennifer Schiavone\nPhoto Editor: Kaya Berne\nDesigners: D.W. Pine, Lon Tweeten and Victor Williams\nAudience and digital: TIME100 AI\nDario and Daniela Amodei\nDario and Vera Bergengruen, Charlie Campbell, Andrew R. Chow, Alejandro de la Garza, Chad de Guzman, Angela Haupt, Will Henshall, Ayesha Javed, Jeffrey Kluger, Issie Lapowsky, Kyla Mandel, Alice Park, Billy Perrigo, Nik Popli, Astha Rajvanshi, Yasmeen Serhan, Simmone Shah, and Justin Worland\nVideo: Joey Lautrup, Lisa Nho and Erica Solano Samantha Cooney, Annabel Gutterman, Erin McMullen, Kitty Ruskin, Kari Sonde, Nadia Suleman, Juwayriah Wright and Meg Zukin\nWriters:'}, {'url': 'https://www.mygreatlearning.com/blog/ai-researchers-and-leaders/', 'content': 'Demis, as one of the top AI Leaders,\xa0worked in the field of autobiographical memory and amnesia, where he was the co-author of a number of papers that were influential in the field. Ruslan Salakhutdinov\nAssociate Professor, Carnegie Mellon University\nWebsite: http://www.cs.cmu.edu/~rsalakhu/, Twitter: @rsalakhu\nRuslan Salakhutdinov is a Computer Science professor in the Machine Learning Department at Carnegie Mellon University and has previously held the position of the Director of AI Research at Apple. Artificial Intelligence Tutorial for Beginners in 2024 | Learn AI Tutorial from Experts\nTop 20 Applications of Deep Learning in 2024 Across Industries\nBasics of building an Artificial Intelligence Chatbot – 2024\nIOT Career Opportunities: Ultimate Guide 2024\nTop Machine Learning Projects in 2024\nLeave a Comment Cancel Reply\nYour email address will not be published. He is the lead of all the general Artificial Intelligence efforts at Google and even the program AlphaGo which was the first ever to beat a professional at the game of Go. She has received numerous awards and recognition for her work, including the ELLE Magazine’s 2017 Women in Tech, a Global Thinker of 2015 by Foreign Policy and the prestigious “Great Immigrants: The Pride of America” by Carnegie Foundation in 2016.\n'}, {'url': 'https://aimagazine.com/top10/top-10-ai-leaders', 'content': "We celebrate some of the leading women in the AI sector across the UK and Europe as they commit to harnessing the technology to transform businesses…\nTop 10: Women in AI in the US\nBased on our Technology Top 100 Women 2024, we consider some of the leading women in AI across the United States who are committed to AI transformation…\nTop 10: AI Companies in the USA\nAI Magazine considers some of the leading US-based AI companies to watch in 2024, as they continue expanding the capabilities of the technology…\nTop 10: Technology & AI Events in 2024\n He has been named one of MIT Technology Review's Innovators Under 35.\n7: Jeremy Howard\nOrganisation: fast.ai\nPosition: Founder and researcher\nJeremy Howard is a data scientist and entrepreneur who is known for his work in deep learning and machine learning. 10: Alex Smola\nOrganisation: Boson.ai\nPosition: CEO and Co-founder\nAlex Smola, Co-founder and CEO of Boson.ai, computer scientist and former professor at Carnegie Mellon University, is celebrated for his expertise in machine learning and optimisation. The following AI leaders have made significant contributions to the field of AI, from developing new algorithms and models to exploring the social and ethical implications of AI.\n Top 10: Machine Learning Leaders\nTop 10: VR Companies\nAI connects your brand with the world’s leading executives in the fields of AI strategy, machine learning and digitally disruptive technologies – thought leaders and innovators driving this pioneering sector."}, {'url': 'https://www.nature.com/articles/d41586-020-03411-0', 'content': 'King Abdullah International Medical Research Center\nHouston, Texas (US)\nBaylor College of Medicine (BCM)\nHouston, Texas (US)\nBaylor College of Medicine (BCM)\nHouston, Texas (US)\nBaylor College of Medicine (BCM)\nRelated Articles\nArtificial-intelligence research escalates amid calls for caution\nThe race to the top among the world’s leaders in artificial intelligence\nNetwork effect: Visualizing AI connections in the natural sciences\nAI will change the world, so it’s time to change AI\nFour AI technologies that could transform the way we live and work\nSamsung’s head researcher wants human–AI interactions to be a multisensory experience\nSliced, diced and digested: AI-generated science ready in minutes\nAuto articles: An experiment in AI-generated content\nPartner content: Advancing precision medicine using AI and big data\nPartner content: Using AI to accelerate drug discovery\nPartner content: Using AI to make healthcare more human\nPartner content: Strengthening links in the discovery chain\nPartner content: LMU Munich harnesses AI to drive discovery\nPartner content: Related Articles\nArtificial-intelligence research escalates amid calls for caution\nThe race to the top among the world’s leaders in artificial intelligence\nNetwork effect: Visualizing AI connections in the natural sciences\nAI will change the world, so it’s time to change AI\nFour AI technologies that could transform the way we live and work\nSamsung’s head researcher wants human–AI interactions to be a multisensory experience\nSliced, diced and digested: AI-generated science ready in minutes\nAuto articles: An experiment in AI-generated content\nPartner content: Advancing precision medicine using AI and big data\nPartner content: Using AI to accelerate drug discovery\nPartner content: Using AI to make healthcare more human\nPartner content: Strengthening links in the discovery chain\nPartner content: LMU Munich harnesses AI to drive discovery\nPartner content: Breaking AI out of the computer science bubble\nPartner content: Discovering a theory to visualize the world\nPartner content: Supporting the technology game-changers\nPartner content: Data-driven AI illuminates the future\nPartner content: The humans at the heart of AI\nPartner content: New reach for computer imaging\nPartner content: Building natural trust in artificial intelligence\nPartner content: Raising a global centre for deep learning\nPartner content: Japan’s new centre of gravity for clinical data science\nPartner content: AI researchers wanted in Germany\nSubjects\nSign up to Nature Briefing\nAn essential round-up of science news, opinion and analysis, delivered to your inbox every weekday.\n Breaking AI out of the computer science bubble\nPartner content: Discovering a theory to visualize the world\nPartner content: Supporting the technology game-changers\nPartner content: Data-driven AI illuminates the future\nPartner content: The humans at the heart of AI\nPartner content: New reach for computer imaging\nPartner content: Building natural trust in artificial intelligence\nPartner content: Raising a global centre for deep learning\nPartner content: Japan’s new centre of gravity for clinical data science\nPartner content: AI researchers wanted in Germany\nSubjects\nLatest on:\nChatGPT one year on: who is using it, how and why?\n Advertisement\nSix researchers who are shaping the future of artificial intelligence\nYou can also search for this author in PubMed\nGoogle Scholar\nYou can also search for this author in PubMed\nGoogle Scholar\nYou can also search for this author in PubMed\nGoogle Scholar\nAndy Tay is a science writer in Singapore.\n'}, {'url': 'https://www.springboard.com/blog/data-science/ai-machine-learning-experts-to-follow/', 'content': '[2024 Guide]\n12 Best Data Warehousing Training Courses to Grow Your Skills\nData Analytics Bootcamp\nData Science Bootcamp\nData Engineering Bootcamp\nMachine Learning Engineering and AI Bootcamp\nIntroduction to Data Analytics\nData Science Prep\nCybersecurity Bootcamp\nSoftware Engineering Bootcamp\nSoftware Engineering Bootcamp for Beginners\nSoftware Engineering Prep\nUI/UX Design Bootcamp\nUX Design Bootcamp\nIntroduction to Design\nTake our quiz\nAll courses\nHow it works\nMentorship\nJob guarantee\nStudent outcomes\nStudent stories\nCommunity\nPayment options\nScholarships\nUniversities\nBusinesses\nHire our graduates\nAffiliates\nBlog\nCompare bootcamps\nFree courses\nLearn data science\nLearn coding\nLearn cybersecurity\nLearn data analytics\nOur story\nBecome a mentor\nJoin our team\nPress inquiries: Jasmine Kyung\nSenior Operations Engineer at Raytheon Technologies\nRead Story\nBrandon Beidel\nSenior Data Scientist at Red Ventures\nRead Story\nPizon Shetu\nData Scientist at Whiterock AI\nRead Story\nAndrej Karpathy\nSo much fun to wave at other Teslas on the road. Rana el Kaliouby\n.@drfeifei humbled and honored to have my work with @Affectiva be included among the most influential women working in #AI in the USA!\nhttps://t.co/RLspMZOJeL via @teamrework\nCo-founder and CEO of the emotion recognition software company Affectiva, Rana el Kaliouby is consistently named as one of the most influential women in tech. https://t.co/1SCBTnl0Jf\nStanford professor and AI guru Fei-Fei Li was the head of Google Cloud, but left the tech company in late 2018 to head up Stanford’s Human-Centered AI Institute, which is predicated on the idea that the development and shaping of AI technology must be humanistic as well as technological. A believer that AI can help improve lives and reduce global inequalities, Cisse is a co-founder of the group Black in AI, which is a community of AI students, researchers, and experts who aim to increase the representation of black people in the industry.'}]
# Thought:```json
# {
#  "action": "Final Answer",
#  "action_input": "Some popular researchers in the area of Artificial Intelligence include Geoffrey Hinton, Yann LeCun, Yoshua Bengio, Fei-Fei Li, Demis Hassabis, Andrew Ng, and Stuart Russell. These researchers have made significant contributions to the field of AI, and their work has helped to advance the development of AI technologies."
# }
# ```

# > Finished chain.
# The function calculates the length of the input string.
# {'a': {'title': 'A', 'type': 'string'}}


# > Entering new AgentExecutor chain...
# Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?
# Observation: Based on the provided context, Lilian Weng's blog posts cover topics like:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Learning with not Enough Data** (specifically, data generation) 

# Thought:Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?
# Observation: Based on the provided context, Lilian Weng's blog posts cover topics like:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Data Generation (specifically in the context of learning with limited data)** 

# Thought:Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?
# Observation: Based on the provided context, Lilian Weng's blog posts cover topics like:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Learning with not Enough Data** (specifically, data generation) 

# Thought:Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?
# Observation: Based on the provided context, Lilian Weng's blog posts cover the following main research topics:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Data Generation (specifically in the context of "Learning with not Enough Data")** 

# Thought:Retrying langchain_google_genai.chat_models._chat_with_retry.<locals>._chat_with_retry in 2.0 seconds as it raised ResourceExhausted: 429 Resource has been exhausted (e.g. check quota)..
# Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?Retrying langchain_google_genai.chat_models._chat_with_retry.<locals>._chat_with_retry in 2.0 seconds as it raised ResourceExhausted: 429 Resource has been exhausted (e.g. check quota)..

# Observation: Based on the provided context, Lilian Weng's blog posts cover the following research topics:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Learning with not Enough Data** (specifically, data generation) 

# Thought:Retrying langchain_google_genai.chat_models._chat_with_retry.<locals>._chat_with_retry in 2.0 seconds as it raised ResourceExhausted: 429 Resource has been exhausted (e.g. check quota)..
# Retrying langchain_google_genai.chat_models._chat_with_retry.<locals>._chat_with_retry in 4.0 seconds as it raised ResourceExhausted: 429 Resource has been exhausted (e.g. check quota)..
# Retrying langchain_google_genai.chat_models._chat_with_retry.<locals>._chat_with_retry in 8.0 seconds as it raised ResourceExhausted: 429 Resource has been exhausted (e.g. check quota)..
# Retrying langchain_google_genai.chat_models._chat_with_retry.<locals>._chat_with_retry in 16.0 seconds as it raised ResourceExhausted: 429 Resource has been exhausted (e.g. check quota)..
# Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?
# Observation: Based on the provided context, Lilian Weng's blog posts cover these main research topics:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Learning with not Enough Data (specifically, data generation)** 

# Thought:Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?
# Observation: Based on the provided context, Lilian Weng's blog posts cover the following main research topics:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Learning with not Enough Data** (specifically, data generation) 

# Thought:Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?
# Observation: Based on the provided context, Lilian Weng's blog posts cover the following main research topics:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Learning with not Enough Data** (specifically, data generation) 

# Thought:Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?
# Observation: Based on the provided context, Lilian Weng's blog posts cover topics related to:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Learning with not Enough Data** (Specifically, Data Generation) 

# Thought:Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?
# Observation: Based on the provided context, Lilian Weng's blog posts cover the following main research topics:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Data Generation** (specifically in the context of learning with limited data) 

# Thought:Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?
# Observation: Based on the provided context, Lilian Weng's blog posts cover topics like:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Learning with not Enough Data (specifically data generation)** 

# Thought:Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?
# Observation: Based on the provided context, Lilian Weng's blog posts cover topics like:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Data Generation (specifically in the context of learning with limited data)** 

# Thought:Retrying langchain_google_genai.chat_models._chat_with_retry.<locals>._chat_with_retry in 2.0 seconds as it raised ResourceExhausted: 429 Resource has been exhausted (e.g. check quota)..
# Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?Retrying langchain_google_genai.chat_models._chat_with_retry.<locals>._chat_with_retry in 2.0 seconds as it raised ResourceExhausted: 429 Resource has been exhausted (e.g. check quota)..

# Observation: Based on the provided context, Lilian Weng's blog posts cover the following research topics:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Learning with not Enough Data** (specifically, data generation) 

# Thought:Retrying langchain_google_genai.chat_models._chat_with_retry.<locals>._chat_with_retry in 2.0 seconds as it raised ResourceExhausted: 429 Resource has been exhausted (e.g. check quota)..
# Retrying langchain_google_genai.chat_models._chat_with_retry.<locals>._chat_with_retry in 4.0 seconds as it raised ResourceExhausted: 429 Resource has been exhausted (e.g. check quota)..
# Retrying langchain_google_genai.chat_models._chat_with_retry.<locals>._chat_with_retry in 8.0 seconds as it raised ResourceExhausted: 429 Resource has been exhausted (e.g. check quota)..
# Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?
# Observation: Based on the provided context, Lilian Weng's blog posts cover the following research topics:

# * **Prompt Engineering**
# * **Diffusion Models**
# * **Learning with not Enough Data** (specifically, data generation) 

# Thought:Thought: I need to find the research topics covered by Lilian Weng in her blogs. I can use the `lilianweng_search` tool to get this information. 
# Action: lilianweng_search
# Action Input: What are the main research topics covered in Lilian Weng's blog posts?
# Observation: Based on the provided context, Lilian Weng's blog posts cover topics like:

# * **Prompt Engineering:** This post likely discusses techniques and strategies for crafting effective prompts for AI models. 
# * **Diffusion Models:** This post likely explains the concept of diffusion models, which are a type of generative model used in areas like image generation.
# * **Learning with not Enough Data Part 3: Data Generation:** This post likely focuses on methods for generating synthetic data to improve model performance when limited real data is available. 

# These topics suggest that Lilian Weng's blog focuses on **machine learning and artificial intelligence**, specifically on areas like **natural language processing, computer vision, and data augmentation**. 

# Thought:

# > Finished chain.
