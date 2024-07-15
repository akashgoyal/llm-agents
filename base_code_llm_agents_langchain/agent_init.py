from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import llm_init as llm_initials
import create_tools as agent_tools

def init_agent_ddg_subtask():
    """Initialize an agent with DuckDuckGo search and subtask breakdown tools"""
    
    llm = llm_initials.init_gemini()
    ddg_search_tool = agent_tools.create_ddg_search_tool()
    subtask_tool = agent_tools.create_break_into_subtasks_tool()
    
    tools = [ddg_search_tool, subtask_tool]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )
    
    return agent

def init_agent_google_analysis():   
    """Initialize an agent with Google search and analysis tools"""

    llm = llm_initials.init_claude()
    google_search_tool = agent_tools.create_google_search_tool()
    analysis_tool = agent_tools.create_analysis_tool()
    
    tools = [google_search_tool, analysis_tool]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )
    return agent

def init_agent_tavily_all_tools():
    """Initialize an agent with Tavily search and all other tools"""
    
    llm = llm_initials.init_openai()
    tavily_search_tool = agent_tools.create_tavily_search_tool()
    analysis_tool = agent_tools.create_analysis_tool()
    subtask_tool = agent_tools.create_break_into_subtasks_tool()
    
    tools = [tavily_search_tool, analysis_tool, subtask_tool]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )
    
    return agent


# Usage example
if __name__ == "__main__":
    print("Initializing agents...")
    
    agent1 = init_agent_ddg_subtask()
    print("Agent 1 (DuckDuckGo + Subtask) initialized")
    
    # agent2 = init_agent_google_analysis()
    # print("Agent 2 (Google + Analysis) initialized")
    
    # agent3 = init_agent_tavily_all_tools()
    # print("Agent 3 (Tavily + All Tools) initialized")
    
    print("All agents initialized successfully!")

    # Example usage of an agent
    agent1_response = agent1.run("What are the latest developments in AI?")
    print("Agent 1 response:", agent1_response)

# > Entering new AgentExecutor chain...
# ```json
# {
#  "action": "DuckDuckGo Search",
#  "action_input": "latest developments in AI"
# }
# ```
# Observation: Learn how generative AI is evolving and integrating into business and everyday life in 2024. Discover the latest developments in multimodal AI, open source models, GPU shortages, regulation and more. AI for everything: 10 Breakthrough Technologies 2024 Generative AI tools like ChatGPT reached mass adoption in record time, and reset the course of an entire industry. The web page predicts four trends in artificial intelligence for the year 2024, such as customized chatbots, generative video, AI-generated election disinformation, and AI-powered law enforcement. It also reviews the performance of its predictions for 2023 and the challenges and opportunities of AI in various domains. Artificial Intelligence News. Everything on AI including futuristic robots with artificial intelligence, computer models of human intelligence and more. MIT-Takeda Program wraps up with 16 publications, a patent, and nearly two dozen projects completed. The program focused on AI in health care, drawing on Takeda's R&D experience in drug development and MIT's deep expertise in AI. June 18, 2024. Read full story.
# Thought:```json
# {
#  "action": "Final Answer",
#  "action_input": "The latest developments in AI include the widespread adoption of generative AI tools like ChatGPT, leading to advancements in customized chatbots, generative video, and AI-powered law enforcement. There are also concerns about AI-generated election disinformation. Additionally, there are ongoing efforts to integrate AI into business and everyday life, as well as research into multimodal AI, open source models, and AI in healthcare."
# }
# ```

# > Finished chain.
# Agent 1 response: The latest developments in AI include the widespread adoption of generative AI tools like ChatGPT, leading to advancements in customized chatbots, generative video, and AI-powered law enforcement. There are also concerns about AI-generated election disinformation. Additionally, there are ongoing efforts to integrate AI into business and everyday life, as well as research into multimodal AI, open source models, and AI in healthcare.


    # Example usage of an agent
    agent1_response = agent1.run("I want to build an LLM agent with three tools. One tool will analyse input, second will make sure response is ethical, third will do formatting of output. What components are involved and how to decide?")
    print("Agent 1 response:", agent1_response)

# > Entering new AgentExecutor chain...
# ```json
# {
#   "action": "Break into Subtasks",
#   "action_input": "Building an LLM agent with three tools: input analysis, ethical response, and output formatting. What components are involved in each tool and how to decide on implementation details?"
# }
# ```
# Observation: []
# Thought:```json
# {
#   "action": "Final Answer",
#   "action_input": "I understand you're interested in building an LLM agent with three specific tools: input analysis, ethical response, and output formatting.  To break down the components and decision-making process for each, let's explore them individually:\n\n**1. Input Analysis Tool:**\n\n* **Components:**\n    * **Tokenization:** Breaking down the input text into smaller units (tokens) for processing.\n    * **Part-of-Speech Tagging:** Identifying the grammatical role of each word (e.g., noun, verb, adjective).\n    * **Named Entity Recognition (NER):** Identifying and classifying entities (e.g., people, organizations, locations) in the text.\n    * **Sentiment Analysis:** Determining the overall emotional tone of the input (e.g., positive, negative, neutral).\n    * **Intent Classification:** Understanding the user's goal or purpose behind the input.\n* **Decision-Making:**\n    * **Choose the right tokenization method:** Consider the language and specific requirements of your agent.\n    * **Select appropriate tagging and NER models:**  Choose models trained on relevant data and domain knowledge.\n    * **Optimize sentiment analysis for your application:**  Use a model that aligns with your needs (e.g., fine-tuning a general-purpose model on your specific domain).\n    * **Develop a robust intent classification system:**  Train a model on a diverse set of user inputs to accurately identify intents.\n\n**2. Ethical Response Tool:**\n\n* **Components:**\n    * **Bias Detection:** Identifying potential biases in the generated response.\n    * **Fairness Evaluation:** Assessing the fairness and equity of the response.\n    * **Safety and Harm Detection:** Identifying potentially harmful or offensive content.\n    * **Ethical Guidelines:** Defining and implementing a set of ethical guidelines for response generation.\n* **Decision-Making:**\n    * **Choose appropriate bias detection techniques:**  Use methods that align with your ethical priorities.\n    * **Develop robust fairness evaluation metrics:**  Define metrics that accurately assess the fairness of responses.\n    * **Implement strong safety and harm detection mechanisms:**  Use reliable methods to filter out harmful content.\n    * **Establish clear ethical guidelines:**  Define a set of principles and values to guide response generation.\n\n**3. Output Formatting Tool:**\n\n* **Components:**\n    * **Text Formatting:**  Applying formatting rules (e.g., line breaks, indentation, bolding) to enhance readability.\n    * **Code Generation:** Generating structured code based on the input and response.\n    * **Visualization:** Creating charts, graphs, or other visual representations of the output.\n    * **Multimodal Generation:** Combining text, images, audio, or video in the output.\n* **Decision-Making:**\n    * **Choose appropriate formatting styles:**  Align the formatting with the target audience and application.\n    * **Select the right code generation framework:**  Choose a framework that supports the required programming languages.\n    * **Decide on the most effective visualization techniques:**  Use visualizations that effectively communicate the output.\n    * **Determine the best approach for multimodal generation:**  Consider the purpose and context of the multimodal output."
# }
# ```

# > Finished chain.
# Agent 1 response: I understand you're interested in building an LLM agent with three specific tools: input analysis, ethical response, and output formatting.  To break down the components and decision-making process for each, let's explore them individually:

# **1. Input Analysis Tool:**

# * **Components:**
#     * **Tokenization:** Breaking down the input text into smaller units (tokens) for processing.
#     * **Part-of-Speech Tagging:** Identifying the grammatical role of each word (e.g., noun, verb, adjective).
#     * **Named Entity Recognition (NER):** Identifying and classifying entities (e.g., people, organizations, locations) in the text.
#     * **Sentiment Analysis:** Determining the overall emotional tone of the input (e.g., positive, negative, neutral).
#     * **Intent Classification:** Understanding the user's goal or purpose behind the input.
# * **Decision-Making:**
#     * **Choose the right tokenization method:** Consider the language and specific requirements of your agent.
#     * **Select appropriate tagging and NER models:**  Choose models trained on relevant data and domain knowledge.
#     * **Optimize sentiment analysis for your application:**  Use a model that aligns with your needs (e.g., fine-tuning a general-purpose model on your specific domain).
#     * **Develop a robust intent classification system:**  Train a model on a diverse set of user inputs to accurately identify intents.

# **2. Ethical Response Tool:**

# * **Components:**
#     * **Bias Detection:** Identifying potential biases in the generated response.
#     * **Fairness Evaluation:** Assessing the fairness and equity of the response.
#     * **Safety and Harm Detection:** Identifying potentially harmful or offensive content.
#     * **Ethical Guidelines:** Defining and implementing a set of ethical guidelines for response generation.
# * **Decision-Making:**
#     * **Choose appropriate bias detection techniques:**  Use methods that align with your ethical priorities.
#     * **Develop robust fairness evaluation metrics:**  Define metrics that accurately assess the fairness of responses.
#     * **Implement strong safety and harm detection mechanisms:**  Use reliable methods to filter out harmful content.
#     * **Establish clear ethical guidelines:**  Define a set of principles and values to guide response generation.

# **3. Output Formatting Tool:**

# * **Components:**
#     * **Text Formatting:**  Applying formatting rules (e.g., line breaks, indentation, bolding) to enhance readability.
#     * **Code Generation:** Generating structured code based on the input and response.
#     * **Visualization:** Creating charts, graphs, or other visual representations of the output.
#     * **Multimodal Generation:** Combining text, images, audio, or video in the output.
# * **Decision-Making:**
#     * **Choose appropriate formatting styles:**  Align the formatting with the target audience and application.
#     * **Select the right code generation framework:**  Choose a framework that supports the required programming languages.
#     * **Decide on the most effective visualization techniques:**  Use visualizations that effectively communicate the output.
#     * **Determine the best approach for multimodal generation:**  Consider the purpose and context of the multimodal output.