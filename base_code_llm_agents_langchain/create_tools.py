from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from typing import List
import llm_init as llm_initials

# Initialize LLM for analysis and subtask breakdown
llm_openai = llm_initials.init_openai()
llm_claude = llm_initials.init_claude()
llm_gemini = llm_initials.init_gemini()

def create_ddg_search_tool():
    ddg_search = llm_initials.init_duckduckgo()
    return Tool(
        name="DuckDuckGo Search",
        func=ddg_search.run,
        description="Useful for when you need to answer questions about current events or general information."
    )

def create_tavily_search_tool():
    tavily_search = llm_initials.init_tavily()
    return Tool(
        name="Tavily Search",
        func=tavily_search.run,
        description="Useful for when you need to perform a comprehensive web search with high-quality results."
    )

def create_google_search_tool():
    google_search = llm_initials.init_googlesearch()
    return Tool(
        name="Google Search",
        func=google_search.run,
        description="Useful for when you need to find specific information or recent web results."
    )

def create_analysis_tool():
    def analyze_text(text: str) -> str:
        llm = llm_gemini
        """Analyze the given text and provide insights."""
        template = """
        Analyze the following text and provide key insights:
        
        Text: {text}
        
        Analysis:
        1. Main topic:
        2. Key points:
        3. Tone and sentiment:
        4. Potential implications:
        5. Additional observations:
        """
        prompt = PromptTemplate(template=template, input_variables=["text"])
        return llm.invoke(prompt.format(text=text)).content
    
    return Tool(
        name="Text Analysis",
        func=analyze_text,
        description="Useful for analyzing and extracting insights from a given text."
    )


def create_break_into_subtasks_tool():
    """Create and return the Break into Subtasks Tool"""
    def break_into_subtasks(prompt: str) -> List[str]:
        """Break a given prompt into subtasks."""
        llm = llm_gemini 
        template = """
        Break the following prompt into smaller, manageable subtasks:
        
        Prompt: {prompt}
        
        Subtasks:
        1.
        2.
        3.
        ...
        
        Please list at least 3 subtasks, but no more than 7.
        """
        prompt_template = PromptTemplate(template=template, input_variables=["prompt"])
        response = llm.invoke(prompt_template.format(prompt=prompt))
        # Extract subtasks from the response
        subtasks = [line.strip() for line in response.content.split('\n') if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.'))]
        return subtasks

    return Tool(
        name="Break into Subtasks",
        func=break_into_subtasks,
        description="Useful for breaking a complex prompt or task into smaller, manageable subtasks. Input should be a string containing the prompt to break down."
    )
