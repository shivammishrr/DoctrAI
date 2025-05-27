import os
from typing import Dict, Any
from langchain.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from dotenv import load_dotenv

# Conditional import for TavilySearchResults
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:
    try:
        from langchain.tools.tavily_search import TavilySearchResults # Older langchain
    except ImportError:
        TavilySearchResults = None
        print("Warning: TavilySearchResults could not be imported. Tavily search tool will not be available.")

from core.model_manager import ModelManager # For Google Search tool

load_dotenv()

class ToolExecutor:
    """Executes the actual Python functions for the tools defined in tool_definitions.py."""

    def __init__(self, model_manager: ModelManager):
        """
        Initializes the ToolExecutor with a ModelManager instance (for tools that make LLM calls)
        and sets up LangChain tools.
        """
        self.model_manager = model_manager
        
        # Initialize LangChain tools
        self.arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=3, load_max_docs=5, doc_content_chars_max=2000))
        self.wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000))
        
        self.tavily_tool = None
        if TavilySearchResults:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if tavily_api_key:
                self.tavily_tool = TavilySearchResults(max_results=5)
            else:
                print("Warning: TAVILY_API_KEY not found. Tavily search tool will be disabled.")
        
        # Dispatch table to map function names (from schemas) to methods
        self.dispatch_table = {
            "tavily_medical_search": self.execute_tavily_medical_search,
            "arxiv_medical_search": self.execute_arxiv_medical_search,
            "wikipedia_medical_search": self.execute_wikipedia_medical_search,
            "google_medical_search": self.execute_google_medical_search,
        }

    def execute_tavily_medical_search(self, query: str) -> str:
        """Executes a Tavily search for medical information."""
        if not self.tavily_tool:
            return "Tavily Search tool is not available or configured."
        try:
            print(f"Executing Tavily Search for: {query}")
            result = self.tavily_tool.invoke({"query": query})
            return str(result)[:4000] # Truncate for safety
        except Exception as e:
            print(f"Error during Tavily search: {e}")
            return f"Error performing Tavily search: {str(e)}"

    def execute_arxiv_medical_search(self, query: str) -> str:
        """Executes an ArXiv search for medical papers."""
        try:
            print(f"Executing ArXiv Search for: {query}")
            result = self.arxiv_tool.invoke(query)
            return str(result)[:4000] # Truncate for safety
        except Exception as e:
            print(f"Error during ArXiv search: {e}")
            return f"Error performing ArXiv search: {str(e)}"

    def execute_wikipedia_medical_search(self, query: str) -> str:
        """Executes a Wikipedia search for medical information."""
        try:
            print(f"Executing Wikipedia Search for: {query}")
            result = self.wikipedia_tool.invoke(query)
            return str(result)[:4000] # Truncate for safety
        except Exception as e:
            print(f"Error during Wikipedia search: {e}")
            return f"Error performing Wikipedia search: {str(e)}"

    def execute_google_medical_search(self, query: str) -> str:
        """Performs a Google search using an LLM to summarize and structure results."""
        try:
            print(f"Executing Google Search (LLM-based) for: {query}")
            search_query = f"medical information about {query}"
            prompt = f"""You are a medical search summarization tool. 
            Based on the query \"{search_query}\", provide a concise summary of key medical facts, 
            common medical perspectives, any recent relevant research (within 2 years if applicable), 
            and list 2-3 links to reputable medical resources (like Mayo Clinic, NIH, CDC, WebMD). 
            Focus on factual, structured information. Limit response to 500 words."""
            
            response_message, model_used = self.model_manager.create_completion(
                messages=[
                    {"role": "system", "content": "You are a medical search engine that returns structured, factual results."},
                    {"role": "user", "content": prompt}
                ],
                reasoning_level="basic",
                temperature=0.3,
                max_tokens=800  # Increased slightly for structured output
            )
            print(f"LLM ({model_used}) used for Google search summarization.")
            if response_message and response_message.content:
                return response_message.content
            return "Could not retrieve a summary from the LLM for Google search."
        except Exception as e:
            print(f"Error performing LLM-based Google search: {e}")
            return f"Error performing Google search: {str(e)}" 