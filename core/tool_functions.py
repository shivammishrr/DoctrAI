import os
from typing import Dict, Any
from langchain.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from dotenv import load_dotenv
from tavily import TavilyClient # <-- Import TavilyClient

from core.model_manager import ModelManager

load_dotenv()

class ToolExecutor:
    """Executes the actual Python functions for the tools defined in tool_definitions.py."""

    def __init__(self, model_manager: ModelManager):
        """
        Initializes the ToolExecutor with a ModelManager instance (for tools that make LLM calls)
        and sets up LangChain and other tools.
        """
        self.model_manager = model_manager
        
        # Initialize LangChain tools
        self.arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=3, load_max_docs=5, doc_content_chars_max=2000))
        self.wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000))
        
        # --- SETUP FOR TAVILY SEARCH ---
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            print("Warning: TAVILY_API_KEY not found. Tavily search tool will be disabled.")
            self.tavily_client = None
        else:
            self.tavily_client = TavilyClient(api_key=self.tavily_api_key)

        # Dispatch table to map function names (from schemas) to methods
        self.dispatch_table = {
            "arxiv_medical_search": self.execute_arxiv_medical_search,
            "wikipedia_medical_search": self.execute_wikipedia_medical_search,
            "tavily_medical_search": self.execute_tavily_medical_search, # <-- Point to the new Tavily function
        }

    # ... execute_arxiv_medical_search and execute_wikipedia_medical_search methods are unchanged ...
    def execute_arxiv_medical_search(self, query: str) -> str:
        """Executes an ArXiv search for medical papers."""
        try:
            print(f"Executing ArXiv Search for: {query}")
            result = self.arxiv_tool.invoke(query)
            print(f"ArXiv Result Preview: {str(result)[:250]}...")
            return str(result)[:4000] # Truncate for safety
        except Exception as e:
            print(f"Error during ArXiv search: {e}")
            return f"Error performing ArXiv search: {str(e)}"

    def execute_wikipedia_medical_search(self, query: str) -> str:
        """Executes a Wikipedia search for medical information."""
        try:
            print(f"Executing Wikipedia Search for: {query}")
            result = self.wikipedia_tool.invoke(query)
            print(f"Wikipedia Result Preview: {str(result)[:250]}...")
            return str(result)[:4000] # Truncate for safety
        except Exception as e:
            print(f"Error during Wikipedia search: {e}")
            return f"Error performing Wikipedia search: {str(e)}"
    
    # Tavily Search
    def execute_tavily_medical_search(self, query: str) -> str:
        """Performs a web search using the Tavily API."""
        if not self.tavily_client:
            return "Tavily Search tool is not available. Please check TAVILY_API_KEY in your .env file."
        
        try:
            print(f"Executing Tavily Search for: {query}")
            
            # Use the Tavily client to search
            # 'search_depth="advanced"' provides more comprehensive results for the LLM
            response = self.tavily_client.search(query=query, search_depth="advanced", max_results=5)
            
            # The 'results' key contains a list of sources with content
            results = response.get('results', [])
            if not results:
                return "No results found from Tavily Search."

            # Format the results for the LLM
            formatted_results = []
            for i, result in enumerate(results):
                title = result.get('title', 'No Title')
                url = result.get('url', '#')
                content = result.get('content', 'No content available.')
                formatted_results.append(f"{i+1}. Title: {title}\n   URL: {url}\n   Content: {content.strip()}\n")
            
            final_result = "\n".join(formatted_results)
            print(f"Tavily Search Result Preview: {final_result[:250]}...")
            return final_result[:8000] # Tavily can be verbose, allow a larger context
            
        except Exception as e:
            print(f"Error during Tavily Search: {e}")
            return f"Error performing Tavily search: {str(e)}"