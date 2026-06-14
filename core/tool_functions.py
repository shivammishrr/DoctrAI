import os
import logging
from typing import Dict, Any
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from tavily import TavilyClient

from core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

        self.arxiv_tool = ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper(top_k_results=3, load_max_docs=5, doc_content_chars_max=2000)
        )
        self.wikipedia_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
        )

        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY not found. Tavily search tool will be disabled.")
            self.tavily_client = None
        else:
            self.tavily_client = TavilyClient(api_key=self.tavily_api_key)

        self.dispatch_table = {
            "arxiv_medical_search": self.execute_arxiv_medical_search,
            "wikipedia_medical_search": self.execute_wikipedia_medical_search,
            "tavily_medical_search": self.execute_tavily_medical_search,
        }

    def execute_arxiv_medical_search(self, query: str) -> str:
        try:
            logger.info(f"ArXiv Search: {query}")
            result = self.arxiv_tool.invoke(query)
            return str(result)[:4000]
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return f"Error performing ArXiv search: {str(e)}"

    def execute_wikipedia_medical_search(self, query: str) -> str:
        try:
            logger.info(f"Wikipedia Search: {query}")
            result = self.wikipedia_tool.invoke(query)
            return str(result)[:4000]
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return f"Error performing Wikipedia search: {str(e)}"

    def execute_tavily_medical_search(self, query: str) -> str:
        if not self.tavily_client:
            return "Tavily Search tool is not available. Check TAVILY_API_KEY."
        try:
            logger.info(f"Tavily Search: {query}")
            response = self.tavily_client.search(query=query, search_depth="advanced", max_results=5)
            results = response.get('results', [])
            if not results:
                return "No results found from Tavily Search."

            formatted_results = []
            for i, result in enumerate(results):
                title = result.get('title', 'No Title')
                url = result.get('url', '#')
                content = result.get('content', 'No content available.')
                formatted_results.append(
                    f"{i+1}. Title: {title}\n   URL: {url}\n   Content: {content.strip()}\n"
                )

            final_result = "\n".join(formatted_results)
            return final_result[:8000]
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return f"Error performing Tavily search: {str(e)}"
