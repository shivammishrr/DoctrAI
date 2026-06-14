from unittest.mock import MagicMock, patch
import pytest

from core.tool_functions import ToolExecutor


class TestToolExecutor:
    def test_init_with_keys(self, mock_env_vars):
        with patch("core.tool_functions.ArxivQueryRun"), \
             patch("core.tool_functions.WikipediaQueryRun"), \
             patch("core.tool_functions.TavilyClient"):
            mm = MagicMock()
            executor = ToolExecutor(mm)
            assert executor.tavily_client is not None
            assert "arxiv_medical_search" in executor.dispatch_table
            assert "wikipedia_medical_search" in executor.dispatch_table
            assert "tavily_medical_search" in executor.dispatch_table

    def test_init_without_tavily_key(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test"}, clear=True), \
             patch("core.tool_functions.ArxivQueryRun"), \
             patch("core.tool_functions.WikipediaQueryRun"):
            mm = MagicMock()
            executor = ToolExecutor(mm)
            assert executor.tavily_client is None

    def test_execute_arxiv_medical_search_success(self, mock_env_vars):
        mock_arxiv = MagicMock()
        mock_arxiv.invoke.return_value = "ArXiv: Some medical paper about COVID-19."

        with patch("core.tool_functions.ArxivQueryRun", return_value=mock_arxiv), \
             patch("core.tool_functions.WikipediaQueryRun"), \
             patch("core.tool_functions.TavilyClient"):
            mm = MagicMock()
            executor = ToolExecutor(mm)
            result = executor.execute_arxiv_medical_search("COVID-19 treatment")
            assert "COVID-19" in result
            mock_arxiv.invoke.assert_called_once_with("COVID-19 treatment")

    def test_execute_arxiv_medical_search_error(self, mock_env_vars):
        mock_arxiv = MagicMock()
        mock_arxiv.invoke.side_effect = Exception("ArXiv down")

        with patch("core.tool_functions.ArxivQueryRun", return_value=mock_arxiv), \
             patch("core.tool_functions.WikipediaQueryRun"), \
             patch("core.tool_functions.TavilyClient"):
            mm = MagicMock()
            executor = ToolExecutor(mm)
            result = executor.execute_arxiv_medical_search("test")
            assert "Error performing ArXiv search" in result

    def test_execute_wikipedia_medical_search_success(self, mock_env_vars):
        mock_wiki = MagicMock()
        mock_wiki.invoke.return_value = "Wikipedia: Headache is a common condition."

        with patch("core.tool_functions.WikipediaQueryRun", return_value=mock_wiki), \
             patch("core.tool_functions.ArxivQueryRun"), \
             patch("core.tool_functions.TavilyClient"):
            mm = MagicMock()
            executor = ToolExecutor(mm)
            result = executor.execute_wikipedia_medical_search("headache")
            assert "Headache" in result

    def test_execute_tavily_medical_search_success(self, mock_env_vars):
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {
            "results": [
                {"title": "Medical News", "url": "https://example.com", "content": "Latest medical findings."}
            ]
        }

        with patch("core.tool_functions.TavilyClient", return_value=mock_tavily), \
             patch("core.tool_functions.ArxivQueryRun"), \
             patch("core.tool_functions.WikipediaQueryRun"):
            mm = MagicMock()
            executor = ToolExecutor(mm)
            result = executor.execute_tavily_medical_search("latest treatments")
            assert "Medical News" in result
            mock_tavily.search.assert_called_once_with(
                query="latest treatments", search_depth="advanced", max_results=5
            )

    def test_execute_tavily_medical_search_no_results(self, mock_env_vars):
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {"results": []}

        with patch("core.tool_functions.TavilyClient", return_value=mock_tavily), \
             patch("core.tool_functions.ArxivQueryRun"), \
             patch("core.tool_functions.WikipediaQueryRun"):
            mm = MagicMock()
            executor = ToolExecutor(mm)
            result = executor.execute_tavily_medical_search("unknown condition")
            assert "No results found" in result

    def test_execute_tavily_medical_search_no_client(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test"}, clear=True), \
             patch("core.tool_functions.ArxivQueryRun"), \
             patch("core.tool_functions.WikipediaQueryRun"):
            mm = MagicMock()
            executor = ToolExecutor(mm)
            result = executor.execute_tavily_medical_search("test")
            assert "not available" in result

    def test_execute_tavily_medical_search_error(self, mock_env_vars):
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = Exception("API Error")

        with patch("core.tool_functions.TavilyClient", return_value=mock_tavily), \
             patch("core.tool_functions.ArxivQueryRun"), \
             patch("core.tool_functions.WikipediaQueryRun"):
            mm = MagicMock()
            executor = ToolExecutor(mm)
            result = executor.execute_tavily_medical_search("test")
            assert "Error performing Tavily search" in result

    def test_dispatch_table_all_tools_present(self, mock_env_vars):
        with patch("core.tool_functions.ArxivQueryRun"), \
             patch("core.tool_functions.WikipediaQueryRun"), \
             patch("core.tool_functions.TavilyClient"):
            mm = MagicMock()
            executor = ToolExecutor(mm)
            assert callable(executor.dispatch_table["arxiv_medical_search"])
            assert callable(executor.dispatch_table["wikipedia_medical_search"])
            assert callable(executor.dispatch_table["tavily_medical_search"])
