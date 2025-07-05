from typing import List, Dict, Any

# Tool schemas for Groq function calling

ARXIV_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "arxiv_medical_search",
        "description": "Searches ArXiv for medical research papers and preprints. Useful for cutting-edge research not yet peer-reviewed.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for ArXiv (e.g., specific medical topics, techniques, or author names)."
                }
            },
            "required": ["query"]
        }
    }
}

WIKIPEDIA_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wikipedia_medical_search",
        "description": "Searches Wikipedia for general medical information, definitions, and overviews of conditions or treatments.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The medical term or topic to search on Wikipedia."
                }
            },
            "required": ["query"]
        }
    }
}

TAVILY_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "tavily_medical_search",
        "description": "Performs a comprehensive web search using the Tavily API, optimized for providing concise and relevant answers to medical questions. Use this for up-to-date information, recent news, or when other specialized tools are insufficient.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The medical search query for Tavily."
                }
            },
            "required": ["query"]
        }
    }
}

# List of all available tool schemas
ALL_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    ARXIV_SEARCH_SCHEMA,
    WIKIPEDIA_SEARCH_SCHEMA,
    TAVILY_SEARCH_SCHEMA
]