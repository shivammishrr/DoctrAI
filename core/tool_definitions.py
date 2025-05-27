from typing import List, Dict, Any

# Tool schemas for Groq function calling

TAVILY_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "tavily_medical_search",
        "description": "Performs a targeted medical search using Tavily Search API. Use for finding recent medical papers, clinical trials, or specific medical information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The medical search query."
                }
            },
            "required": ["query"]
        }
    }
}

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

GOOGLE_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "google_medical_search",
        "description": "Performs a general Google search for medical information, prioritizing reputable sources. Use when other specialized tools don\'t yield sufficient results or for very broad queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The medical search query for Google."
                }
            },
            "required": ["query"]
        }
    }
}

# List of all available tool schemas
ALL_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    TAVILY_SEARCH_SCHEMA,
    ARXIV_SEARCH_SCHEMA,
    WIKIPEDIA_SEARCH_SCHEMA,
    GOOGLE_SEARCH_SCHEMA
] 