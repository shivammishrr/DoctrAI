from typing import List, Dict, Any

ASK_CLARIFYING_QUESTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_clarifying_question",
        "description": "Ask the user a targeted follow-up question to gather more information or clarify ambiguity before providing your final answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarifying question to ask the user."
                }
            },
            "required": ["question"]
        }
    }
}

FINAL_ANSWER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "FinalAnswer",
        "description": "Provide your final, conclusive response to the user. Call this when you have sufficient information or have completed your research.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "The comprehensive final answer to provide to the user, including appropriate medical disclaimers."
                }
            },
            "required": ["summary"]
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

TAVILY_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "tavily_medical_search",
        "description": "Performs a comprehensive web search using the Tavily API for up-to-date medical information, recent news, or when other specialized tools are insufficient.",
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

CONVERSATION_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    ASK_CLARIFYING_QUESTION_SCHEMA,
    FINAL_ANSWER_SCHEMA,
]

RESEARCH_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    ARXIV_SEARCH_SCHEMA,
    WIKIPEDIA_SEARCH_SCHEMA,
    TAVILY_SEARCH_SCHEMA,
]

ALL_TOOL_SCHEMAS: List[Dict[str, Any]] = CONVERSATION_TOOL_SCHEMAS + RESEARCH_TOOL_SCHEMAS
