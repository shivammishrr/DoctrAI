from core.tool_definitions import (
    CONVERSATION_TOOL_SCHEMAS,
    RESEARCH_TOOL_SCHEMAS,
    ALL_TOOL_SCHEMAS,
    ASK_CLARIFYING_QUESTION_SCHEMA,
    FINAL_ANSWER_SCHEMA,
    ARXIV_SEARCH_SCHEMA,
    WIKIPEDIA_SEARCH_SCHEMA,
    TAVILY_SEARCH_SCHEMA,
)


class TestToolDefinitions:
    def test_conversation_tools_count(self):
        assert len(CONVERSATION_TOOL_SCHEMAS) == 2

    def test_research_tools_count(self):
        assert len(RESEARCH_TOOL_SCHEMAS) == 3

    def test_all_tools_count(self):
        assert len(ALL_TOOL_SCHEMAS) == 5

    def test_ask_clarifying_question_schema(self):
        name = ASK_CLARIFYING_QUESTION_SCHEMA["function"]["name"]
        assert name == "ask_clarifying_question"
        props = ASK_CLARIFYING_QUESTION_SCHEMA["function"]["parameters"]["properties"]
        assert "question" in props
        assert "required" in ASK_CLARIFYING_QUESTION_SCHEMA["function"]["parameters"]

    def test_final_answer_schema(self):
        name = FINAL_ANSWER_SCHEMA["function"]["name"]
        assert name == "FinalAnswer"
        props = FINAL_ANSWER_SCHEMA["function"]["parameters"]["properties"]
        assert "summary" in props

    def test_arxiv_schema(self):
        name = ARXIV_SEARCH_SCHEMA["function"]["name"]
        assert name == "arxiv_medical_search"
        props = ARXIV_SEARCH_SCHEMA["function"]["parameters"]["properties"]
        assert "query" in props

    def test_wikipedia_schema(self):
        name = WIKIPEDIA_SEARCH_SCHEMA["function"]["name"]
        assert name == "wikipedia_medical_search"
        props = WIKIPEDIA_SEARCH_SCHEMA["function"]["parameters"]["properties"]
        assert "query" in props

    def test_tavily_schema(self):
        name = TAVILY_SEARCH_SCHEMA["function"]["name"]
        assert name == "tavily_medical_search"
        props = TAVILY_SEARCH_SCHEMA["function"]["parameters"]["properties"]
        assert "query" in props

    def test_all_schemas_have_required_fields(self):
        for schema in ALL_TOOL_SCHEMAS:
            assert "type" in schema
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]
            assert "properties" in schema["function"]["parameters"]
