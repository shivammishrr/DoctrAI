from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def mock_groq_client():
    with patch("core.model_manager.Groq") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_tavily_client():
    with patch("core.tool_functions.TavilyClient") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_env_vars():
    with patch.dict("os.environ", {
        "GROQ_API_KEY": "test-groq-key",
        "TAVILY_API_KEY": "test-tavily-key",
    }, clear=True):
        yield


@pytest.fixture
def sample_chat_completion():
    def _make(content=None, tool_calls=None):
        message = MagicMock()
        message.content = content
        dumped = {"role": "assistant", "content": content, "tool_calls": None}
        if tool_calls:
            dumped["tool_calls"] = [
                {
                    "id": tc["id"],
                    "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                    "type": "function",
                }
                for tc in tool_calls
            ]
            tcs = []
            for tc in tool_calls:
                tc_mock = MagicMock()
                tc_mock.id = tc["id"]
                tc_mock.function.name = tc["function"]["name"]
                tc_mock.function.arguments = tc["function"]["arguments"]
                tcs.append(tc_mock)
            message.tool_calls = tcs
        else:
            message.tool_calls = None
        message.model_dump.return_value = dumped
        return message
    return _make
