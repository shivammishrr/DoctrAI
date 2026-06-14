import os
from unittest.mock import MagicMock, patch
import pytest

from core.model_manager import ModelManager


class TestModelManager:
    def test_init_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                ModelManager()

    def test_init_success(self, mock_env_vars, mock_groq_client):
        mm = ModelManager()
        assert mm.current_model == "llama3-70b-8192"
        assert len(mm.model_configs) == 4

    def test_estimate_token_count_empty(self, mock_env_vars, mock_groq_client):
        mm = ModelManager()
        assert mm._estimate_token_count([]) == 0

    def test_estimate_token_count_simple(self, mock_env_vars, mock_groq_client):
        mm = ModelManager()
        msgs = [{"role": "user", "content": "hello world"}]
        assert mm._estimate_token_count(msgs) == len("hello world") // 4

    def test_get_next_fallback_model_highest_priority(self, mock_env_vars, mock_groq_client):
        mm = ModelManager()
        next_model = mm.get_next_fallback_model("llama3-70b-8192", "high")
        assert next_model == "llama3-8b-8192"

    def test_get_next_fallback_model_no_eligible(self, mock_env_vars, mock_groq_client):
        mm = ModelManager()
        next_model = mm.get_next_fallback_model("gemma-7b-it", "basic")
        assert next_model is None

    def test_truncate_messages_basic(self, mock_env_vars, mock_groq_client):
        mm = ModelManager()
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "A" * 1000},
        ]
        truncated = mm._truncate_messages(msgs, target_token_count=200)
        assert len(truncated) == 2
        assert truncated[0]["role"] == "system"
        assert "[Content truncated" in truncated[1]["content"]

    def test_truncate_messages_only_system(self, mock_env_vars, mock_groq_client):
        mm = ModelManager()
        msgs = [{"role": "system", "content": "System prompt only."}]
        truncated = mm._truncate_messages(msgs, target_token_count=1)
        assert len(truncated) == 1

    def test_truncate_messages_with_tool_calls(self, mock_env_vars, mock_groq_client):
        mm = ModelManager()
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}, "type": "function"}]},
        ]
        truncated = mm._truncate_messages(msgs)
        assert len(truncated) == 3

    def test_create_completion_success(self, mock_env_vars, mock_groq_client):
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Test response"
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_groq_client.chat.completions.create.return_value = mock_response

        mm = ModelManager()
        msg, model = mm.create_completion([{"role": "user", "content": "hello"}])

        assert msg.content == "Test response"
        assert model == "llama3-70b-8192"
        mock_groq_client.chat.completions.create.assert_called_once()

    def test_create_completion_with_tools(self, mock_env_vars, mock_groq_client):
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = []
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_groq_client.chat.completions.create.return_value = mock_response

        mm = ModelManager()
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        msg, model = mm.create_completion([{"role": "user", "content": "use tool"}], tools=tools, tool_choice="auto")

        assert msg is not None
        call_kwargs = mock_groq_client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "auto"

    def test_create_completion_triggers_truncation(self, mock_env_vars, mock_groq_client):
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Test"
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_groq_client.chat.completions.create.return_value = mock_response

        mm = ModelManager()
        long_content = "A" * 10000
        msg, model = mm.create_completion([{"role": "user", "content": long_content}])

        assert msg is not None

    def test_create_completion_all_models_fail(self, mock_env_vars, mock_groq_client):
        mock_groq_client.chat.completions.create.side_effect = Exception("API Error")

        mm = ModelManager()
        msg, model = mm.create_completion([{"role": "user", "content": "hello"}])

        assert msg is None
        assert mock_groq_client.chat.completions.create.call_count >= 1

    def test_fallback_on_context_length(self, mock_env_vars, mock_groq_client):
        mock_groq_client.chat.completions.create.side_effect = [
            Exception("context_length_exceeded"),
            Exception("context_length_exceeded"),
            MagicMock(
                choices=[MagicMock(
                    message=MagicMock(content="Fallback response")
                )]
            ),
        ]

        mm = ModelManager()
        msg, model = mm.create_completion([{"role": "user", "content": "hello"}])

        assert msg is not None
        assert msg.content == "Fallback response"
        assert mock_groq_client.chat.completions.create.call_count == 3

    def test_fallback_on_rate_limit(self, mock_env_vars, mock_groq_client):
        mock_groq_client.chat.completions.create.side_effect = [
            Exception("rate limit exceeded"),
            MagicMock(
                choices=[MagicMock(
                    message=MagicMock(content="Rate limit fallback")
                )]
            ),
        ]

        mm = ModelManager()
        msg, model = mm.create_completion([{"role": "user", "content": "hello"}])

        assert msg is not None
        assert msg.content == "Rate limit fallback"
