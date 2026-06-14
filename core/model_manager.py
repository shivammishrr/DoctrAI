import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from groq import Groq
from groq.types.chat import ChatCompletionMessage

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        self.client = Groq(api_key=self.api_key)
        self.model_configs = self._initialize_model_configs()
        self.current_model = "llama3-70b-8192"
        self.fallback_attempts = 0
        self.max_fallback_attempts = 3
        self.input_truncation_factor = 0.7

    def _initialize_model_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "llama3-70b-8192": {
                "context_window": 8192,
                "tokens_per_minute": 6000,
                "reasoning_level": "high",
                "priority": 1
            },
            "llama3-8b-8192": {
                "context_window": 8192,
                "tokens_per_minute": 15000,
                "reasoning_level": "medium",
                "priority": 2
            },
            "mixtral-8x7b-32768": {
                "context_window": 32768,
                "tokens_per_minute": 6000,
                "reasoning_level": "high",
                "priority": 3
            },
            "gemma-7b-it": {
                "context_window": 8192,
                "tokens_per_minute": 15000,
                "reasoning_level": "medium",
                "priority": 4
            }
        }

    def _truncate_messages(self, messages: List[Dict[str, str]], target_token_count: Optional[int] = None) -> List[Dict[str, str]]:
        truncated_messages = []
        if messages and messages[0]["role"] == "system":
            truncated_messages.append(messages[0])
            messages_to_truncate = messages[1:]
        else:
            messages_to_truncate = messages

        for msg in reversed(messages_to_truncate):
            if target_token_count and self._estimate_token_count(truncated_messages + [msg]) <= target_token_count:
                insert_pos = 1 if (truncated_messages and truncated_messages[0]["role"] == "system") else 0
                truncated_messages.insert(insert_pos, msg)
                continue

            content = msg.get("content", "")
            if isinstance(content, str) and content:
                truncation_length = int(len(content) * self.input_truncation_factor)
                truncated_content = content[:truncation_length] + "\n\n[Content truncated due to token limits]"
                new_msg = {"role": msg["role"], "content": truncated_content}
                if "tool_calls" in msg and msg["tool_calls"] is not None:
                    new_msg["tool_calls"] = msg["tool_calls"]
                insert_pos = 1 if (truncated_messages and truncated_messages[0]["role"] == "system") else 0
                truncated_messages.insert(insert_pos, new_msg)
            elif "tool_calls" in msg and msg["tool_calls"] is not None:
                insert_pos = 1 if (truncated_messages and truncated_messages[0]["role"] == "system") else 0
                truncated_messages.insert(insert_pos, msg)

        while (target_token_count
               and self._estimate_token_count(truncated_messages) > target_token_count
               and len(truncated_messages) > (1 if (truncated_messages and truncated_messages[0]["role"] == "system") else 0)):
            if truncated_messages[0]["role"] == "system" and len(truncated_messages) > 1:
                truncated_messages.pop(1)
            elif truncated_messages[0]["role"] != "system":
                truncated_messages.pop(0)
            else:
                break

        return truncated_messages

    def _estimate_token_count(self, messages: List[Dict[str, Any]]) -> int:
        total_chars = 0
        for msg in messages:
            if isinstance(msg.get("content"), str):
                total_chars += len(msg["content"])
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if hasattr(tc, "function") and hasattr(tc.function, "arguments"):
                        total_chars += len(tc.function.arguments)
                    elif isinstance(tc, dict) and tc.get("function"):
                        total_chars += len(tc["function"].get("arguments", ""))
        return total_chars // 4

    def get_next_fallback_model(self, current_model: str, reasoning_level: str = "high") -> Optional[str]:
        eligible_models = []
        for model_name, config in self.model_configs.items():
            if model_name == current_model:
                continue
            current_reasoning = config["reasoning_level"]
            if reasoning_level == "high" and current_reasoning in ("high", "medium"):
                eligible_models.append(model_name)
            elif reasoning_level == "medium" and current_reasoning in ("medium", "basic"):
                eligible_models.append(model_name)
            elif reasoning_level == "basic" and current_reasoning == "basic":
                eligible_models.append(model_name)

        if not eligible_models:
            return None

        eligible_models.sort(key=lambda m: self.model_configs[m]["priority"])
        return eligible_models[0]

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        reasoning_level: str = "high",
        temperature: float = 0.5,
        max_tokens: int = 1500,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Tuple[Optional[ChatCompletionMessage], str]:
        active_model_name = self.current_model
        attempt = 0
        original_messages = [m.copy() for m in messages]

        while attempt <= self.max_fallback_attempts:
            current_messages_for_api = [m.copy() for m in original_messages]

            try:
                if active_model_name not in self.model_configs:
                    logger.warning(f"Model {active_model_name} not in configs. Finding fallback.")
                    active_model_name = self.get_next_fallback_model(active_model_name, reasoning_level) or list(self.model_configs.keys())[0]
                    if not active_model_name:
                        raise Exception("No models available in configuration.")
                    logger.info(f"Switched to model: {active_model_name}")

                model_context_window = self.model_configs[active_model_name]["context_window"]
                estimated_tokens = self._estimate_token_count(current_messages_for_api)
                safe_input_tokens = model_context_window - max_tokens - 50

                if estimated_tokens > safe_input_tokens:
                    logger.info(f"Input too large for {active_model_name} (~{estimated_tokens} tokens). Truncating...")
                    current_messages_for_api = self._truncate_messages(current_messages_for_api, safe_input_tokens)
                    estimated_tokens = self._estimate_token_count(current_messages_for_api)
                    logger.info(f"Truncated to ~{estimated_tokens} tokens.")
                    if estimated_tokens > safe_input_tokens:
                        logger.warning("Truncation still oversized. API call may fail.")

                api_params: Dict[str, Any] = {
                    "messages": current_messages_for_api,
                    "model": active_model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if tools:
                    api_params["tools"] = tools
                if tool_choice:
                    api_params["tool_choice"] = tool_choice

                logger.debug(f"API call with {active_model_name}. Tokens: ~{estimated_tokens}. Max_gen: {max_tokens}")
                response = self.client.chat.completions.create(**api_params)
                self.current_model = active_model_name
                return response.choices[0].message, active_model_name

            except Exception as e:
                error_str = str(e).lower()
                logger.warning(f"Error with {active_model_name} (Attempt {attempt + 1}): {error_str}")
                attempt += 1

                should_fallback = (
                    "context_length_exceeded" in error_str
                    or "too large" in error_str
                    or "maximum context length" in error_str
                    or "rate limit" in error_str
                    or "tpm" in error_str
                    or "tpd" in error_str
                )

                if should_fallback:
                    next_model = self.get_next_fallback_model(active_model_name, reasoning_level)
                    if next_model:
                        logger.info(f"Falling back to model: {next_model}")
                        active_model_name = next_model
                        original_messages = [m.copy() for m in messages]
                        continue
                    else:
                        logger.warning("No fallback model available.")
                        break
                else:
                    next_model = self.get_next_fallback_model(active_model_name, reasoning_level)
                    if next_model:
                        logger.info(f"Falling back to model: {next_model}")
                        active_model_name = next_model
                        original_messages = [m.copy() for m in messages]
                        continue
                    else:
                        logger.warning("No fallback model available for this error.")
                        break

        logger.error(f"Failed after {self.max_fallback_attempts + 1} attempts.")
        return None, active_model_name
