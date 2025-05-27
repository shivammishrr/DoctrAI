import os
from typing import List, Dict, Any, Optional, Tuple
from groq import Groq
from groq.types.chat import ChatCompletionMessage

class ModelManager:
    """Manages model selection, fallback, and API calls for Groq, supporting function calling."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY") # Ensure your .env uses GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        self.client = Groq(api_key=self.api_key)
        self.model_configs = self._initialize_model_configs()
        self.current_model = "llama3-70b-8192"  # Updated default Llama3 model
        self.fallback_attempts = 0
        self.max_fallback_attempts = 3 # Max attempts for a single create_completion call
        self.input_truncation_factor = 0.7

    def _initialize_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize model configurations with their limits and capabilities."""
        # Update with current Groq models and their known (or estimated) limits
        # These limits might change, refer to Groq documentation for up-to-date info
        return {
            "llama3-70b-8192": { # Llama 3 70B
                "context_window": 8192,
                "tokens_per_minute": 6000, # Example, check Groq for actual limits
                "reasoning_level": "high",
                "priority": 1
            },
            "llama3-8b-8192": { # Llama 3 8B
                "context_window": 8192,
                "tokens_per_minute": 15000, # Example
                "reasoning_level": "medium",
                "priority": 2
            },
            "mixtral-8x7b-32768": { # Mixtral
                "context_window": 32768,
                "tokens_per_minute": 6000, # Example
                "reasoning_level": "high",
                "priority": 3
            },
            "gemma-7b-it": { # Gemma
                "context_window": 8192,
                "tokens_per_minute": 15000, # Example
                "reasoning_level": "medium",
                "priority": 4
            }
        }

    def _truncate_messages(self, messages: List[Dict[str, str]], target_token_count: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Truncate message content to reduce token count.
        Prioritizes keeping system messages and truncating user/assistant messages.
        """
        truncated_messages = []
        # Keep system message intact
        if messages and messages[0]["role"] == "system":
            truncated_messages.append(messages[0])
            messages_to_truncate = messages[1:]
        else:
            messages_to_truncate = messages

        # Simple truncation from the end of content for user/assistant messages
        for msg in reversed(messages_to_truncate): # Process newest first for potential full truncation
            if target_token_count and self._estimate_token_count(truncated_messages + [msg]) <= target_token_count:
                truncated_messages.insert(1 if truncated_messages and truncated_messages[0]["role"]=="system" else 0, msg)
                continue

            content = msg.get("content", "")
            if isinstance(content, str) and content: # Ensure content is a non-empty string
                # More aggressive truncation if needed
                truncation_length = int(len(content) * self.input_truncation_factor)
                truncated_content = content[:truncation_length] + "\n\n[Content truncated due to token limits]"
                new_msg = {"role": msg["role"], "content": truncated_content}
                # Add tool_calls if it exists
                if "tool_calls" in msg and msg["tool_calls"] is not None:
                    new_msg["tool_calls"] = msg["tool_calls"]

                truncated_messages.insert(1 if truncated_messages and truncated_messages[0]["role"]=="system" else 0, new_msg)
            elif "tool_calls" in msg and msg["tool_calls"] is not None: # Keep tool call requests
                 truncated_messages.insert(1 if truncated_messages and truncated_messages[0]["role"]=="system" else 0, msg)


        # If still too large, remove oldest non-system messages
        while target_token_count and self._estimate_token_count(truncated_messages) > target_token_count and len(truncated_messages) > (1 if truncated_messages and truncated_messages[0]["role"]=="system" else 0):
            if truncated_messages[0]["role"] == "system" and len(truncated_messages) > 1:
                truncated_messages.pop(1)
            elif truncated_messages[0]["role"] != "system":
                 truncated_messages.pop(0)
            else:
                break # Only system message left

        return truncated_messages


    def _estimate_token_count(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count in messages (rough approximation)."""
        total_chars = 0
        for msg in messages:
            if isinstance(msg.get("content"), str):
                total_chars += len(msg["content"])
            # Could add more complex logic for tool_calls, etc. if needed
        return total_chars // 4 # A very rough estimate: 1 token ≈ 4 characters

    def get_next_fallback_model(self, current_model: str, reasoning_level: str = "high") -> Optional[str]:
        """Get the next available model based on reasoning level and priority."""
        eligible_models = []
        for model_name, config in self.model_configs.items():
            if model_name == current_model:
                continue

            current_reasoning = config["reasoning_level"]
            if reasoning_level == "high" and (current_reasoning == "high" or current_reasoning == "medium"):
                eligible_models.append(model_name)
            elif reasoning_level == "medium" and (current_reasoning == "medium" or current_reasoning == "basic"):
                eligible_models.append(model_name)
            elif reasoning_level == "basic" and current_reasoning == "basic":
                 eligible_models.append(model_name)

        if not eligible_models:
            return None

        # Sort by priority (lower number = higher priority)
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
        """
        Create a completion with Groq, handling token limits and model fallbacks.
        Returns the API response message object and the model name used.
        """
        active_model_name = self.current_model
        attempt = 0
        original_messages = [m.copy() for m in messages] # Deep copy for retries

        while attempt <= self.max_fallback_attempts:
            current_messages_for_api = [m.copy() for m in original_messages] # Use fresh copy for each attempt's modifications
            
            try:
                # Ensure active_model_name is valid
                if active_model_name not in self.model_configs:
                    print(f"Warning: Model {active_model_name} not in configs. Attempting to find a default.")
                    active_model_name = self.get_next_fallback_model(active_model_name, reasoning_level) or list(self.model_configs.keys())[0]
                    if not active_model_name: # Should not happen if configs exist
                         raise Exception("No models available in configuration.")
                    print(f"Switched to model: {active_model_name}")


                model_context_window = self.model_configs[active_model_name]["context_window"]
                
                # Estimate current token count
                estimated_tokens = self._estimate_token_count(current_messages_for_api)

                # Truncate if estimated tokens exceed 90% of context window (leaving room for generation)
                # Max_tokens for generation should also be considered.
                safe_input_tokens = model_context_window - max_tokens - 50 # 50 as buffer
                if estimated_tokens > safe_input_tokens:
                    print(f"Input too large for {active_model_name} (~{estimated_tokens} tokens, needs to be < {safe_input_tokens}). Truncating...")
                    current_messages_for_api = self._truncate_messages(current_messages_for_api, safe_input_tokens)
                    estimated_tokens = self._estimate_token_count(current_messages_for_api)
                    print(f"Truncated input to approximately {estimated_tokens} tokens.")
                    if estimated_tokens > safe_input_tokens:
                        print("Warning: Truncation still resulted in oversized input. API call might fail.")


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
                
                print(f"Attempting API call with {active_model_name}. Tokens: ~{estimated_tokens}. Max_gen: {max_tokens}")
                response = self.client.chat.completions.create(**api_params)
                self.current_model = active_model_name # Update current model if successful
                return response.choices[0].message, active_model_name

            except Exception as e:
                error_str = str(e).lower()
                print(f"Error with model {active_model_name} (Attempt {attempt + 1}): {error_str}")
                attempt += 1

                if "context_length_exceeded" in error_str or "too large" in error_str or "maximum context length" in error_str :
                    print("Context length error. Retrying with truncation or next model.")
                    # Truncation already attempted, try next model if this was the error source
                    next_model = self.get_next_fallback_model(active_model_name, reasoning_level)
                    if next_model:
                        print(f"Falling back to model: {next_model}")
                        active_model_name = next_model
                        original_messages = [m.copy() for m in messages] # Reset messages for new model
                        continue
                    else:
                        print("No fallback model available for context length error.")
                        break # Break while loop

                elif "rate limit" in error_str or "tpm" in error_str or "tpd" in error_str:
                    print("Rate limit error.")
                    next_model = self.get_next_fallback_model(active_model_name, reasoning_level)
                    if next_model:
                        print(f"Falling back to model: {next_model}")
                        active_model_name = next_model
                        original_messages = [m.copy() for m in messages] # Reset messages for new model
                        continue
                    else:
                        print("No fallback model available for rate limit error.")
                        break # Break while loop
                else: # Other API errors
                    print("Other API error.")
                    next_model = self.get_next_fallback_model(active_model_name, reasoning_level)
                    if next_model:
                        print(f"Falling back to model: {next_model}")
                        active_model_name = next_model
                        original_messages = [m.copy() for m in messages] # Reset messages for new model
                        continue
                    else:
                        print("No fallback model available for this error.")
                        break # Break while loop
        
        print(f"Failed to get completion after {self.max_fallback_attempts +1} attempts with available models.")
        return None, active_model_name # Or raise the last exception 