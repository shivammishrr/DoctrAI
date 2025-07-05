import re
import json
import threading
from queue import Queue
from typing import Callable, Optional, Dict, Any

from core.model_manager import ModelManager
from core.research_orchestrator import MedicalResearchOrchestrator
from core.prompts import get_persona_prompt

class DoctorAI:
    """
    Manages conversational interactions and orchestrates deep medical research.
    It now has two distinct entry points:
    1. process_turn(): For standard ReAct-based conversational turns.
    2. start_deep_research(): For directly initiating a non-blocking research task.
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.research_orchestrator = MedicalResearchOrchestrator(self.model_manager)
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def _get_or_create_session(self, session_id: str, persona: str) -> Dict[str, Any]:
        """Retrieves or initializes a session with a persona-specific prompt."""
        if session_id not in self.sessions:
            system_prompt = get_persona_prompt(persona)
            self.sessions[session_id] = {
                "history": [{"role": "system", "content": system_prompt}]
            }
        return self.sessions[session_id]

    def _parse_react_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parses the LLM's ReAct response into a structured dictionary.
        This parser is designed to be robust against extra text added by the LLM.
        """
        try:
            thought_match = re.search(r"Thought: (.*?)\nAction:", response_text, re.DOTALL)
            action_match = re.search(r"Action: (.*?)\nAction Input:", response_text, re.DOTALL)
            action_input_prefix_match = re.search(r"Action Input:", response_text, re.DOTALL)

            if not (thought_match and action_match and action_input_prefix_match):
                print(f"Warning: ReAct response parsing failed. Key markers not found. Response: {response_text}")
                # MODIFIED: Instead of returning None, return the raw text as a final answer.
                # This makes the agent more resilient if it fails to format correctly.
                return {"type": "final_answer", "content": response_text}

            json_start_index = action_input_prefix_match.end()
            json_str = response_text[json_start_index:].strip()
            
            first_brace = json_str.find('{')
            if first_brace == -1: return None
            json_str = json_str[first_brace:]

            brace_count = 0
            end_index = -1
            for i, char in enumerate(json_str):
                if char == '{': brace_count += 1
                elif char == '}': brace_count -= 1
                if brace_count == 0:
                    end_index = i + 1
                    break
            
            if end_index == -1: return None

            isolated_json_str = json_str[:end_index]
            action_input = json.loads(isolated_json_str)
            
            return {
                "thought": thought_match.group(1).strip(),
                "action": action_match.group(1).strip(),
                "action_input": action_input,
            }
        except Exception as e:
            print(f"Error parsing ReAct response: {e}\nResponse text: {response_text}")
            # MODIFIED: Graceful fallback
            return {"type": "final_answer", "content": response_text}

    # NEW: Dedicated method to start deep research directly from the UI.
    def start_deep_research(self, session_id: str, query: str, persona: str, progress_queue: Queue):
        """
        Initiates a deep research task in a non-blocking background thread.
        This method is called when the user explicitly enables 'Deep Research' mode.
        
        Args:
            session_id: The ID of the current user session.
            query: The research topic provided by the user.
            persona: The AI persona for context (e.g., 'symptom').
            progress_queue: The queue to send real-time progress updates to the UI.
        """
        # This logic is moved from the old `process_turn` method.
        def research_callback(message: str, status: str):
            """Puts progress updates into the queue for the UI to display."""
            progress_queue.put({"type": "progress", "message": message, "status": status})

        def research_thread_target():
            """The target function for the background thread."""
            # Use a dictionary to capture the return value from the thread.
            result_container = {}
            
            # NOTE: Assumes your orchestrator has these methods.
            # If your orchestrator takes the callback directly, you can simplify this.
            self.research_orchestrator.set_progress_callback(research_callback)
            
            # Run the main research workflow.
            # We assume `run_research_workflow` populates the `result_container`.
            self.research_orchestrator.run_research_workflow(query, result_container)
            
            # Once complete, put the final report into the queue.
            final_report = result_container.get("report", "Research completed, but the final report is missing.")
            progress_queue.put({"type": "complete", "report": final_report})

        # Create and start the background thread.
        research_thread = threading.Thread(target=research_thread_target)
        research_thread.start()
        
        # This function returns immediately, allowing the UI to remain responsive.

    # MODIFIED: This method is now ONLY for conversation.
    def process_turn(self, session_id: str, user_input: str, persona: str) -> Dict[str, Any]:
        """
        Processes one turn of a standard conversation using the ReAct agent framework.
        This does NOT handle deep research initiation anymore.
        """
        session = self._get_or_create_session(session_id, persona)
        
        # Handle special system messages or standard user input
        if user_input.startswith("SYSTEM_MESSAGE:"):
            # This is for providing context back to the agent, e.g., after research is done.
            session["history"].append({"role": "tool_observation", "content": f"Observation: {user_input}"})
        else:
            session["history"].append({"role": "user", "content": user_input})

        # Get the AI's ReAct response
        response_message, _ = self.model_manager.create_completion(
            messages=session["history"],
            reasoning_level="high", temperature=0.4, max_tokens=1000
        )

        if not response_message or not response_message.content:
            return {"type": "error", "content": "I apologize, but I encountered a problem and can't respond right now."}

        parsed_response = self._parse_react_response(response_message.content)
        
        # If parsing fails or returns a raw answer, use it directly.
        if not parsed_response or parsed_response.get("action") is None:
            return {"type": "final_answer", "content": parsed_response.get("content", "I'm having trouble formulating a structured response. Could you rephrase?")}
        
        # Add the full thought/action process to history for context in the next turn
        session["history"].append({"role": "assistant", "content": response_message.content})

        action = parsed_response.get("action")
        action_input = parsed_response.get("action_input", {})

        # MODIFIED: Simplified action handling
        if action == "ask_clarifying_question":
            question = action_input.get("question", "I have a follow-up question, but it seems to be malformed. Can you clarify?")
            return {"type": "conversation", "content": question} # Changed type for consistency
        
        elif action == "FinalAnswer":
            summary = action_input.get("summary", "I have a final answer, but it seems to be malformed.")
            return {"type": "conversation", "content": summary} # Changed type for consistency

        # MODIFIED: The 'initiate_deep_research' action is no longer handled here.
        # It has been moved to its own dedicated `start_deep_research` method.

        else:
            # Fallback for any other unexpected action
            unknown_action_response = f"I'm thinking about performing an action ('{action}'), but I need more clarity. Could you please rephrase your request?"
            return {"type": "conversation", "content": unknown_action_response}