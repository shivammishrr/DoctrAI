import json
import threading
import logging
from queue import Queue
from typing import Dict, Any

from core.model_manager import ModelManager
from core.research_orchestrator import MedicalResearchOrchestrator
from core.prompts import get_persona_prompt
from core.tool_definitions import ALL_TOOL_SCHEMAS

logger = logging.getLogger(__name__)

MAX_REACT_ITERATIONS = 5


class DoctorAI:
    def __init__(self):
        self.model_manager = ModelManager()
        self.research_orchestrator = MedicalResearchOrchestrator(self.model_manager)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._session_lock = threading.Lock()

    def _get_or_create_session(self, session_id: str, persona: str) -> Dict[str, Any]:
        with self._session_lock:
            if session_id not in self.sessions:
                system_prompt = get_persona_prompt(persona)
                self.sessions[session_id] = {
                    "history": [{"role": "system", "content": system_prompt}]
                }
            return self.sessions[session_id]

    def add_system_observation(self, session_id: str, observation: str) -> None:
        with self._session_lock:
            session = self.sessions.get(session_id)
            if session:
                session["history"].append({
                    "role": "system",
                    "content": f"[System]: {observation}"
                })

    def start_deep_research(self, session_id: str, query: str, persona: str, progress_queue: Queue):
        def research_callback(message: str, status: str):
            progress_queue.put({"type": "progress", "message": message, "status": status})

        def research_thread_target():
            result_container = {}
            self.research_orchestrator.set_progress_callback(research_callback)
            self.research_orchestrator.run_research_workflow(query, result_container)
            final_report = result_container.get("report", "Research completed, but the final report is missing.")
            progress_queue.put({"type": "complete", "report": final_report})

        research_thread = threading.Thread(target=research_thread_target, daemon=True)
        research_thread.start()

    def process_turn(self, session_id: str, user_input: str, persona: str) -> Dict[str, Any]:
        session = self._get_or_create_session(session_id, persona)

        with self._session_lock:
            session["history"].append({"role": "user", "content": user_input})

        for iteration in range(MAX_REACT_ITERATIONS):
            response_message, _ = self.model_manager.create_completion(
                messages=self._get_history(session_id),
                tools=ALL_TOOL_SCHEMAS,
                tool_choice="auto",
                reasoning_level="high",
                temperature=0.4,
                max_tokens=1000,
            )

            if not response_message:
                return {"type": "error", "content": "I apologize, but I encountered a problem and can't respond right now."}

            with self._session_lock:
                session["history"].append(response_message.model_dump(exclude_none=True))

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    if fn_name == "ask_clarifying_question":
                        question = fn_args.get("question", "I have a follow-up question. Can you clarify?")
                        return {"type": "conversation", "content": question, "finish_reason": "ask_clarifying_question"}

                    if fn_name == "FinalAnswer":
                        summary = fn_args.get("summary", "I have completed my analysis.")
                        return {"type": "conversation", "content": summary, "finish_reason": "final_answer"}

                    result = self._execute_research_tool(tool_call)
                    with self._session_lock:
                        session["history"].append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result)[:8000],
                        })
            elif response_message.content:
                return {"type": "conversation", "content": response_message.content, "finish_reason": "text"}

        return {
            "type": "error",
            "content": "I couldn't complete my analysis within the allowed steps. Please rephrase or try again."
        }

    def _execute_research_tool(self, tool_call) -> str:
        fn_name = tool_call.function.name
        try:
            fn_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return f"Error: Invalid arguments for {fn_name}"

        dispatch = self.research_orchestrator.tool_executor.dispatch_table
        if fn_name in dispatch:
            query = next(
                (v for v in fn_args.values() if isinstance(v, str)),
                None
            )
            if not query:
                return f"Error: No valid query argument found for {fn_name}"
            try:
                result = dispatch[fn_name](query)
                return result
            except Exception as e:
                logger.error(f"Tool {fn_name} execution error: {e}")
                return f"Error executing {fn_name}: {str(e)}"

        return f"Error: Unknown tool '{fn_name}'."

    def _get_history(self, session_id: str) -> list:
        with self._session_lock:
            session = self.sessions.get(session_id)
            if not session:
                return []
            return [m.copy() for m in session["history"]]
