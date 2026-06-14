import re
import json
import threading
import traceback
import logging
from typing import List, Dict, Any, Optional, Callable

from core.model_manager import ModelManager
from core.tool_definitions import RESEARCH_TOOL_SCHEMAS
from core.tool_functions import ToolExecutor
from core.prompts import (
    GENERATE_RESEARCH_QUESTIONS_PROMPT,
    SYSTEM_MESSAGE_RESEARCH_PLANNER,
    SYNTHESIZE_RESEARCH_FINDINGS_PROMPT,
    CRITIQUE_RESEARCH_FINDINGS_PROMPT,
    SYSTEM_MESSAGE_RESEARCH_VALIDATOR,
    GENERATE_FINAL_REPORT_PROMPT,
    SYSTEM_MESSAGE_REPORT_WRITER,
    SYSTEM_MESSAGE_TOOL_USER,
)

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 5


class MedicalResearchOrchestrator:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.tool_executor = ToolExecutor(model_manager)
        self.progress_callback: Optional[Callable[[str, str], None]] = None
        self.current_research_state: Dict[str, Any] = {}
        self._state_lock = threading.Lock()

    def set_progress_callback(self, callback_function: Callable[[str, str], None]) -> None:
        self.progress_callback = callback_function

    def _report_progress(self, message: str, status: str = "running") -> None:
        logger.info(f"Research: {message} [Status: {status}]")
        if self.progress_callback:
            try:
                self.progress_callback(message, status)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def _update_research_state(self, update_dict: Dict[str, Any]) -> None:
        with self._state_lock:
            self.current_research_state.update(update_dict)

    def get_research_state_copy(self) -> Dict[str, Any]:
        with self._state_lock:
            return dict(self.current_research_state)

    def _generate_research_questions(self, initial_query: str) -> List[str]:
        self._report_progress(f"Generating research questions for: '{initial_query}'")
        self._update_research_state({"status_message": "Generating research questions..."})

        prompt = GENERATE_RESEARCH_QUESTIONS_PROMPT.format(initial_query=initial_query)
        response_message, model_used = self.model_manager.create_completion(
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE_RESEARCH_PLANNER},
                {"role": "user", "content": prompt}
            ],
            reasoning_level="medium",
            temperature=0.4,
            max_tokens=300,
        )

        if not response_message or not response_message.content:
            raise ValueError("Failed to generate research questions from LLM.")

        content = response_message.content
        questions = [
            match.group(1).strip()
            for line in content.split('\n')
            if (match := re.match(r"^\d+\.\s*(.*)", line.strip()))
        ]

        if not questions:
            self._report_progress("Could not parse research questions from LLM response.", "warning")
            return [initial_query]

        self._report_progress(f"Generated {len(questions)} research questions.", "complete")
        self._update_research_state({"generated_questions": questions, "status_message": "Research questions generated."})
        return questions

    def _research_single_question_with_tools(self, question: str) -> str:
        self._report_progress(f"Starting research for question: '{question}'")
        self._update_research_state({
            "current_question_researching": question,
            "status_message": f"Researching: {question}",
            "tool_calls_for_current_question": [],
        })

        current_turn_messages = [
            {"role": "system", "content": SYSTEM_MESSAGE_TOOL_USER},
            {"role": "user", "content": f"Please research the following medical question using the available tools: {question}"}
        ]

        tool_iterations = 0
        accumulated_tool_results = []

        while tool_iterations < MAX_TOOL_ITERATIONS:
            self._report_progress(f"LLM Call (tool use attempt {tool_iterations + 1})")
            response_message, model_used = self.model_manager.create_completion(
                messages=current_turn_messages,
                tools=RESEARCH_TOOL_SCHEMAS,
                tool_choice="auto",
                reasoning_level="high",
                temperature=0.2,
                max_tokens=1000,
            )

            if not response_message:
                self._report_progress(f"No response from LLM for question: '{question}'", "error")
                break

            current_turn_messages.append(response_message.model_dump(exclude_none=True))

            if response_message.tool_calls:
                self._update_research_state({"status_message": "Executing tool(s)..."})

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args_str = tool_call.function.arguments
                    tool_call_id = tool_call.id

                    self._report_progress(f"Tool Call: {function_name}")
                    tool_entry = {"name": function_name, "args": function_args_str, "id": tool_call_id, "status": "pending"}

                    with self._state_lock:
                        self.current_research_state.setdefault("tool_calls_for_current_question", []).append(tool_entry)

                    try:
                        args_dict = json.loads(function_args_str)
                        if function_name in self.tool_executor.dispatch_table:
                            query_arg = next(
                                (args_dict.get(key) for key in ["query", "search_query", "topic"] if key in args_dict),
                                None
                            )
                            if query_arg is None:
                                raise ValueError("Missing required 'query' argument.")
                            tool_result = self.tool_executor.dispatch_table[function_name](query_arg)
                            with self._state_lock:
                                if self.current_research_state["tool_calls_for_current_question"]:
                                    self.current_research_state["tool_calls_for_current_question"][-1]["status"] = "success"
                        else:
                            raise ValueError(f"Tool '{function_name}' is not recognized.")
                    except Exception as e:
                        tool_result = f"Error executing tool '{function_name}': {str(e)}"
                        with self._state_lock:
                            if self.current_research_state["tool_calls_for_current_question"]:
                                self.current_research_state["tool_calls_for_current_question"][-1]["status"] = "error_execution"

                    current_turn_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": function_name,
                        "content": tool_result,
                    })
                    accumulated_tool_results.append(tool_result)

                tool_iterations += 1
            elif response_message.content:
                accumulated_tool_results.append(f"LLM Direct Response: {response_message.content}")
                break
            else:
                break

        if not accumulated_tool_results:
            return f"Could not gather information for the question: '{question}'."

        synthesis_prompt = SYNTHESIZE_RESEARCH_FINDINGS_PROMPT.format(question=question)
        synthesis_messages = current_turn_messages + [{"role": "user", "content": synthesis_prompt}]

        response_message, model_used = self.model_manager.create_completion(
            messages=synthesis_messages,
            reasoning_level="high",
            temperature=0.5,
            max_tokens=1500,
        )
        if response_message and response_message.content:
            return response_message.content
        return "\n\n".join(accumulated_tool_results)

    def _critique_findings(self, findings_text: str, original_query: str) -> str:
        self._report_progress(f"Critiquing findings for: '{original_query}'")
        self._update_research_state({"status_message": "Critiquing research findings..."})

        prompt = CRITIQUE_RESEARCH_FINDINGS_PROMPT.format(
            original_query=original_query, findings_text=findings_text
        )
        response_message, _ = self.model_manager.create_completion(
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE_RESEARCH_VALIDATOR},
                {"role": "user", "content": prompt},
            ],
            reasoning_level="high",
            temperature=0.3,
            max_tokens=800,
        )

        if response_message and response_message.content:
            self._update_research_state({
                "critique_complete": True,
                "critique_content": response_message.content,
            })
            return response_message.content
        return "Critique could not be generated."

    def _generate_final_report(
        self,
        initial_query: str,
        research_summary_per_question: Dict[str, str],
        overall_critique: str,
    ) -> str:
        self._report_progress("Generating final comprehensive report.")
        self._update_research_state({"status_message": "Generating final report..."})

        formatted_research = "\n\n".join(
            f"Question: {q}\nResearch Summary:\n{res}"
            for q, res in research_summary_per_question.items()
        )
        prompt = GENERATE_FINAL_REPORT_PROMPT.format(
            initial_query=initial_query,
            formatted_research=formatted_research,
            overall_critique=overall_critique,
        )
        response_message, _ = self.model_manager.create_completion(
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE_REPORT_WRITER},
                {"role": "user", "content": prompt},
            ],
            reasoning_level="high",
            temperature=0.4,
            max_tokens=3000,
        )

        if response_message and response_message.content:
            self._update_research_state({"final_report_generated": True})
            return response_message.content
        raise ValueError("LLM failed to generate the final report.")

    def run_research_workflow(self, initial_query: str, result_container: Dict[str, Any]) -> None:
        try:
            self._update_research_state({"initial_query": initial_query})
            self._report_progress(f"Starting full research workflow for: '{initial_query}'")

            research_questions = self._generate_research_questions(initial_query)

            all_question_research_syntheses: Dict[str, str] = {}
            for i, q_text in enumerate(research_questions):
                self._report_progress(f"Processing question {i+1}/{len(research_questions)}")
                self._update_research_state({
                    "current_question_processing_index": i,
                    "total_questions_to_process": len(research_questions),
                })
                all_question_research_syntheses[q_text] = self._research_single_question_with_tools(q_text)

            collated_syntheses = "\n\n---\n\n".join(
                f"Question: {q}\nSynthesized Answer: {ans}"
                for q, ans in all_question_research_syntheses.items()
            )
            if not collated_syntheses.strip():
                raise ValueError("Research failed: No information was gathered.")

            overall_critique = self._critique_findings(collated_syntheses, initial_query)
            final_report = self._generate_final_report(
                initial_query, all_question_research_syntheses, overall_critique
            )

            result_container['status'] = 'complete'
            result_container['report'] = final_report
            self._report_progress("Research workflow completed successfully.", "complete")

        except Exception as e:
            error_message = f"An error occurred during the research workflow: {str(e)}"
            tb_str = traceback.format_exc()
            logger.error(f"Research workflow failed:\n{tb_str}")
            self._report_progress(error_message, "error")
            result_container['status'] = 'error'
            result_container['report'] = (
                f"I apologize, but I encountered an unexpected error during the deep research process: {str(e)}"
            )
