import re
import json
import datetime
from typing import List, Dict, Any, Optional, Callable

from core.model_manager import ModelManager
from core.tool_definitions import ALL_TOOL_SCHEMAS
from core.tool_functions import ToolExecutor

MAX_TOOL_ITERATIONS = 5 # Maximum times the LLM can call tools for a single research question

class MedicalResearchOrchestrator:
    """Orchestrates medical research using LLM-driven tool calls and synthesis."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.tool_executor = ToolExecutor(model_manager) # Initialize ToolExecutor
        self.progress_callback: Optional[Callable[[str, str], None]] = None
        self.current_research_state: Dict[str, Any] = {}

    def set_progress_callback(self, callback_function: Callable[[str, str], None]) -> None:
        """Set a callback function to report progress (e.g., to a Streamlit UI)."""
        self.progress_callback = callback_function

    def _report_progress(self, message: str, status: str = "running") -> None:
        """Report progress using the callback if available."""
        print(f"Research Orchestrator: {message} [Status: {status}]")
        if self.progress_callback:
            try:
                self.progress_callback(message, status)
            except Exception as e:
                print(f"Error in progress callback: {e}")

    def _update_research_state(self, update_dict: Dict[str, Any]) -> None:
        """Update the current research state for UI reporting."""
        self.current_research_state.update(update_dict)
        # For Streamlit, this might involve st.session_state or a similar mechanism
        # For now, we'll rely on the progress callback to convey detailed state changes.

    def _generate_research_questions(self, initial_query: str) -> List[str]:
        """Generate specific research questions from an initial user query."""
        self._report_progress(f"LLM Call: Generating research questions for: '{initial_query}'")
        self._update_research_state({"status_message": "Generating research questions..."})

        prompt = f"""You are a medical research expert. Given the medical query: '{initial_query}', 
        generate 3-5 specific, answerable research questions that would help provide a comprehensive understanding. 
        These questions will be researched using external tools. Focus on distinct aspects like causes, treatments, 
        diagnostics, recent advancements, and patient perspectives if applicable.
        Format your response as a numbered list of questions ONLY. Example:
        1. What are the primary causes of X?
        2. How is X typically diagnosed?
        3. What are the current treatment options for X? 
        """
        
        response_message, model_used = self.model_manager.create_completion(
            messages=[
                {"role": "system", "content": "You are a medical research planning expert."},
                {"role": "user", "content": prompt}
            ],
            reasoning_level="medium",
            temperature=0.4,
            max_tokens=300
        )

        if not response_message or not response_message.content:
            self._report_progress("Failed to generate research questions from LLM.", "error")
            return [f"General information about {initial_query}"] # Fallback

        content = response_message.content
        print(f"LLM ({model_used}) generated questions: {content}")
        
        # Extract questions (more robust parsing)
        questions = []
        for line in content.split('\n'):
            match = re.match(r"^\d+\.\s*(.*)", line.strip())
            if match:
                questions.append(match.group(1).strip())
        
        if not questions:
            self._report_progress("Could not parse research questions from LLM response.", "warning")
            # Fallback if parsing fails but content exists
            questions = [q.strip() for q in content.split('\n') if q.strip() and len(q.split()) > 3] 
            if not questions: 
                 return [f"General information about {initial_query}"]

        self._report_progress(f"Generated {len(questions)} research questions.", "complete")
        self._update_research_state({"generated_questions": questions, "status_message": "Research questions generated."})
        return questions

    def _research_single_question_with_tools(self, question: str, conversation_history: List[Dict[str, Any]]) -> str:
        """Research a single question using LLM-driven tool selection and execution."""
        self._report_progress(f"Starting research for question: '{question}'") 
        self._update_research_state({
            "current_question_researching": question,
            "status_message": f"Researching: {question}",
            "tool_calls_for_current_question": []
        })

        # Append the current question to the conversation history for the tool-using LLM call
        current_turn_messages = conversation_history + [
            {"role": "user", "content": f"Please research the following medical question using the available tools: {question}"}
        ]
        
        tool_iterations = 0
        accumulated_tool_results = []

        while tool_iterations < MAX_TOOL_ITERATIONS:
            self._report_progress(f"LLM Call (tool use attempt {tool_iterations + 1}): Deciding tool for '{question}'")
            self._update_research_state({"status_message": f"LLM deciding on tool for: {question[:50]}..."})

            response_message, model_used = self.model_manager.create_completion(
                messages=current_turn_messages,
                tools=ALL_TOOL_SCHEMAS,
                tool_choice="auto", # Let the LLM decide
                reasoning_level="high", 
                temperature=0.2, # Lower temp for more deterministic tool choice
                max_tokens=1000 
            )

            if not response_message:
                self._report_progress(f"No response from LLM for tool decision on question: '{question}'", "error")
                break # Critical error, stop for this question

            current_turn_messages.append(response_message.model_dump(exclude_none=True)) # Add LLM's response to history

            if response_message.tool_calls:
                self._report_progress(f"LLM ({model_used}) decided to use tool(s). Executing...")
                self._update_research_state({"status_message": "LLM selected tool(s). Executing..."})

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args_str = tool_call.function.arguments
                    tool_call_id = tool_call.id
                    
                    self._report_progress(f"Tool Call: {function_name}, Args: {function_args_str}")                   
                    self._update_research_state({
                        "status_message": f"Executing tool: {function_name}",
                        "last_tool_called": function_name,
                        "last_tool_args": function_args_str
                    })
                    self.current_research_state.setdefault("tool_calls_for_current_question", []).append(
                        {"name": function_name, "args": function_args_str, "id": tool_call_id, "status": "pending"}
                    )

                    try:
                        args_dict = json.loads(function_args_str)
                        if function_name in self.tool_executor.dispatch_table:
                            # Assuming all current tools take a single 'query' argument
                            query_arg = args_dict.get("query", args_dict.get("search_query", args_dict.get("topic")))
                            if query_arg is None:
                                raise ValueError("Missing 'query' argument for tool.")

                            tool_result = self.tool_executor.dispatch_table[function_name](query_arg)
                            self._report_progress(f"Tool '{function_name}' executed. Result length: {len(tool_result)}")
                            self.current_research_state["tool_calls_for_current_question"][-1]["status"] = "success"
                            self.current_research_state["tool_calls_for_current_question"][-1]["result_preview"] = tool_result[:200] + "..."
                        else:
                            tool_result = f"Error: Tool '{function_name}' is not recognized."
                            self._report_progress(tool_result, "error")
                            self.current_research_state["tool_calls_for_current_question"][-1]["status"] = "error_unknown_tool"

                    except json.JSONDecodeError:
                        tool_result = f"Error: Invalid JSON arguments for {function_name}: {function_args_str}"
                        self._report_progress(tool_result, "error")
                        self.current_research_state["tool_calls_for_current_question"][-1]["status"] = "error_json_args"
                    except Exception as e:
                        tool_result = f"Error executing tool '{function_name}': {str(e)}"
                        self._report_progress(tool_result, "error")
                        self.current_research_state["tool_calls_for_current_question"][-1]["status"] = "error_execution"
                    
                    # Append tool result to conversation for next LLM turn
                    current_turn_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": function_name,
                        "content": tool_result
                    })
                    accumulated_tool_results.append(f"Tool: {function_name}\nArguments: {function_args_str}\nResult: {tool_result}\n---")
                
                tool_iterations += 1 # Count this as one iteration of tool use
            
            elif response_message.content: # LLM responded directly without tool use
                self._report_progress(f"LLM ({model_used}) provided a direct answer for '{question}' without further tool use.")
                self._update_research_state({"status_message": "LLM provided direct answer.", "final_answer_for_question": response_message.content})
                accumulated_tool_results.append(f"LLM Direct Response: {response_message.content}")
                break # End tool use loop for this question
            
            else: # No tool call and no content, might be an issue or LLM thinks it's done
                self._report_progress(f"LLM ({model_used}) did not call a tool and provided no content for '{question}'. Assuming research for this question is complete based on prior context.", "warning")
                break

        if not accumulated_tool_results:
            self._report_progress(f"No tool results or direct LLM answer obtained for question: '{question}' after {tool_iterations} iterations.", "warning")
            return f"Could not gather information for the question: '{question}'. The LLM did not successfully use any tools or provide a direct answer after {MAX_TOOL_ITERATIONS} attempts."

        # After tool loop, synthesize the findings for the current question
        self._report_progress(f"LLM Call: Synthesizing findings for question: '{question}'")
        self._update_research_state({"status_message": f"Synthesizing findings for: {question[:50]}..."})

        synthesis_prompt_messages = conversation_history + [
            {"role": "user", "content": f"Based on the preceding conversation and tool executions related to the question '{question}', please synthesize all the gathered information into a comprehensive answer. Ensure the answer directly addresses the question. If the information is insufficient, state that clearly."}
        ]
        # Add the accumulated tool results explicitly if not already in current_turn_messages effectively
        if current_turn_messages[-1]["role"] != "user": # ensure last message is not user to avoid double prompt.
            synthesis_prompt_messages.append(current_turn_messages[-1])
            
        response_message, model_used = self.model_manager.create_completion(
            messages=synthesis_prompt_messages,
            # No tools for synthesis, we want a direct textual answer
            reasoning_level="high",
            temperature=0.5,
            max_tokens=1000 
        )

        if response_message and response_message.content:
            self._report_progress(f"LLM ({model_used}) synthesized answer for '{question}'. Length: {len(response_message.content)}")
            self._update_research_state({"status_message": "Synthesis complete.", "synthesized_answer_for_question": response_message.content})
            return response_message.content
        else:
            self._report_progress(f"LLM failed to synthesize an answer for '{question}'. Returning raw accumulated data.", "error")
            return "\n\n".join(accumulated_tool_results) if accumulated_tool_results else f"Synthesis failed for '{question}' and no raw data to return."

    def _critique_findings(self, findings_text: str, original_query: str) -> str:
        """Critique and validate medical research findings using an LLM call."""
        self._report_progress(f"LLM Call: Critiquing findings related to: '{original_query}'")
        self._update_research_state({"status_message": "Critiquing research findings..."})
        
        prompt = f"""You are a medical research validation and critique agent. Review the following medical information, which was gathered in response to the query '{original_query}'. Critically evaluate it:

        {findings_text}
        
        Please provide:
        1. An assessment of medical accuracy (is it supported by current medical consensus?)
        2. Identification of any potential misinformation, bias, or outdated information.
        3. Verification of claims against known medical standards where possible.
        4. Suggestions for important information that might be missing or areas needing further investigation.
        5. An overall confidence rating in the findings (High, Medium, or Low).
        Focus on being objective, evidence-based, and constructive in your critique. Keep it concise.
        """

        response_message, model_used = self.model_manager.create_completion(
            messages=[
                {"role": "system", "content": "You are an expert medical research validator.",},
                {"role": "user", "content": prompt}
            ],
            reasoning_level="high",
            temperature=0.3,
            max_tokens=800
        )

        if response_message and response_message.content:
            self._report_progress(f"LLM ({model_used}) provided critique. Length: {len(response_message.content)}")
            self._update_research_state({"critique_complete": True, "critique_content": response_message.content})
            return response_message.content
        else:
            self._report_progress("LLM failed to provide a critique.", "error")
            self._update_research_state({"critique_complete": False, "status_message": "Critique failed."})
            return "Critique could not be generated."

    def _generate_final_report(self, initial_query: str, research_summary_per_question: Dict[str, str], overall_critique: str) -> str:
        """Generate a comprehensive final report from all research and critique."""
        self._report_progress("LLM Call: Generating final comprehensive report.")
        self._update_research_state({"status_message": "Generating final report..."})

        formatted_research = ""
        for q, res in research_summary_per_question.items():
            formatted_research += f"Question: {q}\nResearch Summary: {res[:1000]}...\n\n" # Truncate individual summaries

        prompt = f"""You are a medical AI assistant creating a final research report.
        Original User Query: {initial_query}

        Research Findings (summarized per question):
        {formatted_research.strip()}

        Overall Critique of Findings:
        {overall_critique}

        Based on all the above, create a comprehensive, well-structured medical research report that:
        1. Directly addresses the original user query: '{initial_query}'.
        2. Summarizes key findings from the research, incorporating insights from the critique.
        3. Highlights medical consensus, areas of uncertainty, or ongoing research as identified.
        4. Provides evidence-based information and, if appropriate for the query type, general recommendations (clearly stating these are not medical advice).
        5. Uses clear language, proper medical terminology where needed but remains accessible.
        6. Concludes with a brief summary and a disclaimer that this is for informational purposes and not a substitute for professional medical advice.
        
        Format with clear section headers (e.g., ## Introduction, ## Key Findings, ## Detailed Analysis, ## Conclusion, ## Disclaimer).
        Do NOT include any HTML tags. Use Markdown for formatting if necessary (headers, lists).
        """

        response_message, model_used = self.model_manager.create_completion(
            messages=[
                {"role": "system", "content": "You are an expert medical report writer."},
                {"role": "user", "content": prompt}
            ],
            reasoning_level="high",
            temperature=0.4,
            max_tokens=2000 # Allow for a longer, comprehensive report
        )

        if response_message and response_message.content:
            report_content = response_message.content
            # Simple HTML tag stripping (can be made more robust if needed)
            report_content = re.sub(r'<[^>]+>', '', report_content)
            self._report_progress(f"LLM ({model_used}) generated final report. Length: {len(report_content)}", "complete")
            self._update_research_state({"final_report_generated": True, "final_report_content": report_content})
            return report_content
        else:
            self._report_progress("LLM failed to generate the final report.", "error")
            self._update_research_state({"final_report_generated": False, "status_message": "Final report generation failed."})
            return "Final report could not be generated. Accumulated research might be incomplete or failed synthesis."

    def run_research_workflow(self, initial_query: str) -> str:
        """Main method to run the entire deep medical research workflow."""
        self.current_research_state = { # Reset state for a new run
            "initial_query": initial_query,
            "status_message": "Starting research workflow...",
            "generated_questions": [],
            "research_results_per_question": {},
            "critique_complete": False,
            "final_report_generated": False
        }
        self._report_progress(f"Starting full research workflow for: '{initial_query}'")

        # Step 1: Generate research questions
        research_questions = self._generate_research_questions(initial_query)
        if not research_questions:
            final_error_report = "Failed to initiate research: Could not generate research questions."
            self._report_progress(final_error_report, "error")
            return final_error_report
        self.current_research_state["generated_questions"] = research_questions
        
        # Step 2: Research each question using tools
        # Initial system message for the conversation involving tool use
        base_conversation_history = [
            {"role": "system", "content": "You are a helpful AI medical research assistant. Your goal is to answer medical questions thoroughly by using the provided tools. When a user asks a question, decide if a tool is needed. If so, call the appropriate tool with the correct arguments. After receiving the tool's output, analyze it and decide if more tool calls are needed or if you can answer the question. If multiple tools seem relevant, you can use them sequentially. Once all necessary information for a specific question is gathered, you will be asked to synthesize it."}
        ]

        all_question_research_syntheses: Dict[str, str] = {}
        for i, q_text in enumerate(research_questions):
            self._report_progress(f"Processing question {i+1}/{len(research_questions)}: {q_text}")
            self._update_research_state({"current_question_processing_index": i, "total_questions_to_process": len(research_questions)})
            
            # Pass a copy of the base history so each question starts fresh but with system context
            question_specific_synthesis = self._research_single_question_with_tools(q_text, list(base_conversation_history))
            all_question_research_syntheses[q_text] = question_specific_synthesis
            self.current_research_state["research_results_per_question"][q_text] = question_specific_synthesis
            self._report_progress(f"Completed research and initial synthesis for question: '{q_text}'", "complete")

        # Step 3: Collate all syntheses and critique them
        collated_syntheses = "\n\n---\n\n".join(
            f"Question: {q}\nSynthesized Answer: {ans}"
            for q, ans in all_question_research_syntheses.items()
        )
        if not collated_syntheses.strip():
            final_error_report = "Research failed: No information was gathered or synthesized for any question."
            self._report_progress(final_error_report, "error")
            return final_error_report
            
        overall_critique = self._critique_findings(collated_syntheses, initial_query)
        self.current_research_state["overall_critique"] = overall_critique

        # Step 4: Generate final report
        final_report = self._generate_final_report(initial_query, all_question_research_syntheses, overall_critique)
        self._report_progress("Research workflow completed.", "complete")
        self.current_research_state["status_message"] = "Research workflow complete."
        return final_report 