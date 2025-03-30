import os
import re
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from groq import Groq
import streamlit as st
from langchain.tools import ArxivQueryRun, WikipediaQueryRun, Tool
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from dotenv import load_dotenv
try:
    from langchain.tools.tavily_search import TavilySearchResults
except ImportError:
    # Fallback for older langchain versions
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
    except ImportError:
        TavilySearchResults = None

load_dotenv()

class ModelManager:
    """Manages model selection and fallback for handling token limits."""
    
    def __init__(self):
        self.api_key = os.getenv("grok_api_key")
        self.client = Groq(api_key=self.api_key)
        self.model_configs = self._initialize_model_configs()
        self.current_model = "llama-3.3-70b-specdec"  # Default model
        self.fallback_attempts = 0
        self.max_fallback_attempts = 3
        self.input_truncation_factor = 0.7  # Reduce input by this factor when too large
    
    def _initialize_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize model configurations with their limits and capabilities."""
        return {
            # High-reasoning models (for complex medical analysis)
            "llama-3.3-70b-specdec": {
                "requests_per_day": 1000,
                "tokens_per_day": 100000,
                "tokens_per_minute": 6000,
                "reasoning_level": "high",
                "priority": 1
            },
            "llama-3.3-70b-versatile": {
                "requests_per_day": 1000,
                "tokens_per_day": 100000,
                "tokens_per_minute": 6000,
                "reasoning_level": "high",
                "priority": 2
            },
            "qwen-2.5-32b": {
                "requests_per_day": 1000,
                "tokens_per_day": float('inf'),  # No limit
                "tokens_per_minute": 6000,
                "reasoning_level": "high",
                "priority": 3
            },
            "mistral-saba-24b": {
                "requests_per_day": 1000,
                "tokens_per_day": 500000,
                "tokens_per_minute": 6000,
                "reasoning_level": "high",
                "priority": 4
            },
            
            # Medium-reasoning models (for standard medical information)
            "llama-3.2-11b-vision-preview": {
                "requests_per_day": 7000,
                "tokens_per_day": 500000,
                "tokens_per_minute": 7000,
                "reasoning_level": "medium",
                "priority": 5
            },
            "gemma2-9b-it": {
                "requests_per_day": 14400,
                "tokens_per_day": 500000,
                "tokens_per_minute": 15000,
                "reasoning_level": "medium",
                "priority": 6
            },
            
            # Basic models (for simple tasks)
            "llama-3.1-8b-instant": {
                "requests_per_day": 14400,
                "tokens_per_day": 500000,
                "tokens_per_minute": 6000,
                "reasoning_level": "basic",
                "priority": 7
            },
            "llama-2-7b": {
                "requests_per_day": 7000,
                "tokens_per_day": float('inf'),  # No limit
                "tokens_per_minute": 6000,
                "reasoning_level": "basic",
                "priority": 8
            }
        }
    
    def _truncate_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Truncate message content to reduce token count."""
        truncated_messages = []
        
        # Keep system messages intact, but truncate user and assistant messages
        for msg in messages:
            if msg["role"] == "system":
                # Keep system messages as they are
                truncated_messages.append(msg)
            else:
                # Truncate content of user and assistant messages
                content = msg["content"]
                # Calculate truncation based on message length
                # Longer messages get truncated more aggressively
                truncation_length = int(len(content) * self.input_truncation_factor)
                if len(content) > 1000:  # Only truncate longer messages
                    truncated_content = content[:truncation_length] + "\n\n[Content truncated due to token limits]"
                    truncated_messages.append({"role": msg["role"], "content": truncated_content})
                else:
                    truncated_messages.append(msg)
        
        return truncated_messages
    
    def _estimate_token_count(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token count in messages (rough approximation)."""
        # A very rough estimate: 1 token ≈ 4 characters for English text
        total_chars = sum(len(msg["content"]) for msg in messages)
        return total_chars // 4
    
    def get_next_fallback_model(self, reasoning_level: str = "high") -> str:
        """Get the next available model based on reasoning level required and priority."""
        available_models = [
            model for model, config in self.model_configs.items()
            if config["reasoning_level"] == reasoning_level or 
               (reasoning_level == "high" and config["reasoning_level"] == "medium") or
               (reasoning_level == "medium" and config["reasoning_level"] == "basic")
        ]
        
        # Sort by priority (lower number = higher priority)
        available_models.sort(key=lambda m: self.model_configs[m]["priority"])
        
        # Skip the current model and return the next one
        for model in available_models:
            if model != self.current_model:
                return model
        
        # If no other model is available, return the current one
        return self.current_model
    
    def create_completion(self, messages: List[Dict[str, str]], reasoning_level: str = "high", 
                         temperature: float = 0.5, max_tokens: int = 1000) -> Tuple[str, str]:
        """Create a completion with automatic fallback to other models if rate limited."""
        self.fallback_attempts = 0
        original_messages = messages.copy()
        current_messages = messages
        input_truncated = False
        
        while self.fallback_attempts < self.max_fallback_attempts:
            try:
                # Estimate token count of input
                estimated_tokens = self._estimate_token_count(current_messages)
                model_token_limit = self.model_configs[self.current_model]["tokens_per_minute"]
                
                # If estimated tokens exceed the model's per-minute limit, truncate input
                if estimated_tokens > model_token_limit * 0.9:  # 90% of limit as safety margin
                    print(f"Input too large for {self.current_model} (~{estimated_tokens} tokens, limit {model_token_limit})")
                    if not input_truncated:
                        # First try truncating the input
                        current_messages = self._truncate_messages(current_messages)
                        input_truncated = True
                        print(f"Truncated input to approximately {self._estimate_token_count(current_messages)} tokens")
                        continue
                
                response = self.client.chat.completions.create(
                    messages=current_messages,
                    model=self.current_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content, self.current_model
            
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for different types of rate limit errors
                if "too large" in error_str or "tpm" in error_str:
                    # Input size/tokens per minute limit
                    print(f"Input size error with model {self.current_model}: {str(e)}")
                    
                    if not input_truncated:
                        # First try truncating the input
                        current_messages = self._truncate_messages(current_messages)
                        input_truncated = True
                        print(f"Truncated input to approximately {self._estimate_token_count(current_messages)} tokens")
                        continue
                    else:
                        # If already truncated, try a different model
                        next_model = self.get_next_fallback_model(reasoning_level)
                        if next_model == self.current_model:
                            raise Exception(f"All models exhausted. Last error: {str(e)}")
                        
                        print(f"Falling back to model: {next_model}")
                        self.current_model = next_model
                        # Reset messages but keep truncation
                        current_messages = self._truncate_messages(original_messages) if input_truncated else original_messages
                        self.fallback_attempts += 1
                
                elif "rate limit" in error_str or "tpd" in error_str:
                    # Daily token limit reached
                    print(f"Rate limit error with model {self.current_model}: {str(e)}")
                    
                    # Try a different model
                    next_model = self.get_next_fallback_model(reasoning_level)
                    if next_model == self.current_model:
                        raise Exception(f"All models exhausted. Last error: {str(e)}")
                    
                    print(f"Falling back to model: {next_model}")
                    self.current_model = next_model
                    self.fallback_attempts += 1
                
                else:
                    # Other API error
                    print(f"API error with model {self.current_model}: {str(e)}")
                    
                    # Try a different model for other errors too
                    next_model = self.get_next_fallback_model(reasoning_level)
                    if next_model == self.current_model:
                        raise Exception(f"All models exhausted. Last error: {str(e)}")
                    
                    print(f"Falling back to model: {next_model}")
                    self.current_model = next_model
                    self.fallback_attempts += 1
        
        raise Exception(f"Maximum fallback attempts ({self.max_fallback_attempts}) reached.")

class DoctorAI:
    def __init__(self):
        self.model_manager = ModelManager()
        self.context = """You are an advanced AI medical assistant with comprehensive knowledge across medical specialties. 
        Provide direct, clear, and actionable medical information and advice based on established medical science and 
        clinical guidelines. Focus on delivering practical insights and evidence-based recommendations."""
        self.research_system = MedicalResearchOrchestrator()
    
    def get_medical_advice(self, symptoms: str, deep_research: bool = False) -> str:
        """Get medical advice based on described symptoms"""
        if not deep_research:
            prompt = f"""Based on the following symptoms, provide general medical advice and recommendations:
            Symptoms: {symptoms}
            
            Please include:
            1. Possible causes
            2. General recommendations
            3. When to seek immediate medical attention"""

            content, model_used = self.model_manager.create_completion(
                messages=[
                    {"role": "system", "content": self.context},
                    {"role": "user", "content": prompt}
                ],
                reasoning_level="high",
                temperature=0.5,
                max_tokens=1000
            )
            
            print(f"Used model: {model_used} for medical advice")
            return content
        else:
            return self.research_system.run_research(f"Medical advice for symptoms: {symptoms}")
    
    def get_medication_info(self, medication: str, deep_research: bool = False) -> str:
        """Get information about a specific medication"""
        if not deep_research:
            prompt = f"""Provide comprehensive information about the following medication:
            Medication: {medication}
            
            Please include:
            1. Classification and mechanism of action
            2. Common uses and indications
            3. Typical dosing
            4. Common side effects
            5. Important warnings and contraindications
            6. Drug interactions to be aware of"""

            content, model_used = self.model_manager.create_completion(
                messages=[
                    {"role": "system", "content": self.context},
                    {"role": "user", "content": prompt}
                ],
                reasoning_level="medium",  # Medication info requires medium reasoning
                temperature=0.5,
                max_tokens=1000
            )
            
            print(f"Used model: {model_used} for medication info")
            return content
        else:
            return self.research_system.run_research(f"Medication information for: {medication}")
    
    def get_lifestyle_advice(self, condition: str, deep_research: bool = False) -> str:
        """Get lifestyle recommendations for managing a specific health condition"""
        if not deep_research:
            prompt = f"""Provide lifestyle recommendations for managing the following health condition:
            Condition: {condition}
            
            Please include:
            1. Dietary recommendations
            2. Exercise guidelines
            3. Stress management techniques
            4. Sleep recommendations
            5. Habits to avoid
            6. When to consult healthcare professionals"""

            content, model_used = self.model_manager.create_completion(
                messages=[
                    {"role": "system", "content": self.context},
                    {"role": "user", "content": prompt}
                ],
                reasoning_level="medium",  # Lifestyle advice requires medium reasoning
                temperature=0.5,
                max_tokens=1000
            )
            
            print(f"Used model: {model_used} for lifestyle advice")
            return content
        else:
            return self.research_system.run_research(f"Lifestyle recommendations for: {condition}")
    
class MedicalResearchOrchestrator:
    """Orchestrates multiple research agents to perform deep medical research."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.tools = self._create_tools()
        self.progress_callback = None
        self.current_question_index = 0
        self.current_tool_index = 0
        self.questions = []
        self.current_question = ""
        self.current_tool = ""
        self.current_findings = ""
    
    def set_progress_callback(self, callback_function):
        """Set a callback function to report progress."""
        self.progress_callback = callback_function
    
    def _report_progress(self, message, status="running"):
        """Report progress to the UI."""
        if self.progress_callback:
            self.progress_callback(message, status)
        print(f"Research Progress: {message} [{status}]")
    
    def _create_tools(self):
        """Create research tools using LangChain's built-in tools."""
        tools = []
        
        # Add Tavily search tool if API key is available
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key and TavilySearchResults:
            tools.append(TavilySearchResults(max_results=5))
        
        # Add arXiv tool
        tools.append(ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=3)))
        
        # Add Wikipedia tool
        tools.append(WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))
        
        # Add Google Search Tool - using a basic wrapper
        tools.append(
            Tool.from_function(
                func=self._google_search,
                name="GoogleSearch",
                description="Search Google for medical information. Input should be a search query."
            )
        )
        
        # Add Medical Critique Agent
        tools.append(
            Tool.from_function(
                func=self._critique_findings,
                name="MedicalCritiqueAgent",
                description="Review and validate medical research findings. Input should be medical information to validate."
            )
        )
        
        return tools
    
    def _google_search(self, query: str) -> str:
        """Perform a Google search and return results."""
        try:
            search_query = f"medical {query}"
            prompt = f"""You are a medical search tool that returns structured information about "{search_query}".
            
            Please provide:
            1. A summary of key medical facts about this topic
            2. Common medical perspectives and consensus
            3. Any recent research developments (within last 2 years if applicable)
            4. Links to reputable medical resources (Mayo Clinic, NIH, CDC, WebMD, etc.)
            
            Format your response as if you're presenting search results with clear sections.
            """
            
            content, model_used = self.model_manager.create_completion(
                messages=[
                    {"role": "system", "content": "You are a medical search engine that returns structured, factual results."},
                    {"role": "user", "content": prompt}
                ],
                reasoning_level="basic",  # Search can use basic reasoning
                temperature=0.3,
                max_tokens=800
            )
            
            print(f"Used model: {model_used} for Google search")
            return content
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    def _critique_findings(self, findings: str) -> str:
        """Critique and validate medical research findings."""
        try:
            prompt = f"""You are a medical research validator and critique agent. Review the following medical information and critically evaluate it:

            {findings}
            
            Please provide:
            1. An assessment of the medical accuracy (supported by current medical consensus)
            2. Identification of any potential misinformation or outdated information
            3. Verification of claims against known medical standards
            4. Suggestions for additional important information that might be missing
            5. Overall validity rating (High, Medium, or Low confidence)
            
            Focus on being objective and evidence-based in your critique.
            """
            
            content, model_used = self.model_manager.create_completion(
                messages=[
                    {"role": "system", "content": "You are a medical research validator with expertise in evaluating medical information."},
                    {"role": "user", "content": prompt}
                ],
                reasoning_level="high",  # Critique requires high reasoning
                temperature=0.3,
                max_tokens=800
            )
            
            print(f"Used model: {model_used} for critique findings")
            return content
        except Exception as e:
            return f"Error critiquing findings: {str(e)}"
    
    def _generate_research_questions(self, query: str) -> List[str]:
        """Generate research questions based on the user query."""
        self._report_progress("LLM call: Generating targeted research questions")
        prompt = f"""You are a medical research expert. Given the following medical query, generate 3-5 specific research questions that would help provide a comprehensive answer.
        
        Query: {query}
        
        Generate questions that cover different aspects of the query, including potential causes, treatments, recent research, and medical consensus.
        Format your response as a numbered list of questions only.
        """
        
        try:
            content, model_used = self.model_manager.create_completion(
                messages=[
                    {"role": "system", "content": "You are a medical research expert."},
                    {"role": "user", "content": prompt}
                ],
                reasoning_level="medium",  # Question generation requires medium reasoning
                temperature=0.5,
                max_tokens=1000
            )
            
            print(f"Used model: {model_used} for generating research questions")
            self._report_progress("LLM response received with research questions")
            
            # Extract questions using regex
            questions_text = content
            self._report_progress("Generated {len(questions_text.split('\n'))} research questions:\n{questions_text}", "complete")
            
            # Extract questions using regex
            questions = re.findall(r"\d+\.\s+(.*?)(?=\d+\.|$)", questions_text, re.DOTALL)
            if not questions:
                questions = [q.strip() for q in questions_text.split("\n") if q.strip() and not q.strip().startswith("Questions:")]
            
            return [q.strip() for q in questions]
        except Exception as e:
            self._report_progress(f"Error generating research questions: {str(e)}", "error")
            return [f"General information about {query}"]
    
    def _research_question(self, question: str) -> str:
        """Research a specific question using the available tools."""
        # Create a simpler approach that's compatible with Groq
        research_results = []
        
        # Use each tool sequentially to gather information
        for i, tool in enumerate(self.tools):
            self.current_tool_index = i
            tool_name = getattr(tool, "name", tool.__class__.__name__)
            self.current_tool = tool_name
            
            # Skip the critique agent in the initial research phase
            if tool_name == "MedicalCritiqueAgent":
                continue
                
            try:
                self._report_progress(f"Using {tool_name} to research: {question}")
                tool_result = tool.invoke(question)
                if tool_result and len(tool_result) > 0:
                    result_preview = tool_result[:1000] + "..." if len(tool_result) > 100 else tool_result
                    self.current_findings = result_preview
                    self._report_progress(f"Results from {tool_name} ({len(tool_result)} chars): {result_preview}")
                    research_results.append(f"Results from {tool_name}:\n{tool_result}")
                else:
                    self._report_progress(f"No results from {tool_name}")
            except Exception as e:
                error_message = f"Error using {tool_name}: {str(e)}"
                self._report_progress(error_message, "error")
                print(error_message)
        
        # Combine all research results
        combined_research = "\n\n".join(research_results)
        result_summary = f"Collected {len(research_results)} sources of information"
        self._report_progress(result_summary)
        
        # Use Groq client directly to synthesize the findings
        self._report_progress("LLM call: Synthesizing research findings")
        synthesis_prompt = f"""You are a medical research expert. Synthesize the following research findings to answer this medical question:
        
        Question: {question}
        
        Research Findings:
        {combined_research}
        
        Provide a comprehensive, well-structured answer based on these findings. Focus on medical facts, recent research, and expert consensus.
        If the research findings don't provide enough information, acknowledge the limitations and suggest what additional information would be helpful.
        """
        
        try:
            content, model_used = self.model_manager.create_completion(
                messages=[
                    {"role": "system", "content": "You are a medical research expert."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                reasoning_level="high",  # Synthesis requires high reasoning
                temperature=0.5,
                max_tokens=1000
            )
            
            print(f"Used model: {model_used} for synthesizing research findings")
            self._report_progress("LLM response received with synthesized information")
            return content
        except Exception as e:
            error_message = f"Error synthesizing findings: {str(e)}"
            self._report_progress(error_message, "error")
            return f"Unable to synthesize findings due to an error: {str(e)}"
    
    def _generate_critiqued_report(self, query: str, questions: List[str], validated_results: Dict[str, Dict[str, str]]) -> str:
        """Generate a comprehensive report based on all research results including critiques."""
        # Compile all research results with critiques
        self._report_progress("LLM call: Generating final comprehensive report with validation")
        
        # Prepare a summary of findings for each question to reduce token count
        summarized_findings = {}
        for question, tools_results in validated_results.items():
            summarized_findings[question] = {}
            for tool, result in tools_results.items():
                # Limit each tool result to 500 characters max for the final report
                if len(result) > 500:
                    summarized_findings[question][tool] = result[:500] + "... [truncated for token limit]"
                else:
                    summarized_findings[question][tool] = result
        
        # Format the findings in a readable way, limiting total size
        formatted_findings = ""
        for question, tools_results in summarized_findings.items():
            formatted_findings += f"\nQuestion: {question}\n"
            for tool, result in tools_results.items():
                formatted_findings += f"- {tool} findings: {result[:300]}...\n"
        
        # Ensure the total prompt stays under 5500 tokens (approx. 22,000 chars)
        max_chars = 22000 - len(query) - 1000  # Reserve 1000 chars for the prompt template
        if len(formatted_findings) > max_chars:
            formatted_findings = formatted_findings[:max_chars] + "\n... [additional findings truncated for token limit]"
        
        prompt = f"""You are a medical research assistant tasked with creating a comprehensive medical report.

USER QUERY: {query}

RESEARCH FINDINGS:
{formatted_findings}

Based on the above research findings, create a comprehensive medical report that:
1. Summarizes the key findings related to the query
2. Presents medical consensus where available
3. Notes areas of ongoing research or uncertainty
4. Provides evidence-based recommendations when appropriate
5. Uses proper medical terminology while remaining accessible

Format the report with clear section headers, bullet points for key information, and a conclusion.
Limit your response to a concise, focused report addressing only the most relevant information.
"""
        
        try:
            content, model_used = self.model_manager.create_completion(
                messages=[{"role": "system", "content": "You are a medical research assistant that creates comprehensive reports based on research findings."},
                         {"role": "user", "content": prompt}],
                reasoning_level="high",
                temperature=0.3,
                max_tokens=1500
            )
            
            print(f"Used model: {model_used} for generating final report")
            self._report_progress("Final report generated successfully", "complete")
            
            # Format the final report with proper headers and structure
            final_report = f"""# MEDICAL RESEARCH REPORT
## Query
{query}

## Research Findings
{content}

## Research Methodology
This report was generated using AI-assisted medical research tools including medical databases, 
scientific literature, and clinical guidelines. The information provided is for educational purposes 
and should not replace professional medical advice.

Report generated on: {datetime.datetime.now().strftime("%Y-%m-%d")}
"""
            return final_report
            
        except Exception as e:
            self._report_progress(f"Error generating final report: {str(e)}", "error")
            # Return a basic report with the raw findings in case of failure
            return f"""# MEDICAL RESEARCH REPORT (ERROR GENERATING FULL REPORT)
## Query
{query}

## Raw Research Findings
{formatted_findings}

Note: An error occurred while generating the comprehensive report. The raw research findings are provided above.
"""
    
    def run_research(self, query: str) -> str:
        """Run the research workflow for a given query."""
        # Reset state
        self.current_question_index = 0
        self.current_tool_index = 0
        self.questions = []
        self.current_question = ""
        self.current_tool = ""
        self.current_findings = ""
        
        # Step 1: Generate research questions
        self._report_progress("Starting deep medical research process")
        self._report_progress("Generating focused research questions from your query")
        self.questions = self._generate_research_questions(query)
        question_list = "\n".join([f"- {q}" for q in self.questions])
        self._report_progress(f"Generated {len(self.questions)} research questions:\n{question_list}", "complete")
        
        # Step 2: Research each question sequentially
        research_results = {}
        for i, question in enumerate(self.questions):
            self.current_question_index = i
            self.current_question = question
            self._report_progress(f"Researching question {i+1}/{len(self.questions)}: {question}")
            result = self._research_question(question)
            research_results[question] = result
            self._report_progress(f"Completed research for question {i+1}/{len(self.questions)}", "complete")
        
        # Step 3: Critique and validate findings
        self._report_progress("Validating research findings with Medical Critique Agent")
        validated_results = {}
        for question, result in research_results.items():
            self.current_question = question
            self._report_progress(f"Validating findings for: {question}")
            critique = self._critique_findings(result)
            validated_results[question] = {
                "original_findings": result,
                "critique": critique
            }
        self._report_progress("Completed validation of all research findings", "complete")
        
        # Step 4: Generate final report with critiques
        self._report_progress("Synthesizing all research findings into comprehensive report")
        final_report = self._generate_critiqued_report(query, self.questions, validated_results)
        self._report_progress("Research process completed, final report ready", "complete")
        
        return final_report

# Streamlit Interface
def main():
    st.set_page_config(page_title="DoctrAI Medical Assistant", page_icon="🏥", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .research-status {
        background-color: #121212;
        color: #e0e0e0;
        border-left: 5px solid #BB86FC;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        word-break: break-word;
        overflow-x: auto;
        max-height: 400px;
        overflow-y: auto;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 5px 15px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #388E3C;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .deep-research-toggle {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .deep-research-toggle p {
        margin: 0 10px 0 0;
    }
    .research-header {
        color: #BB86FC;
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 1.2em;
    }
    .research-question {
        color: #03DAC6;
        margin: 8px 0;
        font-weight: bold;
    }
    .research-tool {
        color: #CF6679;
        margin: 3px 0;
    }
    .critique-badge {
        color: #FF9800;
        margin: 5px 0;
        font-weight: bold;
        display: inline-block;
        padding: 3px 8px;
        border: 1px solid #FF9800;
        border-radius: 12px;
        font-size: 0.9em;
    }
    .research-complete {
        color: #4CAF50;
        font-weight: bold;
    }
    .research-error {
        color: #FF5252;
        font-weight: bold;
    }
    .tool-result {
        background-color: #1E1E1E;
        border-left: 3px solid #03DAC6;
        padding: 10px;
        margin: 8px 0;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
    .sequential-question {
        background-color: #1E1E1E;
        border: 1px solid #BB86FC;
        padding: 12px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .question-header {
        color: #03DAC6;
        font-weight: bold;
        margin-bottom: 8px;
        font-size: 1.1em;
        border-bottom: 1px solid #333;
        padding-bottom: 5px;
    }
    .report-container {
        background-color: #FFFFFF;
        color: #333333;
        border: 1px solid #DDDDDD;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        font-family: 'Arial', sans-serif;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .report-header {
        color: #0066CC;
        font-size: 1.4em;
        font-weight: bold;
        border-bottom: 2px solid #0066CC;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .report-section {
        margin: 15px 0;
    }
    .report-section-title {
        color: #0066CC;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App title and introduction
    st.title("🏥 DoctrAI Medical Assistant")
    st.markdown("""
    A powerful AI-powered medical assistant to help you understand symptoms, medications, and lifestyle recommendations.
    """)
    
    # Initialize doctor_ai instance
    doctor_ai = DoctorAI()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Symptom Checker", "Medication Information", "Lifestyle Recommendations"])
    
    # Symptom Checker Tab
    with tab1:
        st.header("Symptom Checker")
        st.markdown("Describe your symptoms below and get general medical advice.")
        
        symptoms = st.text_area("Describe your symptoms:", height=100, key="symptoms_input")
        
        # Deep research toggle near the query input
        col1, col2 = st.columns([3, 1])
        with col2:
            st.markdown('<div class="deep-research-toggle">', unsafe_allow_html=True)
            deep_research = st.toggle("Use Deep Research", key="symptoms_research")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Get Medical Advice", key="symptoms_button"):
            if symptoms:
                with st.spinner("Analyzing symptoms..."):
                    if deep_research:
                        # Show research status
                        research_status = st.empty()
                        questions = []
                        current_tools = set()
                        
                        def report_progress(message, status="running"):
                            # Filter and display only the most relevant information
                            if "Generated" in message and "research questions" in message:
                                # Extract and display research questions
                                questions_text = message.split("Generated")[1].split("research questions")[0].strip()
                                questions_list = message.split("\n- ")
                                questions.clear()
                                for q in questions_list[1:]:
                                    questions.append(q.strip())
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Deep Medical Research</div>
                                <div>Generated {questions_text} focused questions to investigate:</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Researching question" in message:
                                # Show only the current question being researched
                                current_question = message.split("Researching question ")[1].split(":")[1].strip()
                                question_number = message.split("Researching question ")[1].split("/")[0].strip()
                                total_questions = message.split("/")[1].split(":")[0].strip()
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Deep Medical Research</div>
                                <div class="sequential-question">
                                    <div class="question-header">Question {question_number}/{total_questions}</div>
                                    <div>{current_question}</div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Using" in message and "to research" in message:
                                # Show which tool is being used for the current question
                                tool = message.split("Using ")[1].split(" to research")[0].strip()
                                research_question = message.split("to research: ")[1].strip() if "to research:" in message else ""
                                current_tools.add(tool)
                                
                                # Get the current question and question number from the orchestrator
                                question_index = doctor_ai.research_system.current_question_index
                                total_questions = len(doctor_ai.research_system.questions)
                                current_question = doctor_ai.research_system.current_question
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Deep Medical Research</div>
                                <div class="sequential-question">
                                    <div class="question-header">Question {question_index+1}/{total_questions}</div>
                                    <div>{current_question}</div>
                                    <div style="margin-top:10px;">🔎 <span style="color:#CF6679;">Using {tool}</span> to find information...</div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Results from" in message and "chars" in message:
                                # Show results from a specific tool in a beautified manner
                                tool = message.split("Results from ")[1].split(" (")[0].strip()
                                chars = message.split("(")[1].split(" chars")[0].strip()
                                result_preview = message.split("): ")[1].strip() if "): " in message else ""
                                
                                # Get the current question and question number from the orchestrator
                                question_index = doctor_ai.research_system.current_question_index
                                total_questions = len(doctor_ai.research_system.questions)
                                current_question = doctor_ai.research_system.current_question
                                current_findings = doctor_ai.research_system.current_findings
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Deep Medical Research</div>
                                <div class="sequential-question">
                                    <div class="question-header">Question {question_index+1}/{total_questions}</div>
                                    <div>{current_question}</div>
                                    <div style="margin-top:10px;">✅ <span style="color:#CF6679;">{tool}</span> found information ({chars} characters)</div>
                                    <div class="tool-result">{current_findings[:500]}{'...' if len(current_findings) > 500 else ''}</div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Validating findings for" in message:
                                # Show when validation is happening
                                question_text = message.split("Validating findings for: ")[1].strip() if "Validating findings for:" in message else "research findings"
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Deep Medical Research</div>
                                <div class="sequential-question">
                                    <div class="question-header">Validating Research</div>
                                    <div>{question_text}</div>
                                    <div style="margin-top:10px;"><span class="critique-badge">🔍 Medical Critique Agent</span> Validating accuracy of findings</div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Synthesizing" in message and "comprehensive report" in message:
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Deep Medical Research</div>
                                <div class="research-complete">✓ Research complete! Generating final medical report...</div>
                                <div style="margin-top:10px;"><span class="critique-badge">✓ Findings validated by Medical Critique Agent</span></div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif status == "complete" and "Research process completed" in message:
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Deep Medical Research</div>
                                <div class="research-complete">✓ Comprehensive medical report ready</div>
                                <div style="margin-top:10px;"><span class="critique-badge">✓ All findings validated by Medical Critique Agent</span></div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        doctor_ai.research_system.set_progress_callback(report_progress)
                    
                    advice = doctor_ai.get_medical_advice(symptoms, deep_research)
                    
                    st.markdown("### Medical Assessment")
                    
                    # Display the final report in a more professional format if deep research was used
                    if deep_research:
                        # Format the report with professional styling
                        st.markdown("""
                        <div class="report-container">
                            <div class="report-header">Medical Research Report</div>
                            <div class="report-content">
                        """, unsafe_allow_html=True)
                        
                        # Split the report into sections
                        sections = advice.split("\n## ")
                        
                        # Display the first part (Executive Summary)
                        st.markdown(sections[0], unsafe_allow_html=True)
                        
                        # Display remaining sections with better formatting
                        for section in sections[1:]:
                            section_title = section.split("\n")[0]
                            section_content = "\n".join(section.split("\n")[1:])
                            
                            st.markdown(f"""
                            <div class="report-section">
                                <div class="report-section-title">## {section_title}</div>
                                {section_content}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("""
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Regular display for non-deep research results
                        st.markdown(advice)
            else:
                st.warning("Please describe your symptoms first.")
    
    # Medication Information Tab
    with tab2:
        st.header("Medication Information")
        st.markdown("Enter a medication name to get detailed information.")
        
        medication = st.text_input("Medication name:", key="medication_input")
        
        # Deep research toggle near the query input
        col1, col2 = st.columns([3, 1])
        with col2:
            st.markdown('<div class="deep-research-toggle">', unsafe_allow_html=True)
            deep_research_med = st.toggle("Use Deep Research", key="medication_research")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Get Medication Info", key="medication_button"):
            if medication:
                with st.spinner("Researching medication..."):
                    if deep_research_med:
                        # Show research status
                        research_status = st.empty()
                        questions = []
                        current_tools = set()
                        
                        def report_progress(message, status="running"):
                            # Filter and display only the most relevant information
                            if "Generated" in message and "research questions" in message:
                                # Extract and display research questions
                                questions_text = message.split("Generated")[1].split("research questions")[0].strip()
                                questions_list = message.split("\n- ")
                                questions.clear()
                                for q in questions_list[1:]:
                                    questions.append(q.strip())
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Medication Research</div>
                                <div>Generated {questions_text} focused questions to investigate:</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Researching question" in message:
                                # Show only the current question being researched
                                current_question = message.split("Researching question ")[1].split(":")[1].strip()
                                question_number = message.split("Researching question ")[1].split("/")[0].strip()
                                total_questions = message.split("/")[1].split(":")[0].strip()
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Medication Research</div>
                                <div class="sequential-question">
                                    <div class="question-header">Question {question_number}/{total_questions}</div>
                                    <div>{current_question}</div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Using" in message and "to research" in message:
                                # Show which tool is being used for the current question
                                tool = message.split("Using ")[1].split(" to research")[0].strip()
                                research_question = message.split("to research: ")[1].strip() if "to research:" in message else ""
                                current_tools.add(tool)
                                
                                # Get the current question and question number from the orchestrator
                                question_index = doctor_ai.research_system.current_question_index
                                total_questions = len(doctor_ai.research_system.questions)
                                current_question = doctor_ai.research_system.current_question
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Medication Research</div>
                                <div class="sequential-question">
                                    <div class="question-header">Question {question_index+1}/{total_questions}</div>
                                    <div>{current_question}</div>
                                    <div style="margin-top:10px;">🔎 <span style="color:#CF6679;">Using {tool}</span> to find information...</div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Results from" in message and "chars" in message:
                                # Show results from a specific tool in a beautified manner
                                tool = message.split("Results from ")[1].split(" (")[0].strip()
                                chars = message.split("(")[1].split(" chars")[0].strip()
                                result_preview = message.split("): ")[1].strip() if "): " in message else ""
                                
                                # Get the current question and question number from the orchestrator
                                question_index = doctor_ai.research_system.current_question_index
                                total_questions = len(doctor_ai.research_system.questions)
                                current_question = doctor_ai.research_system.current_question
                                current_findings = doctor_ai.research_system.current_findings
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Medication Research</div>
                                <div class="sequential-question">
                                    <div class="question-header">Question {question_index+1}/{total_questions}</div>
                                    <div>{current_question}</div>
                                    <div style="margin-top:10px;">✅ <span style="color:#CF6679;">{tool}</span> found information ({chars} characters)</div>
                                    <div class="tool-result">{current_findings[:500]}{'...' if len(current_findings) > 500 else ''}</div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Validating findings for" in message:
                                # Show when validation is happening
                                question_text = message.split("Validating findings for: ")[1].strip() if "Validating findings for:" in message else "research findings"
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Medication Research</div>
                                <div class="sequential-question">
                                    <div class="question-header">Validating Research</div>
                                    <div>{question_text}</div>
                                    <div style="margin-top:10px;"><span class="critique-badge">🔍 Medical Critique Agent</span> Validating accuracy of findings</div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Synthesizing" in message and "comprehensive report" in message:
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Medication Research</div>
                                <div class="research-complete">✓ Research complete! Generating final medication report...</div>
                                <div style="margin-top:10px;"><span class="critique-badge">✓ Findings validated by Medical Critique Agent</span></div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif status == "complete" and "Research process completed" in message:
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Medication Research</div>
                                <div class="research-complete">✓ Comprehensive medication report ready</div>
                                <div style="margin-top:10px;"><span class="critique-badge">✓ All findings validated by Medical Critique Agent</span></div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        doctor_ai.research_system.set_progress_callback(report_progress)
                    
                    med_info = doctor_ai.get_medication_info(medication, deep_research_med)
                    
                    st.markdown("### Medication Information")
                    
                    # Display the final report in a more professional format if deep research was used
                    if deep_research_med:
                        # Format the report with professional styling
                        st.markdown("""
                        <div class="report-container">
                            <div class="report-header">Medication Research Report</div>
                            <div class="report-content">
                        """, unsafe_allow_html=True)
                        
                        # Split the report into sections
                        sections = med_info.split("\n## ")
                        
                        # Display the first part (Executive Summary)
                        st.markdown(sections[0], unsafe_allow_html=True)
                        
                        # Display remaining sections with better formatting
                        for section in sections[1:]:
                            section_title = section.split("\n")[0]
                            section_content = "\n".join(section.split("\n")[1:])
                            
                            st.markdown(f"""
                            <div class="report-section">
                                <div class="report-section-title">## {section_title}</div>
                                {section_content}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("""
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Regular display for non-deep research results
                        st.markdown(med_info)
            else:
                st.warning("Please enter a medication name first.")
    
    # Lifestyle Recommendations Tab
    with tab3:
        st.header("Lifestyle Recommendations")
        st.markdown("Ask for lifestyle advice based on your health goals.")
        
        lifestyle_query = st.text_area("What lifestyle advice are you looking for?", height=100, key="lifestyle_input")
        
        # Deep research toggle near the query input
        col1, col2 = st.columns([3, 1])
        with col2:
            st.markdown('<div class="deep-research-toggle">', unsafe_allow_html=True)
            deep_research_lifestyle = st.toggle("Use Deep Research", key="lifestyle_research")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Get Lifestyle Advice", key="lifestyle_button"):
            if lifestyle_query:
                with st.spinner("Generating lifestyle recommendations..."):
                    if deep_research_lifestyle:
                        # Show research status
                        research_status = st.empty()
                        questions = []
                        current_tools = set()
                        
                        def report_progress(message, status="running"):
                            # Filter and display only the most relevant information
                            if "Generated" in message and "research questions" in message:
                                # Extract and display research questions
                                questions_text = message.split("Generated")[1].split("research questions")[0].strip()
                                questions_list = message.split("\n- ")
                                questions.clear()
                                for q in questions_list[1:]:
                                    questions.append(q.strip())
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Lifestyle Research</div>
                                <div>Generated {questions_text} focused questions to investigate:</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Researching question" in message:
                                # Show only the current question being researched
                                current_question = message.split("Researching question ")[1].split(":")[1].strip()
                                question_number = message.split("Researching question ")[1].split("/")[0].strip()
                                total_questions = message.split("/")[1].split(":")[0].strip()
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Lifestyle Research</div>
                                <div class="sequential-question">
                                    <div class="question-header">Question {question_number}/{total_questions}</div>
                                    <div>{current_question}</div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Using" in message and "to research" in message:
                                # Show which tool is being used for the current question
                                tool = message.split("Using ")[1].split(" to research")[0].strip()
                                research_question = message.split("to research: ")[1].strip() if "to research:" in message else ""
                                current_tools.add(tool)
                                
                                # Get the current question and question number from the orchestrator
                                question_index = doctor_ai.research_system.current_question_index
                                total_questions = len(doctor_ai.research_system.questions)
                                current_question = doctor_ai.research_system.current_question
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Lifestyle Research</div>
                                <div class="sequential-question">
                                    <div class="question-header">Question {question_index+1}/{total_questions}</div>
                                    <div>{current_question}</div>
                                    <div style="margin-top:10px;">🔎 <span style="color:#CF6679;">Using {tool}</span> to find information...</div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Results from" in message and "chars" in message:
                                # Show results from a specific tool in a beautified manner
                                tool = message.split("Results from ")[1].split(" (")[0].strip()
                                chars = message.split("(")[1].split(" chars")[0].strip()
                                result_preview = message.split("): ")[1].strip() if "): " in message else ""
                                
                                # Get the current question and question number from the orchestrator
                                question_index = doctor_ai.research_system.current_question_index
                                total_questions = len(doctor_ai.research_system.questions)
                                current_question = doctor_ai.research_system.current_question
                                current_findings = doctor_ai.research_system.current_findings
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Lifestyle Research</div>
                                <div class="sequential-question">
                                    <div class="question-header">Question {question_index+1}/{total_questions}</div>
                                    <div>{current_question}</div>
                                    <div style="margin-top:10px;">✅ <span style="color:#CF6679;">{tool}</span> found information ({chars} characters)</div>
                                    <div class="tool-result">{current_findings[:500]}{'...' if len(current_findings) > 500 else ''}</div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Validating findings for" in message:
                                # Show when validation is happening
                                question_text = message.split("Validating findings for: ")[1].strip() if "Validating findings for:" in message else "research findings"
                                
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Lifestyle Research</div>
                                <div class="sequential-question">
                                    <div class="question-header">Validating Research</div>
                                    <div>{question_text}</div>
                                    <div style="margin-top:10px;"><span class="critique-badge">🔍 Medical Critique Agent</span> Validating accuracy of findings</div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif "Synthesizing" in message and "comprehensive report" in message:
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Lifestyle Research</div>
                                <div class="research-complete">✓ Research complete! Generating final lifestyle report...</div>
                                <div style="margin-top:10px;"><span class="critique-badge">✓ Findings validated by Medical Critique Agent</span></div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif status == "complete" and "Research process completed" in message:
                                research_status.markdown(f"""
                                <div class="research-status">
                                <div class="research-header">🔍 Lifestyle Research</div>
                                <div class="research-complete">✓ Comprehensive lifestyle report ready</div>
                                <div style="margin-top:10px;"><span class="critique-badge">✓ All findings validated by Medical Critique Agent</span></div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        doctor_ai.research_system.set_progress_callback(report_progress)
                    
                    lifestyle_advice = doctor_ai.get_lifestyle_advice(lifestyle_query, deep_research_lifestyle)
                    
                    st.markdown("### Lifestyle Recommendations")
                    
                    # Display the final report in a more professional format if deep research was used
                    if deep_research_lifestyle:
                        # Format the report with professional styling
                        st.markdown("""
                        <div class="report-container">
                            <div class="report-header">Lifestyle Research Report</div>
                            <div class="report-content">
                        """, unsafe_allow_html=True)
                        
                        # Split the report into sections
                        sections = lifestyle_advice.split("\n## ")
                        
                        # Display the first part (Executive Summary)
                        st.markdown(sections[0], unsafe_allow_html=True)
                        
                        # Display remaining sections with better formatting
                        for section in sections[1:]:
                            section_title = section.split("\n")[0]
                            section_content = "\n".join(section.split("\n")[1:])
                            
                            st.markdown(f"""
                            <div class="report-section">
                                <div class="report-section-title">## {section_title}</div>
                                {section_content}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("""
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Regular display for non-deep research results
                        st.markdown(lifestyle_advice)
            else:
                st.warning("Please enter your lifestyle query first.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Disclaimer: This AI assistant provides general information and is not a substitute for professional medical advice. Always consult with a healthcare provider for medical concerns.*")

if __name__ == "__main__":
    main()
