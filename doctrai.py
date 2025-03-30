import os
import re
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import datetime
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
            split_pattern = '\n'
            self._report_progress(f"Generated {len(questions_text.split(split_pattern))} research questions:\n{questions_text}", "complete")
            
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
                    self._report_progress(f"Results from {tool_name} ({len(tool_result)} chars)")
                    research_results.append(f"Results from {tool_name}:\n{tool_result}")
                else:
                    self._report_progress(f"No significant results from {tool_name}")
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

IMPORTANT: Do not include any HTML tags or markdown formatting in your response. Use plain text only.
"""
        
        try:
            content, model_used = self.model_manager.create_completion(
                messages=[{"role": "system", "content": "You are a medical research assistant that creates comprehensive reports based on research findings. Do not use HTML tags in your response."},
                         {"role": "user", "content": prompt}],
                reasoning_level="high",
                temperature=0.3,
                max_tokens=1500
            )
            
            # Clean any potential HTML tags from the content
            content = content.replace("<", "&lt;").replace(">", "&gt;")
            
            print(f"Used model: {model_used} for generating final report")
            self._report_progress("Final report generated successfully", "complete")
            
            # Format the final report with proper headers and structure
            final_report = f"""# MEDICAL RESEARCH REPORT
Query
{query}

Research Findings
{content}

Research Methodology
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
Query
{query}

Raw Research Findings
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
    """Main Streamlit interface."""
    global research_status  # Make research_status global
    
    st.set_page_config(
        page_title="DoctrAI - Medical Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e40af, #1e3a8a);
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.7rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .main-header p {
        margin-top: 0.7rem;
        font-size: 1.2rem;
        opacity: 0.95;
    }
    
    /* Card styling */
    .feature-card {
        background: linear-gradient(to right, #1e293b, #0f172a);
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        border-left: 6px solid #3b82f6;
    }
    
    .feature-card h3 {
        color: #60a5fa;
        margin-top: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #334155;
        padding: 0.7rem;
        font-size: 1.05rem;
        background-color: #1e293b;
        color: #e2e8f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #334155;
        padding: 0.7rem;
        font-size: 1.05rem;
        background-color: #1e293b;
        color: #e2e8f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(to right, #3b82f6, #60a5fa);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(to right, #2563eb, #3b82f6);
        transform: translateY(-2px);
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.25);
    }
    
    /* Research status */
    .research-status {
        background: linear-gradient(to right, #1e293b, #0f172a);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        border-left: 4px solid #3b82f6;
    }
    
    .research-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #60a5fa;
        margin-bottom: 0.7rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid #334155;
    }
    
    .research-subheader {
        font-weight: 600;
        color: #93c5fd;
        margin: 0.5rem 0;
    }
    
    .research-questions {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
        border-left: 3px solid #60a5fa;
        white-space: pre-line;
    }
    
    .progress-indicator {
        background-color: #3b82f6;
        color: white;
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .sequential-question {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.7rem;
        border: 1px solid #334155;
    }
    
    .question-header {
        font-weight: 600;
        color: #93c5fd;
        margin-bottom: 0.5rem;
    }
    
    .question-content {
        font-size: 1.05rem;
        line-height: 1.5;
        color: #e2e8f0;
    }
    
    .tool-active {
        background-color: #172554;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.7rem;
        border-left: 4px solid #60a5fa;
    }
    
    .tool-indicator {
        display: flex;
        align-items: center;
        font-weight: 500;
    }
    
    .pulse {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #60a5fa;
        margin-right: 10px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(96, 165, 250, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(96, 165, 250, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(96, 165, 250, 0);
        }
    }
    
    .tool-name-active {
        color: #60a5fa;
        font-weight: 600;
    }
    
    .research-complete {
        background-color: #064e3b;
        color: #d1fae5;
        padding: 0.7rem;
        border-radius: 8px;
        margin-top: 0.7rem;
        font-weight: 500;
    }
    
    .validation-step, .synthesis-step, .complete-step {
        padding: 0.7rem;
        border-radius: 8px;
        margin-top: 0.7rem;
        font-weight: 500;
    }
    
    .validation-step {
        background-color: #172554;
        color: #bfdbfe;
    }
    
    .synthesis-step {
        background-color: #1e3a8a;
        color: #bfdbfe;
    }
    
    .complete-step {
        background-color: #065f46;
        color: #a7f3d0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0f172a;
        padding: 10px 10px 0 10px;
        border-radius: 10px 10px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 10px 10px 0 0;
        padding: 12px 20px;
        height: auto;
        font-weight: 500;
        font-size: 1.05rem;
        color: #94a3b8;
        transition: all 0.2s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(to bottom, #3b82f6, #2563eb) !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 -4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #1e293b;
        border-radius: 0 0 10px 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Toggle switch */
    .stCheckbox > div > label {
        font-weight: 500;
        color: #e2e8f0;
    }
    
    .stCheckbox > div > label > div {
        background-color: #334155;
    }
    
    .stCheckbox > div > label > div[data-baseweb="checkbox"] > div {
        background-color: #3b82f6;
    }
    
    /* Result container */
    .result-container {
        background: linear-gradient(to right, #1e293b, #0f172a);
        border-radius: 12px;
        padding: 1.8rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border-left: 6px solid #3b82f6;
    }
    
    /* Footer */
    .footer {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 2rem;
        text-align: center;
        font-size: 0.9rem;
        color: #94a3b8;
        border-top: 1px solid #334155;
    }
    
    /* Override Streamlit's default text color */
    .stMarkdown, .stMarkdown p, .stText, h1, h2, h3, h4, h5, h6, p, span, div {
        color: #e2e8f0 !important;
    }
    
    /* Override Streamlit's default background */
    .stApp {
        background-color: #0f172a;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'doctor_ai' not in st.session_state:
        st.session_state.doctor_ai = DoctorAI()
        
    doctor_ai = st.session_state.doctor_ai
    
    # Initialize research_status as a global placeholder
    research_status = st.empty()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 DoctrAI Medical Assistant</h1>
        <p>Your AI-powered medical research and advice companion</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for the three main features
    tab1, tab2, tab3 = st.tabs(["Symptom Checker", "Medication Information", "Lifestyle Recommendations"])
    
    # Symptom Checker Tab
    with tab1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Symptom Checker</h3>
            <p>Describe your symptoms in detail for a preliminary assessment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            symptoms = st.text_area("Describe your symptoms:", height=150, 
                                   placeholder="Example: I've been experiencing a persistent headache for the past 3 days, along with mild fever and fatigue...")
        with col2:
            st.write("Research Options")
            deep_research_symptoms = st.checkbox("Deep Research", key="symptoms_research", 
                                              help="Uses multiple research tools to provide more comprehensive information")
        
        if st.button("Get Medical Advice", key="symptom_button"):
            if symptoms:
                with st.spinner("Analyzing symptoms..."):
                    if deep_research_symptoms:
                        # Set up the progress reporting
                        research_status = st.empty()  # Reset placeholder for this specific operation
                        doctor_ai.research_system.set_progress_callback(report_progress)
                        
                        advice = doctor_ai.get_medical_advice(symptoms, deep_research=True)
                        st.markdown(advice, unsafe_allow_html=True)
                    else:
                        advice = doctor_ai.get_medical_advice(symptoms)
                        st.markdown(f"""
                        <div class="result-container">
                            {advice}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please describe your symptoms first.")
    
    # Medication Information Tab
    with tab2:
        st.markdown("""
        <div class="feature-card">
            <h3>💊 Medication Information</h3>
            <p>Get detailed information about medications, including usage, side effects, and interactions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            medication = st.text_input("Enter medication name:", 
                                      placeholder="Example: Ibuprofen, Amoxicillin, Lisinopril...")
        with col2:
            st.write("Research Options")
            deep_research_med = st.checkbox("Deep Research", key="medication_research", 
                                         help="Uses multiple research tools to provide more comprehensive information")
        
        if st.button("Get Medication Info", key="medication_button"):
            if medication:
                with st.spinner("Researching medication information..."):
                    if deep_research_med:
                        # Set up the progress reporting
                        research_status = st.empty()  # Reset placeholder for this specific operation
                        doctor_ai.research_system.set_progress_callback(report_progress)
                        
                        info = doctor_ai.get_medication_info(medication, deep_research=True)
                        st.markdown(info, unsafe_allow_html=True)
                    else:
                        info = doctor_ai.get_medication_info(medication)
                        st.markdown(f"""
                        <div class="result-container">
                            {info}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a medication name first.")
    
    # Lifestyle Recommendations Tab
    with tab3:
        st.markdown("""
        <div class="feature-card">
            <h3>🌱 Lifestyle Recommendations</h3>
            <p>Get personalized lifestyle advice for managing health conditions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            condition = st.text_input("Enter health condition:", 
                                     placeholder="Example: Type 2 Diabetes, Hypertension, Asthma...")
        with col2:
            st.write("Research Options")
            deep_research_lifestyle = st.checkbox("Deep Research", key="lifestyle_research", 
                                               help="Uses multiple research tools to provide more comprehensive information")
        
        if st.button("Get Lifestyle Advice", key="lifestyle_button"):
            if condition:
                with st.spinner("Generating lifestyle recommendations..."):
                    if deep_research_lifestyle:
                        # Set up the progress reporting
                        research_status = st.empty()  # Reset placeholder for this specific operation
                        doctor_ai.research_system.set_progress_callback(report_progress)
                        
                        advice = doctor_ai.get_lifestyle_advice(condition, deep_research=True)
                        st.markdown(advice, unsafe_allow_html=True)
                    else:
                        advice = doctor_ai.get_lifestyle_advice(condition)
                        st.markdown(f"""
                        <div class="result-container">
                            {advice}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a health condition first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <strong>Disclaimer:</strong> This tool provides general information only and should not replace professional medical advice.
        <p>&copy; 2025 DoctrAI Medical Assistant | Developed with ❤️ for healthcare</p>
    </div>
    """, unsafe_allow_html=True)

# Progress reporting function for deep research
def report_progress(message, status="running"):
    """Report progress from the research orchestrator to the UI."""
    global research_status  # Access the global research_status
    
    # Filter and display only the most relevant information
    if "Generated" in message and "research questions" in message:
        # Extract and display research questions
        try:
            if ":\n" in message:
                questions_text = message.split(":\n")[1].strip()
                # Format the questions for better display
                formatted_questions = questions_text.replace("- ", "• ").replace("\n", "<br>")
                
                research_status.markdown(f"""
                <div class="research-status">
                <div class="research-header">🔍 Research Progress</div>
                <div class="research-subheader">Generated Research Questions:</div>
                <div class="research-questions">{formatted_questions}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                questions_text = message.split("Generated")[1].split("research questions")[0].strip()
                research_status.markdown(f"""
                <div class="research-status">
                <div class="research-header">🔍 Research Progress</div>
                <div>Generated {questions_text} focused questions to investigate</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            print(f"Error displaying research questions: {str(e)}")
            research_status.markdown(f"""
            <div class="research-status">
            <div class="research-header">🔍 Research Progress</div>
            <div>Generating research questions...</div>
            </div>
            """, unsafe_allow_html=True)
    
    elif "Researching question" in message:
        # Show the current question being researched with progress indicator
        try:
            # Handle format: "Researching question {i+1}/{len(self.questions)}: {question}"
            if "/" in message and ":" in message:
                progress = message.split("Researching question ")[1].split(":")[0].strip()
                current_question = message.split(":", 1)[1].strip()
                
                research_status.markdown(f"""
                <div class="research-status">
                <div class="research-header">🔍 Research Progress</div>
                <div class="sequential-question">
                    <div class="question-header">Question {progress}</div>
                    <div>{current_question}</div>
                </div>
                </div>
                """, unsafe_allow_html=True)
            # Handle simple format: "Researching question: {question}"
            elif ": " in message:
                current_question = message.split(": ")[1].strip()
                
                research_status.markdown(f"""
                <div class="research-status">
                <div class="research-header">🔍 Research Progress</div>
                <div class="sequential-question">
                    <div class="question-header">Current Research Question</div>
                    <div>{current_question}</div>
                </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            print(f"Error displaying research question: {str(e)}")
            # Fallback if parsing fails
            research_status.markdown(f"""
            <div class="research-status">
            <div class="research-header">🔍 Research Progress</div>
            <div>Researching...</div>
            </div>
            """, unsafe_allow_html=True)
    
    elif "Using" in message and "to research" in message:
        # Show which tool is being used for the current question
        try:
            tool = message.split("Using ")[1].split(" to research")[0].strip()
            research_question = message.split("to research: ")[1].strip() if "to research:" in message else ""
            
            # Get the current question and question number from the orchestrator
            try:
                if hasattr(doctor_ai, 'research_system'):
                    question_index = doctor_ai.research_system.current_question_index
                    total_questions = len(doctor_ai.research_system.questions)
                    current_question = doctor_ai.research_system.current_question
                else:
                    # Parse from the message if possible
                    if "Question" in message and "/" in message:
                        parts = message.split("Question ")
                        if len(parts) > 1:
                            question_info = parts[1].split(":")
                            if len(question_info) > 0:
                                question_numbers = question_info[0].split("/")
                                if len(question_numbers) > 1:
                                    question_index = int(question_numbers[0].strip()) - 1
                                    total_questions = int(question_numbers[1].strip())
                                else:
                                    question_index = 0
                                    total_questions = 1
                            else:
                                question_index = 0
                                total_questions = 1
                        else:
                            question_index = 0
                            total_questions = 1
                    else:
                        question_index = 0
                        total_questions = 1
                    
                    current_question = research_question if research_question else "Research question"
            except Exception as e:
                print(f"Error getting question info: {str(e)}")
                question_index = 0
                total_questions = 1
                current_question = research_question if research_question else "Research question"
            
            research_status.markdown(f"""
            <div class="research-status">
            <div class="research-header">🔍 Research Progress</div>
            <div class="sequential-question">
                <div class="question-header">Question {question_index+1}/{total_questions}</div>
                <div>{current_question}</div>
                <div style="margin-top:10px;">🔎 <span style="color:#CF6679;">Using {tool}</span> to find information...</div>
            </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            print(f"Error displaying tool usage: {str(e)}")
            research_status.markdown(f"""
            <div class="research-status">
            <div class="research-header">🔍 Research Progress</div>
            <div>Using research tools to find information...</div>
            </div>
            """, unsafe_allow_html=True)
    
    elif "Results from" in message and "chars" in message:
        # Show results from a specific tool in a beautified manner
        try:
            tool = message.split("Results from ")[1].split(" (")[0].strip()
            chars = message.split("(")[1].split(" chars")[0].strip()
            
            # Get the current question and question number from the orchestrator
            try:
                if hasattr(doctor_ai, 'research_system'):
                    question_index = doctor_ai.research_system.current_question_index
                    total_questions = len(doctor_ai.research_system.questions)
                    current_question = doctor_ai.research_system.current_question
                    current_findings = doctor_ai.research_system.current_findings
                else:
                    question_index = 0
                    total_questions = 1
                    current_question = "Research question"
                    current_findings = "Research findings in progress..."
            except Exception as e:
                print(f"Error getting research info: {str(e)}")
                question_index = 0
                total_questions = 1
                current_question = "Research question"
                current_findings = "Research findings in progress..."
            
            # Ensure we have some content for current_findings
            if not current_findings or len(current_findings) == 0:
                current_findings = "Research findings in progress..."
            
            research_status.markdown(f"""
            <div class="research-status">
            <div class="research-header">🔍 Research Progress</div>
            <div class="sequential-question">
                <div class="question-header">Question {question_index+1}/{total_questions}</div>
                <div>{current_question}</div>
                <div style="margin-top:10px;">✅ <span style="color:#CF6679;">{tool}</span> found information ({chars} characters)</div>
                <div class="tool-result">{current_findings[:500]}{'...' if len(current_findings) > 500 else ''}</div>
            </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            print(f"Error displaying tool results: {str(e)}")
            research_status.markdown(f"""
            <div class="research-status">
            <div class="research-header">🔍 Research Progress</div>
            <div class="tool-result">
                <div class="tool-name">✓ Research tool found information</div>
            </div>
            </div>
            """, unsafe_allow_html=True)
    
    elif "Validating findings" in message:
        # Show when validation is happening
        try:
            question_text = "research findings"
            if "for:" in message:
                question_text = message.split("for:")[1].strip()
            
            research_status.markdown(f"""
            <div class="research-status">
            <div class="research-header">🔍 Research Progress</div>
            <div class="sequential-question">
                <div class="question-header">Validating Research</div>
                <div>{question_text}</div>
                <div style="margin-top:10px;"><span class="critique-badge">🔍 Medical Critique Agent</span> Validating accuracy of findings</div>
            </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            print(f"Error displaying validation: {str(e)}")
            research_status.markdown(f"""
            <div class="research-status">
            <div class="research-header">🔍 Research Progress</div>
            <div class="validation-step">🔎 Validating research findings for accuracy...</div>
            </div>
            """, unsafe_allow_html=True)
    
    elif "Synthesizing" in message and "comprehensive report" in message:
        research_status.markdown(f"""
        <div class="research-status">
        <div class="research-header">🔍 Research Progress</div>
        <div class="research-complete">✓ Research complete! Generating final report...</div>
        <div style="margin-top:10px;"><span class="critique-badge">✓ Findings validated by Medical Critique Agent</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    elif status == "complete" and "Research process completed" in message:
        research_status.markdown(f"""
        <div class="research-status">
        <div class="research-header">🔍 Research Progress</div>
        <div class="research-complete">✓ Comprehensive report ready</div>
        <div style="margin-top:10px;"><span class="critique-badge">✓ All findings validated by Medical Critique Agent</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    elif "LLM call" in message:
        # Show when LLM is being called
        research_status.markdown(f"""
        <div class="research-status">
        <div class="research-header">🔍 Research Progress</div>
        <div>AI processing: {message.replace("LLM call:", "").strip()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    elif "LLM response received" in message:
        # Show when LLM response is received
        research_status.markdown(f"""
        <div class="research-status">
        <div class="research-header">🔍 Research Progress</div>
        <div>AI completed: {message.replace("LLM response received", "").strip()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    elif status == "error":
        research_status.markdown(f"""
        <div class="research-status" style="border-left: 5px solid #f44336;">
        <div class="research-header" style="color: #f44336;">⚠️ Research Error</div>
        <div>{message}</div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Default display for other messages
        research_status.markdown(f"""
        <div class="research-status">
        <div class="research-header">🔍 Research Progress</div>
        <div>{message}</div>
        </div>
        """, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()
