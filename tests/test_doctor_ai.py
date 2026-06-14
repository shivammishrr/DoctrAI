import json
from unittest.mock import MagicMock, patch
import pytest

from core.doctor_ai import DoctorAI


class TestDoctorAI:
    def test_initialization(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager"):
            ai = DoctorAI()
            assert ai.sessions == {}
            assert ai.research_orchestrator is not None

    def test_get_or_create_session_creates_new(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager"):
            ai = DoctorAI()
            session = ai._get_or_create_session("test-id", "symptom")
            assert "test-id" in ai.sessions
            assert len(session["history"]) == 1
            assert session["history"][0]["role"] == "system"

    def test_get_or_create_session_returns_existing(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager"):
            ai = DoctorAI()
            s1 = ai._get_or_create_session("dup-id", "symptom")
            s2 = ai._get_or_create_session("dup-id", "medication")
            assert s1 is s2
            assert s1["history"][0]["role"] == "system"

    def test_add_system_observation(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager"):
            ai = DoctorAI()
            ai._get_or_create_session("obs-id", "symptom")
            ai.add_system_observation("obs-id", "Research done.")
            session = ai.sessions["obs-id"]
            assert session["history"][-1]["role"] == "system"
            assert "Research done" in session["history"][-1]["content"]

    def test_process_turn_ask_clarifying_question(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager") as MockMM:
            mm_instance = MagicMock()
            MockMM.return_value = mm_instance

            message = MagicMock()
            message.content = "intermediate"
            message.model_dump.return_value = {"role": "assistant", "content": "intermediate", "tool_calls": None}

            tc = MagicMock()
            tc.id = "call_1"
            tc.function.name = "ask_clarifying_question"
            tc.function.arguments = json.dumps({"question": "What is your age?"})
            message.tool_calls = [tc]

            mm_instance.create_completion.return_value = (message, "llama3-70b-8192")

            ai = DoctorAI()
            result = ai.process_turn("sess-1", "I have a headache", "symptom")

            assert result["type"] == "conversation"
            assert "What is your age?" in result["content"]
            assert result["finish_reason"] == "ask_clarifying_question"

    def test_process_turn_final_answer(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager") as MockMM:
            mm_instance = MagicMock()
            MockMM.return_value = mm_instance

            message = MagicMock()
            message.content = "intermediate"
            message.model_dump.return_value = {"role": "assistant", "content": "intermediate", "tool_calls": None}

            tc = MagicMock()
            tc.id = "call_2"
            tc.function.name = "FinalAnswer"
            tc.function.arguments = json.dumps({"summary": "You should rest."})
            message.tool_calls = [tc]

            mm_instance.create_completion.return_value = (message, "llama3-70b-8192")

            ai = DoctorAI()
            result = ai.process_turn("sess-2", "What should I do?", "symptom")

            assert result["type"] == "conversation"
            assert "rest" in result["content"]
            assert result["finish_reason"] == "final_answer"

    def test_process_turn_tool_execution_then_final_answer(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager") as MockMM, \
             patch("core.doctor_ai.DoctorAI._execute_research_tool") as mock_exec:

            mm_instance = MagicMock()
            MockMM.return_value = mm_instance

            tc1 = MagicMock()
            tc1.id = "call_t1"
            tc1.function.name = "tavily_medical_search"
            tc1.function.arguments = json.dumps({"query": "headache causes"})

            msg1 = MagicMock()
            msg1.content = None
            msg1.model_dump.return_value = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_t1", "function": {"name": "tavily_medical_search", "arguments": '{"query": "headache causes"}'}, "type": "function"}]}
            msg1.tool_calls = [tc1]

            tc2 = MagicMock()
            tc2.id = "call_t2"
            tc2.function.name = "FinalAnswer"
            tc2.function.arguments = json.dumps({"summary": "Headache can be caused by stress."})

            msg2 = MagicMock()
            msg2.content = None
            msg2.model_dump.return_value = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_t2", "function": {"name": "FinalAnswer", "arguments": '{"summary": "Headache can be caused by stress."}'}, "type": "function"}]}
            msg2.tool_calls = [tc2]

            mm_instance.create_completion.side_effect = [
                (msg1, "llama3-70b-8192"),
                (msg2, "llama3-70b-8192"),
            ]

            mock_exec.return_value = "Stress is a common cause of headaches."

            ai = DoctorAI()
            result = ai.process_turn("sess-3", "Why do I have headaches?", "symptom")

            assert result["type"] == "conversation"
            assert "stress" in result["content"].lower()
            assert result["finish_reason"] == "final_answer"
            assert mock_exec.call_count == 1

    def test_process_turn_error_response(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager") as MockMM:
            mm_instance = MagicMock()
            MockMM.return_value = mm_instance
            mm_instance.create_completion.return_value = (None, "llama3-70b-8192")

            ai = DoctorAI()
            result = ai.process_turn("sess-4", "Hello", "symptom")
            assert result["type"] == "error"

    def test_process_turn_reaches_max_iterations(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager") as MockMM, \
             patch("core.doctor_ai.DoctorAI._execute_research_tool") as mock_exec:

            mm_instance = MagicMock()
            MockMM.return_value = mm_instance

            def make_tool_call_msg(name, args, call_id):
                tc = MagicMock()
                tc.id = call_id
                tc.function.name = name
                tc.function.arguments = json.dumps(args)

                msg = MagicMock()
                msg.content = None
                msg.tool_calls = [tc]
                msg.model_dump.return_value = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}, "type": "function"}],
                }
                return msg

            search_calls = [
                make_tool_call_msg("tavily_medical_search", {"query": "test"}, f"call_{i}")
                for i in range(5)
            ]
            mm_instance.create_completion.side_effect = [
                (msg, "llama3-70b-8192") for msg in search_calls
            ]

            mock_exec.return_value = "Some result."

            ai = DoctorAI()
            result = ai.process_turn("sess-5", "test query", "symptom")

            assert result["type"] == "error"
            assert "allowed steps" in result["content"]

    def test_start_deep_research(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager"), \
             patch("core.doctor_ai.MedicalResearchOrchestrator") as MockOrch:

            mock_orch_instance = MagicMock()
            MockOrch.return_value = mock_orch_instance

            ai = DoctorAI()
            from queue import Queue
            q = Queue()
            ai.start_deep_research("sess-deep", "test query", "symptom", q)

            assert mock_orch_instance.set_progress_callback.called
            assert mock_orch_instance.run_research_workflow.called

    def test_get_history(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager"):
            ai = DoctorAI()
            ai._get_or_create_session("hist-id", "symptom")
            history = ai._get_history("hist-id")
            assert len(history) == 1
            assert history[0]["role"] == "system"
            assert history is not ai.sessions["hist-id"]["history"]

    def test_get_history_nonexistent(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager"):
            ai = DoctorAI()
            history = ai._get_history("nonexistent")
            assert history == []

    def test_execute_research_tool_unknown(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager"):
            ai = DoctorAI()
            tc = MagicMock()
            tc.function.name = "nonexistent_tool"
            tc.function.arguments = "{}"
            result = ai._execute_research_tool(tc)
            assert "Unknown tool" in result

    def test_execute_research_tool_invalid_json(self, mock_env_vars):
        with patch("core.doctor_ai.ModelManager"):
            ai = DoctorAI()
            tc = MagicMock()
            tc.function.name = "tavily_medical_search"
            tc.function.arguments = "not valid json"
            result = ai._execute_research_tool(tc)
            assert "Invalid arguments" in result
