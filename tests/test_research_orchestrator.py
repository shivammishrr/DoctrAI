from unittest.mock import MagicMock, patch
import json
import pytest

from core.research_orchestrator import MedicalResearchOrchestrator


class TestMedicalResearchOrchestrator:
    def test_initialization(self):
        mm = MagicMock()
        orch = MedicalResearchOrchestrator(mm)
        assert orch.progress_callback is None
        assert orch.current_research_state == {}
        assert orch.tool_executor is not None

    def test_set_progress_callback(self):
        mm = MagicMock()
        orch = MedicalResearchOrchestrator(mm)
        def cb(msg, status): pass
        orch.set_progress_callback(cb)
        assert orch.progress_callback is cb

    def test_get_research_state_copy(self):
        mm = MagicMock()
        orch = MedicalResearchOrchestrator(mm)
        orch._update_research_state({"test_key": "test_value"})
        copy = orch.get_research_state_copy()
        assert copy["test_key"] == "test_value"
        assert copy is not orch.current_research_state

    def test_update_research_state(self):
        mm = MagicMock()
        orch = MedicalResearchOrchestrator(mm)
        orch._update_research_state({"phase": "initial"})
        assert orch.current_research_state["phase"] == "initial"
        orch._update_research_state({"phase": "updated"})
        assert orch.current_research_state["phase"] == "updated"

    def test_generate_research_questions_success(self):
        mm = MagicMock()
        mm.create_completion.return_value = (
            MagicMock(content="1. What causes diabetes?\n2. How is diabetes treated?\n3. What are diabetes complications?"),
            "llama3-70b-8192",
        )
        orch = MedicalResearchOrchestrator(mm)
        questions = orch._generate_research_questions("diabetes")
        assert len(questions) == 3
        assert "diabetes" in questions[0].lower()

    def test_generate_research_questions_fallback(self):
        mm = MagicMock()
        mm.create_completion.return_value = (
            MagicMock(content="No numbered list here."),
            "llama3-70b-8192",
        )
        orch = MedicalResearchOrchestrator(mm)
        questions = orch._generate_research_questions("test query")
        assert questions == ["test query"]

    def test_generate_research_questions_failure(self):
        mm = MagicMock()
        mm.create_completion.return_value = (None, "llama3-70b-8192")
        orch = MedicalResearchOrchestrator(mm)
        with pytest.raises(ValueError, match="Failed to generate research questions"):
            orch._generate_research_questions("test")

    def test_research_single_question_with_tools(self):
        mm = MagicMock()

        tc = MagicMock()
        tc.id = "call_r1"
        tc.function.name = "tavily_medical_search"
        tc.function.arguments = json.dumps({"query": "headache treatment"})

        msg1 = MagicMock()
        msg1.content = None
        msg1.model_dump.return_value = {"role": "assistant", "content": None}
        msg1.tool_calls = [tc]

        msg2 = MagicMock()
        msg2.content = "Migraine treatments include rest and medication."
        msg2.model_dump.return_value = {"role": "assistant", "content": "Migraine treatments include rest and medication."}
        msg2.tool_calls = None

        msg3 = MagicMock()
        msg3.content = "Headache treatment: rest and medication."
        msg3.model_dump.return_value = {"role": "assistant", "content": "Headache treatment: rest and medication."}
        msg3.tool_calls = None

        mm.create_completion.side_effect = [
            (msg1, "llama3-70b-8192"),
            (msg2, "llama3-70b-8192"),
            (msg3, "llama3-70b-8192"),
        ]

        orch = MedicalResearchOrchestrator(mm)
        with patch.object(orch.tool_executor, "execute_tavily_medical_search", return_value="Rest and medication help."):
            result = orch._research_single_question_with_tools("What helps headaches?")
            assert "Migraine" in result or "rest" in result

    def test_critique_findings(self):
        mm = MagicMock()
        mm.create_completion.return_value = (
            MagicMock(content="Confidence: High. Findings are accurate and well-sourced."),
            "llama3-70b-8192",
        )
        orch = MedicalResearchOrchestrator(mm)
        result = orch._critique_findings("Some findings here.", "test query")
        assert "High" in result
        assert orch.current_research_state.get("critique_complete") is True

    def test_critique_findings_no_response(self):
        mm = MagicMock()
        mm.create_completion.return_value = (None, "llama3-70b-8192")
        orch = MedicalResearchOrchestrator(mm)
        result = orch._critique_findings("findings", "query")
        assert result == "Critique could not be generated."

    def test_generate_final_report(self):
        mm = MagicMock()
        mm.create_completion.return_value = (
            MagicMock(content="# Final Report\n\nThis is the report."),
            "llama3-70b-8192",
        )
        orch = MedicalResearchOrchestrator(mm)
        result = orch._generate_final_report(
            "diabetes",
            {"Q1": "A1", "Q2": "A2"},
            "Confidence: High"
        )
        assert "Final Report" in result

    def test_generate_final_report_failure(self):
        mm = MagicMock()
        mm.create_completion.return_value = (None, "llama3-70b-8192")
        orch = MedicalResearchOrchestrator(mm)
        with pytest.raises(ValueError, match="failed to generate"):
            orch._generate_final_report("query", {"Q": "A"}, "critique")

    def test_run_research_workflow_full(self):
        mm = MagicMock()

        def _make_tool_call_msg(name, args, call_id):
            tc = MagicMock()
            tc.id = call_id
            tc.function.name = name
            tc.function.arguments = json.dumps(args)
            msg = MagicMock()
            msg.content = None
            msg.tool_calls = [tc]
            msg.model_dump.return_value = {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}, "type": "function"}]
            }
            return msg

        def _make_text_msg(content):
            msg = MagicMock()
            msg.content = content
            msg.tool_calls = None
            msg.model_dump.return_value = {"role": "assistant", "content": content, "tool_calls": None}
            return msg

        mm.create_completion.side_effect = [
            (_make_text_msg("1. What is diabetes?\n2. How is diabetes treated?"), "llama3-70b-8192"),

            (_make_tool_call_msg("tavily_medical_search", {"query": "diabetes definition"}, "c1"), "llama3-70b-8192"),
            (_make_text_msg("Diabetes is a metabolic disorder."), "llama3-70b-8192"),
            (_make_text_msg("Diabetes is a chronic metabolic disorder."), "llama3-70b-8192"),

            (_make_tool_call_msg("tavily_medical_search", {"query": "diabetes treatment"}, "c2"), "llama3-70b-8192"),
            (_make_text_msg("Treatment includes insulin."), "llama3-70b-8192"),
            (_make_text_msg("Diabetes treatment includes insulin therapy."), "llama3-70b-8192"),

            (_make_text_msg("Confidence: High. Findings are accurate."), "llama3-70b-8192"),

            (_make_text_msg("# Final Report\n\nFull report here."), "llama3-70b-8192"),
        ]

        orch = MedicalResearchOrchestrator(mm)
        with patch.object(orch.tool_executor, "execute_tavily_medical_search", return_value="Some result"):
            result_container = {}
            orch.run_research_workflow("diabetes", result_container)

        assert result_container['status'] == 'complete'
        assert "Final Report" in result_container['report']

    def test_run_research_workflow_error(self):
        mm = MagicMock()
        mm.create_completion.side_effect = Exception("LLM API failure")

        orch = MedicalResearchOrchestrator(mm)
        result_container = {}
        orch.run_research_workflow("diabetes", result_container)

        assert result_container['status'] == 'error'
        assert "error" in result_container['report'].lower()
