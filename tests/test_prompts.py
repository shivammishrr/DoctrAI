from core.prompts import get_persona_prompt, PERSONA_PROMPTS


class TestPrompts:
    def test_get_persona_prompt_symptom(self):
        prompt = get_persona_prompt("symptom")
        assert "Symptom Checker" in prompt or "symptom" in prompt.lower()
        assert "FinalAnswer" in prompt or "tavily_medical_search" in prompt

    def test_get_persona_prompt_medication(self):
        prompt = get_persona_prompt("medication")
        assert "Medication" in prompt

    def test_get_persona_prompt_lifestyle(self):
        prompt = get_persona_prompt("lifestyle")
        assert "Lifestyle" in prompt

    def test_get_persona_prompt_unknown(self):
        prompt = get_persona_prompt("unknown_role")
        assert "medical assistant" in prompt.lower()

    def test_persona_prompts_all_have_content(self):
        for key, prompt_text in PERSONA_PROMPTS.items():
            assert len(prompt_text) > 50, f"Prompt for '{key}' is too short"
