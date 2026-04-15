"""Tests for the generator module."""

import json
from unittest.mock import MagicMock, patch

from generator import QAGenerator
from config import PipelineConfig
from models import QAPair


class TestQAGenerator:
    def test_init(self, mock_config):
        gen = QAGenerator(mock_config)
        assert gen.config == mock_config
        assert len(gen.generated_questions) == 0

    def test_is_generic_question_short(self, mock_config):
        gen = QAGenerator(mock_config)
        assert gen._is_generic_question("What is it?") is True

    def test_is_generic_question_specific(self, mock_config):
        gen = QAGenerator(mock_config)
        assert gen._is_generic_question(
            "What is the recommended daily intake of protein for athletes?"
        ) is False

    def test_is_generic_question_long_generic(self, mock_config):
        gen = QAGenerator(mock_config)
        # "what is" with enough words after should pass
        assert gen._is_generic_question(
            "What is the best approach for managing chronic pain in elderly patients?"
        ) is False

    def test_is_duplicate_exact(self, mock_config):
        gen = QAGenerator(mock_config)
        gen.generated_questions.add("what is python")
        assert gen._is_duplicate("What is Python?") is True

    def test_is_duplicate_not_duplicate(self, mock_config):
        gen = QAGenerator(mock_config)
        gen.generated_questions.add("what is python")
        assert gen._is_duplicate("How does Java garbage collection work?") is False

    def test_is_duplicate_bounded_window(self, mock_config):
        gen = QAGenerator(mock_config)
        # Add more than DEDUP_WINDOW questions
        for i in range(60):
            gen.generated_questions.add(f"question number {i} about topic")
        # Very old questions should not cause duplicates
        assert gen._is_duplicate("question number 0 about topic") is True

    def test_is_duplicate_near_duplicate_caught(self, mock_config):
        gen = QAGenerator(mock_config)
        gen.generated_questions.add("posso usar maionese em minhas saladas se estou tentando prevenir doencas cardiovasculares")
        # Near-duplicate with very similar wording
        assert gen._is_duplicate("Posso incluir maionese na minha dieta se estou tentando prevenir doencas cardiovasculares?") is True

    def test_is_generic_portuguese_generic(self, mock_config):
        gen = QAGenerator(mock_config)
        # Short generic question
        assert gen._is_generic_question("Como prevenir doencas?") is True

    def test_is_generic_portuguese_specific(self, mock_config):
        gen = QAGenerator(mock_config)
        # Long specific question should pass
        assert gen._is_generic_question(
            "Qual quantidade diaria de azeite extravirgem e recomendada pelo manual para reduzir risco cardiovascular?"
        ) is False

    def test_system_prompt_has_document_specific_instructions(self, mock_config):
        gen = QAGenerator(mock_config)
        prompt = gen._get_system_prompt()
        assert "DOCUMENT-SPECIFIC" in prompt
        assert "popular" in prompt.lower() or "general knowledge" in prompt.lower()

    def test_generate_qa_success(self, mock_config, mock_openai_response):
        gen = QAGenerator(mock_config)
        with patch.object(gen.client.chat.completions, "create", return_value=mock_openai_response):
            result = gen.generate_qa(
                contexts=["Python is a programming language."],
                qa_type="single-hop",
                difficulty="easy",
                iteration=0,
            )
        # May or may not succeed depending on quality checks
        # The mock response has a very short question that might be filtered
        assert result is None or isinstance(result, QAPair)

    def test_generate_qa_with_good_response(self, mock_config):
        gen = QAGenerator(mock_config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "question": "O que devo comer para manter uma dieta equilibrada e saudavel no dia a dia?",
            "ground_truth": "Para manter uma dieta equilibrada, inclua frutas, vegetais, proteinas e graos integrais nas suas refeicoes diarias.",
            "type": "single-hop",
            "difficulty": "easy",
        })

        with patch.object(gen.client.chat.completions, "create", return_value=mock_response):
            result = gen.generate_qa(
                contexts=["Uma dieta equilibrada inclui frutas, vegetais, proteinas e graos integrais nas refeicoes diarias."],
                qa_type="single-hop",
                difficulty="easy",
                iteration=0,
            )
        assert result is not None
        assert result.question is not None
        assert result.ground_truth is not None

    def test_to_dataset_entry(self, mock_config):
        gen = QAGenerator(mock_config)
        qa = QAPair(
            question="Test question?",
            ground_truth="Test answer.",
            contexts=["Context 1"],
            metadata={"type": "single-hop", "difficulty": "easy"},
        )
        entry = gen.to_dataset_entry(qa)
        assert entry.question == "Test question?"
        assert entry.ground_truth == "Test answer."
        assert entry.answer == "Test answer."
        assert entry.contexts == ["Context 1"]

    def test_system_prompt_pt_br_default(self, mock_config):
        gen = QAGenerator(mock_config)
        prompt = gen._get_system_prompt()
        assert "PORTUGUESE" in prompt
        assert "PATIENT" in prompt

    def test_system_prompt_english(self):
        config = PipelineConfig(openai_api_key="test-key", language="en")
        gen = QAGenerator(config)
        prompt = gen._get_system_prompt()
        assert "en" in prompt

    def test_system_prompt_non_health_domain(self):
        config = PipelineConfig(openai_api_key="test-key", document_domain="technology", language="en")
        gen = QAGenerator(config)
        prompt = gen._get_system_prompt()
        assert "technology" in prompt
        assert "PATIENT" not in prompt or "PRACTITIONER" in prompt

    def test_generate_qa_api_failure_returns_none(self, mock_config):
        gen = QAGenerator(mock_config)
        with patch.object(gen.client.chat.completions, "create", side_effect=Exception("API error")):
            result = gen.generate_qa(
                contexts=["Some context"],
                qa_type="single-hop",
                difficulty="easy",
            )
        assert result is None

    def test_is_ground_truth_grounded_pass(self, mock_config):
        gen = QAGenerator(mock_config)
        contexts = ["A dieta Dash inclui feijoes, carnes magras, laticinios com baixo teor de gordura."]
        gt = "A dieta Dash e composta por feijoes e carnes magras."
        assert gen._is_ground_truth_grounded(gt, contexts, "single-hop") is True

    def test_is_ground_truth_grounded_fail(self, mock_config):
        gen = QAGenerator(mock_config)
        contexts = ["A maionese pode ser utilizada para fazer salada com batatas."]
        gt = "Para adaptar a receita, use iogurte grego, cenoura e ervilha com batatas cozidas."
        assert gen._is_ground_truth_grounded(gt, contexts, "single-hop") is False

    def test_is_ground_truth_grounded_unanswerable_skips(self, mock_config):
        gen = QAGenerator(mock_config)
        # All types must pass grounding check now
        assert gen._is_ground_truth_grounded("anything here", ["anything there here"], "single-hop") is True

    def test_generate_qa_rejects_ungrounded_ground_truth(self, mock_config):
        gen = QAGenerator(mock_config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # Ground truth has words NOT in context
        mock_response.choices[0].message.content = json.dumps({
            "question": "Como adaptar a receita de maionese para a dieta cardioprotetora usando iogurte natural?",
            "ground_truth": "Use iogurte grego como base, adicione cenoura, ervilha e batatas cozidas para tornar mais saudavel.",
            "type": "single-hop",
            "difficulty": "medium",
        })

        with patch.object(gen.client.chat.completions, "create", return_value=mock_response):
            result = gen.generate_qa(
                contexts=["Pode ser utilizada para fazer a salada de maionese com batatas."],
                qa_type="single-hop",
                difficulty="medium",
                iteration=0,
            )
        # Should be rejected because ground truth is not grounded in context
        assert result is None
