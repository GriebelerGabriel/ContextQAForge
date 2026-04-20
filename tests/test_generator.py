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

    def test_is_duplicate_near_duplicate_caught(self, mock_config):
        gen = QAGenerator(mock_config)
        gen.generated_questions.add("posso usar maionese em minhas saladas se estou tentando prevenir doencas cardiovasculares")
        assert gen._is_duplicate("Posso incluir maionese na minha dieta se estou tentando prevenir doencas cardiovasculares?") is True

    def test_is_generic_portuguese_generic(self, mock_config):
        gen = QAGenerator(mock_config)
        assert gen._is_generic_question("Como prevenir doencas?") is True

    def test_is_generic_portuguese_specific(self, mock_config):
        gen = QAGenerator(mock_config)
        assert gen._is_generic_question(
            "Qual quantidade diaria de azeite extravirgem e recomendada pelo manual para reduzir risco cardiovascular?"
        ) is False

    def test_system_prompt_has_document_specific_instructions(self, mock_config):
        gen = QAGenerator(mock_config)
        prompt = gen._get_system_prompt()
        assert "DOCUMENT-SPECIFIC" in prompt
        assert "popular" in prompt.lower() or "general knowledge" in prompt.lower()

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

    def test_is_ground_truth_grounded_pass(self, mock_config):
        gen = QAGenerator(mock_config)
        contexts = ["A dieta Dash inclui feijoes, carnes magras, laticinios com baixo teor de gordura."]
        gt = "A dieta Dash e composta por feijoes e carnes magras."
        assert gen._is_ground_truth_grounded(gt, contexts) is True

    def test_is_ground_truth_grounded_fail(self, mock_config):
        gen = QAGenerator(mock_config)
        contexts = ["A maionese pode ser utilizada para fazer salada com batatas."]
        gt = "O azeite extravirgem contem polifenois que reduzem inflamacacao arterial em 30%."
        assert gen._is_ground_truth_grounded(gt, contexts) is False

    def test_parse_json_response_plain(self, mock_config):
        result = QAGenerator._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_response_code_block(self, mock_config):
        result = QAGenerator._parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_json_response_invalid(self, mock_config):
        result = QAGenerator._parse_json_response("not json at all")
        assert result is None

    def test_generate_qa_from_section_with_mock(self, mock_config):
        """Test section-based generation with mocked LLM."""
        gen = QAGenerator(mock_config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "question": "Qual e a quantidade de sódio recomendada por dia para adultos segundo a OMS?",
            "ground_truth": "A OMS recomenda que o consumo de sódio nao ultrapasse 2g por dia para adultos.",
            "type": "single-hop",
            "difficulty": "easy",
        })

        with patch.object(gen.client.chat.completions, "create", return_value=mock_response):
            pairs = gen.generate_qa_from_section(
                chunks=["A OMS recomenda que o consumo de sódio nao ultrapasse 2g por dia para adultos."],
                topic_path=["Document Topics", "Sodium", "Recommended Intake"],
                source="who_guideline.pdf",
                num_pairs=1,
            )

        assert len(pairs) == 1
        assert pairs[0].metadata["source"] == "who_guideline.pdf"
        assert pairs[0].metadata["topic_path"] == ["Document Topics", "Sodium", "Recommended Intake"]

    def test_generate_qa_from_section_empty_chunks(self, mock_config):
        """Test that empty chunks return empty list."""
        gen = QAGenerator(mock_config)
        pairs = gen.generate_qa_from_section(
            chunks=[],
            topic_path=["Root"],
            source="test.md",
            num_pairs=1,
        )
        assert pairs == []

    def test_generate_qa_api_failure_returns_empty(self, mock_config):
        gen = QAGenerator(mock_config)
        with patch.object(gen.client.chat.completions, "create", side_effect=Exception("API error")):
            pairs = gen.generate_qa_from_section(
                chunks=["Some context text here."],
                topic_path=["Root"],
                source="test.md",
                num_pairs=1,
            )
        assert pairs == []
