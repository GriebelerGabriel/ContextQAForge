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
        gen.generated_questions.add("qual a quantidade diaria de azeite de oliva recomendada para reduzir risco cardiovascular")
        assert gen._is_duplicate("Qual a quantidade diaria de azeite de oliva recomendada para reduzir risco cardiovascular?") is True

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
        assert "REAL PERSON" in prompt

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

    def test_is_ground_truth_grounded_empty(self, mock_config):
        gen = QAGenerator(mock_config)
        assert gen._is_ground_truth_grounded("", ["some context"]) is False

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
        """Test two-step generation with mocked LLM calls."""
        gen = QAGenerator(mock_config)

        # Step 1 response: extract candidate facts (array)
        extract_response = MagicMock()
        extract_response.choices = [MagicMock()]
        extract_response.choices[0].message.content = json.dumps([
            {"extracted_fact": "A OMS recomenda que o consumo de sodio nao ultrapasse 2g por dia para adultos."},
            {"extracted_fact": "O limite de sal e de 5g por dia equivalente a 2g de sodio."},
            {"extracted_fact": "Reduzir o sodio ajuda a controlar a pressao arterial."},
        ])

        # Step 2 response: generate question
        question_response = MagicMock()
        question_response.choices = [MagicMock()]
        question_response.choices[0].message.content = json.dumps({
            "question": "Qual e a quantidade de sodio recomendada por dia para adultos segundo a OMS?",
            "ground_truth": "A OMS recomenda que o consumo de sodio nao ultrapasse 2g por dia para adultos.",
            "type": "single-hop",
            "difficulty": "easy",
        })

        with patch.object(
            gen.client.chat.completions, "create",
            side_effect=[extract_response, question_response],
        ):
            pairs = gen.generate_qa_from_section(
                chunks=["A OMS recomenda que o consumo de sodio nao ultrapasse 2g por dia para adultos."],
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

    def test_step1_extract_fact_success(self, mock_config):
        """Test Step 1 extraction returns a fact from multiple candidates."""
        gen = QAGenerator(mock_config)
        extract_response = MagicMock()
        extract_response.choices = [MagicMock()]
        extract_response.choices[0].message.content = json.dumps([
            {"extracted_fact": "The DASH diet helps control blood pressure."},
            {"extracted_fact": "Reducing sodium to less than 2g per day is beneficial."},
            {"extracted_fact": "Plant oils contain unsaturated fats that reduce LDL."},
        ])

        with patch.object(gen.client.chat.completions, "create", return_value=extract_response):
            fact = gen._step1_extract_fact(
                contexts=["The DASH diet helps control blood pressure."],
                topic_path=["Health", "Diet"],
                source="test.pdf",
                qa_type="single-hop",
            )

        assert fact is not None
        assert fact in [
            "The DASH diet helps control blood pressure.",
            "Reducing sodium to less than 2g per day is beneficial.",
            "Plant oils contain unsaturated fats that reduce LDL.",
        ]
        # Should be tracked in recent facts
        assert fact in gen._recent_facts

    def test_step1_extract_fact_failure(self, mock_config):
        """Test Step 1 returns None on API error."""
        gen = QAGenerator(mock_config)
        with patch.object(gen.client.chat.completions, "create", side_effect=Exception("API error")):
            fact = gen._step1_extract_fact(
                contexts=["Some context."],
                topic_path=["Root"],
                source="test.pdf",
                qa_type="single-hop",
            )
        assert fact is None

    def test_step2_generate_question_success(self, mock_config):
        """Test Step 2 generates a QAPair from an extracted fact."""
        gen = QAGenerator(mock_config)
        question_response = MagicMock()
        question_response.choices = [MagicMock()]
        question_response.choices[0].message.content = json.dumps({
            "question": "Quanto de sodio posso consumir por dia?",
            "ground_truth": "Voce pode consumir no maximo 2g de sodio por dia.",
            "type": "single-hop",
            "difficulty": "easy",
        })

        with patch.object(gen.client.chat.completions, "create", return_value=question_response):
            pair = gen._step2_generate_question(
                extracted_fact="The recommended daily sodium intake is 2g per day.",
                contexts=["The recommended daily sodium intake for adults is 2g per day."],
                topic_path=["Health", "Sodium"],
                source="test.pdf",
                qa_type="single-hop",
                difficulty="easy",
            )

        assert pair is not None
        assert pair.question == "Quanto de sodio posso consumir por dia?"
        assert pair.ground_truth == "Voce pode consumir no maximo 2g de sodio por dia."

    def test_step2_polarity_mismatch_true_false(self, mock_config):
        """Test Step 2 returns None when polarity doesn't match."""
        gen = QAGenerator(mock_config)
        # LLM returns FALSO when TRUE was requested
        question_response = MagicMock()
        question_response.choices = [MagicMock()]
        question_response.choices[0].message.content = json.dumps({
            "question": "Afirme: Consumir mais sodio e bom.",
            "ground_truth": "FALSO. Consumir mais sodio e prejudicial.",
            "type": "true-false",
            "difficulty": "easy",
        })

        with patch.object(gen.client.chat.completions, "create", return_value=question_response):
            pair = gen._step2_generate_question(
                extracted_fact="Reducing sodium is beneficial.",
                contexts=["Reducing sodium intake is beneficial."],
                topic_path=["Health"],
                source="test.pdf",
                qa_type="true-false",
                difficulty="easy",
                tf_polarity="true",
            )

        assert pair is None

    def test_extract_system_prompt_exists(self, mock_config):
        """Test that the extract system prompt is valid."""
        gen = QAGenerator(mock_config)
        prompt = gen._get_extract_system_prompt()
        assert "extract" in prompt.lower()
        assert "extracted_fact" in prompt

    def test_question_examples_have_required_fields(self, mock_config):
        """Test that all QUESTION_EXAMPLES have the required fields."""
        for qa_type, example in QAGenerator.QUESTION_EXAMPLES.items():
            assert "extracted_fact" in example, f"Missing extracted_fact in {qa_type}"
            assert "question" in example, f"Missing question in {qa_type}"
            assert "ground_truth" in example, f"Missing ground_truth in {qa_type}"

    def test_extract_examples_have_required_fields(self, mock_config):
        """Test that all EXTRACT_EXAMPLES have the required fields."""
        for qa_type, example in QAGenerator.EXTRACT_EXAMPLES.items():
            assert "contexts" in example, f"Missing contexts in {qa_type}"
            assert "extracted_fact" in example, f"Missing extracted_fact in {qa_type}"

    def test_step1_single_object_fallback(self, mock_config):
        """Test Step 1 handles single JSON object response (backward compat)."""
        gen = QAGenerator(mock_config)
        extract_response = MagicMock()
        extract_response.choices = [MagicMock()]
        extract_response.choices[0].message.content = json.dumps({
            "extracted_fact": "The DASH diet helps control blood pressure.",
        })

        with patch.object(gen.client.chat.completions, "create", return_value=extract_response):
            fact = gen._step1_extract_fact(
                contexts=["The DASH diet helps control blood pressure."],
                topic_path=["Health", "Diet"],
                source="test.pdf",
                qa_type="single-hop",
            )

        assert fact == "The DASH diet helps control blood pressure."

    def test_pick_freshest_fact_prefers_novel(self, mock_config):
        """Test that _pick_freshest_fact prefers facts different from recent ones."""
        gen = QAGenerator(mock_config)
        gen._recent_facts.append("Sodium intake should be limited to 2g per day for adults.")
        gen._recent_facts.append("The DASH diet is rich in fiber and low in salt.")

        candidates = [
            "Sodium intake should be limited to 2g per day for adults.",  # exact duplicate
            "The DASH diet helps control blood pressure.",                  # similar to recent
            "Replacing butter with olive oil reduces LDL cholesterol.",     # novel
        ]

        picked = gen._pick_freshest_fact(candidates)
        # The novel fact should be preferred (bottom 40% = 1 item, the novel one)
        assert picked == "Replacing butter with olive oil reduces LDL cholesterol."

    def test_pick_freshest_fact_no_recent(self, mock_config):
        """Test that _pick_freshest_fact works when no recent facts exist."""
        gen = QAGenerator(mock_config)
        candidates = ["fact A", "fact B", "fact C"]
        picked = gen._pick_freshest_fact(candidates)
        assert picked in candidates

    def test_recent_facts_tracked_across_extractions(self, mock_config):
        """Test that extracted facts are tracked and affect subsequent picks."""
        gen = QAGenerator(mock_config)

        # First extraction
        extract_response_1 = MagicMock()
        extract_response_1.choices = [MagicMock()]
        extract_response_1.choices[0].message.content = json.dumps([
            {"extracted_fact": "Sodium limit is 2g per day."},
            {"extracted_fact": "DASH diet lowers blood pressure."},
            {"extracted_fact": "Exercise helps the heart."},
        ])

        with patch.object(gen.client.chat.completions, "create", return_value=extract_response_1):
            fact1 = gen._step1_extract_fact(
                contexts=["Some context."], topic_path=["Root"], source="test.pdf", qa_type="single-hop",
            )

        assert len(gen._recent_facts) == 1

        # Second extraction — prompt should now include avoidance section
        extract_response_2 = MagicMock()
        extract_response_2.choices = [MagicMock()]
        extract_response_2.choices[0].message.content = json.dumps([
            {"extracted_fact": "Fiber reduces cholesterol."},
            {"extracted_fact": "Sugar should be limited."},
            {"extracted_fact": "Potassium helps blood pressure."},
        ])

        with patch.object(gen.client.chat.completions, "create", return_value=extract_response_2) as mock_create:
            fact2 = gen._step1_extract_fact(
                contexts=["Some other context."], topic_path=["Root"], source="test.pdf", qa_type="single-hop",
            )
            # Verify the prompt included the avoidance section
            call_args = mock_create.call_args
            prompt_text = call_args.kwargs.get("messages", call_args[1].get("messages", []))[1]["content"]
            assert "FACTS ALREADY USED" in prompt_text

        assert len(gen._recent_facts) == 2
