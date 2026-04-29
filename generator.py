"""QA generation module with section-based context.

Generates QA pairs grounded in document sections from the topic tree.
Uses the stronger model (gpt-4o) for QA, cheaper model was already used
for tree refinement.
"""

import json
import logging
import random
import re
import time
from typing import Dict, List, Literal, Optional, Set

from openai import OpenAI

logger = logging.getLogger(__name__)

from config import PipelineConfig
from models import DatasetEntry, QAPair


class QAGenerator:
    """Generates QA pairs from document sections."""

    # Question types with descriptions for diversity
    QA_TYPES: List[Dict] = [
        {
            "type": "single-hop",
            "description": "Answer directly found in one context chunk",
        },
        {
            "type": "multi-hop",
            "description": "Requires connecting information across multiple context chunks",
        },
        {
            "type": "inference",
            "description": "Requires reasoning or inferring from the provided contexts",
        },
        {
            "type": "paraphrase",
            "description": "Question uses different wording than the context",
        },
        {
            "type": "true-false",
            "description": "A statement that must be verified as true or false based on the contexts",
        },
    ]

    # Difficulty levels
    DIFFICULTIES: List[str] = ["easy", "medium", "hard"]

    # Few-shot examples for each type (patient/person perspective, in Portuguese)
    FEW_SHOT_EXAMPLES = {
        "single-hop": {
            "contexts": [
                "The recommended daily sodium intake for adults is 2g per day (equivalent to 5g of salt).",
            ],
            "question": "Quanto de sal posso consumir por dia sem passar do limite recomendado?",
            "ground_truth": "Você pode consumir no máximo 5g de sal por dia, o que equivale a 2g de sódio.",
            "difficulty": "easy",
        },
        "multi-hop": {
            "contexts": [
                "The DASH diet was developed for people with hypertension and helps control blood pressure.",
                "The DASH diet is rich in fiber, vitamins and minerals, and low in salt, sweets and saturated fat.",
            ],
            "question": "Se eu tenho pressão alta, qual dieta devo seguir e quais alimentos ela inclui?",
            "ground_truth": "Se você tem pressão alta, a dieta Dash foi feita especialmente para isso. Ela inclui alimentos ricos em fibras, vitaminas e minerais (como feijões, grãos integrais, frutas e legumes) e tem pouco sal, doces e gordura saturada.",
            "difficulty": "medium",
        },
        "inference": {
            "contexts": [
                "Reducing sodium intake to <2 g/day was more beneficial for blood pressure than consuming >2 g/day.",
                "Higher sodium intake was associated with higher risk of stroke and coronary heart disease.",
            ],
            "question": "Se eu consigo reduzir meu consumo de sódio para menos de 2g por dia, isso realmente faz diferença na minha pressão arterial?",
            "ground_truth": "Sim, faz muita diferença. Consumir menos de 2g de sódio por dia foi mais eficaz para reduzir a pressão arterial do que apenas diminuir mas continuar acima de 2g. Além disso, consumir mais sódio está ligado a maior risco de derrame e doenças cardíacas.",
            "difficulty": "hard",
        },
        "paraphrase": {
            "contexts": [
                "Replacing butter with plant oils containing predominantly unsaturated fat decreases LDL cholesterol concentrations.",
            ],
            "question": "É melhor cozinhar com óleo vegetal ou manteiga se eu quiser baixar meu colesterol?",
            "ground_truth": "É melhor usar óleos vegetais, porque eles têm gorduras insaturadas que ajudam a reduzir o colesterol ruim (LDL), ao contrário da manteiga.",
            "difficulty": "medium",
        },
        "true-false": {
            "contexts": [
                "Reducing sodium intake to <2 g/day was more beneficial for blood pressure than reducing sodium intake but still consuming >2 g/day.",
            ],
            "question": "Afirme: Consumir menos de 2g de sódio por dia é mais benéfico para a minha pressão arterial do que apenas reduzir mas continuar consumindo mais de 2g.",
            "ground_truth": "VERDADEIRO. Reduzir a ingestão de sódio para menos de 2 g/dia foi mais benéfico para a pressão arterial do que reduzir mas ainda consumir mais de 2 g/dia.",
            "difficulty": "easy",
        },
    }

    # Maximum number of past questions to check for duplicates
    DEDUP_WINDOW = 20

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.generated_questions: Set[str] = set()
        self.true_false_count = {"true": 0, "false": 0}

    def _get_system_prompt(self) -> str:
        """Generate system prompt based on config domain and language."""
        domain = self.config.document_domain
        language = self.config.language

        # Language instruction
        if language == "pt-BR":
            lang_instruction = (
                "Important: The input contexts may be in ENGLISH, "
                "but you MUST generate the question and answer in PORTUGUESE (Brazilian Portuguese)."
            )
        else:
            lang_instruction = (
                f"Generate the question and answer in the following language: {language}."
            )

        # Domain-specific perspective instructions
        if domain in ("general", "health", "nutrition", "medicine", "healthcare"):
            perspective_instruction = (
                "CRITICAL: You MUST write questions as if a REAL PERSON is asking for PERSONAL, PRACTICAL ADVICE.\n"
                "The person is talking to a doctor, nutritionist, or pharmacist — NOT reading a textbook.\n\n"
                "   FORBIDDEN (academic/impersonal):\n"
                '   - "Quais são as diretrizes para..." (asking about guidelines)\n'
                '   - "Explique o papel de..." (academic explanation)\n'
                '   - "Qual a importância de..." (essay question)\n'
                '   - "Por que é importante considerar..." (academic analysis)\n'
                '   - "Quais são os fatores de risco..." (textbook question)\n'
                '   - "O que são doenças não transmissíveis?" (definition)\n'
                '   - "Descreva as características de..." (describe)\n'
                "   \n"
                "   REQUIRED (real person asking about THEIR life):\n"
                '   - "O que devo comer para..." (what should I eat)\n'
                '   - "Posso comer ovos todos os dias se tenho colesterol alto?" (can I eat this)\n'
                '   - "Qual é melhor: arroz integral ou branco?" (which is better)\n'
                '   - "O que devo evitar se quiser baixar a pressão?" (what should I avoid)\n'
                '   - "Como posso preparar uma refeição saudável?" (how can I prepare)\n'
                '   - "Quanto de sal posso consumir por dia?" (how much can I consume)\n'
                '   - "É verdade que consumir menos de 2g de sódio por dia ajuda?" (is it true that...)\n'
                '   - "Se eu tiver pressão alta, qual dieta devo seguir?" (if I have X, what should I do)\n'
                '   - "Tenho hipertensão, posso beber vinho?" (I have X, can I do Y)\n'
                "   \n"
                "   PATTERNS TO USE:\n"
                '   - "Posso...?" / "Devo...?" / "Qual é o melhor...?" / "Quanto...?" / "É verdade que...?"\n'
                '   - "Se eu [situação], o que [acontece/devo fazer]?"\n'
                '   - "O que [posso/devo] fazer para [objetivo]?"\n'
                "   \n"
                '   Every question must sound like someone sitting in front of a health professional. '
                'Use first person: "eu", "posso", "devo", "meu", "minha". '
                'The answer should sound like that professional giving direct personal advice.'
            )
            domain_description = domain if domain != "general" else "their health and nutrition"
        else:
            perspective_instruction = (
                "Write questions from the perspective of a PRACTITIONER or USER in this domain, "
                "seeking practical, actionable information. Questions should be specific and grounded "
                "in the provided contexts, not academic or definitional."
            )
            domain_description = domain

        domain_extra = ""
        if domain and domain != "general":
            domain_extra = (
                f"\n8. Frame questions in the context of {domain.upper()}. "
                f"The documents are about {domain}, so questions should be relevant to this domain.\n"
            )

        return f"""You are an expert QA dataset generator. Your task is to create natural question-answer pairs where a person is asking questions about {domain_description}.

{lang_instruction}

Requirements:
1. Questions MUST be answerable using ONLY the provided contexts
2. GROUND TRUTH MUST BE EXTRACTED FROM THE CONTEXTS — every fact in the ground_truth must come directly from the context text
3. Do NOT use outside knowledge — if the contexts don't contain the answer, do not make one up
4. Ensure question diversity across types: single-hop, multi-hop, inference, paraphrase, true-false
5. Control difficulty: easy (direct lookup), medium (some reasoning), hard (complex inference)
6. {lang_instruction}
7. {perspective_instruction}
8. DOCUMENT-SPECIFIC QUESTIONS ONLY:
   - The question MUST require the SPECIFIC information in the provided contexts to answer
   - Do NOT ask questions that anyone could answer from general/popular knowledge
   - BAD: "Como proteger meu coração?" (too generic, common knowledge)
   - BAD: "O que é hipertensão?" (general medical knowledge)
   - GOOD: "Qual dieta específica foi desenvolvida para hipertensos e quais alimentos ela recomenda?" (requires the document)
   - GOOD: "Qual quantidade diária de azeite extravirgem ajuda a reduzir risco cardiovascular?" (requires specific detail from document)
   - Reference specific numbers, names, lists, recommendations, or recipes found in the contexts
   - The ground_truth MUST include specific details (numbers, names, quantities) that come from the contexts
9. NO DOCUMENT/INSTITUTION REFERENCES:
   - Do NOT mention the source document, institution, organization, or guideline name in questions or answers
   - BAD: "Segundo o Hospital de Clínicas de Porto Alegre..." (referencing institution)
   - BAD: "De acordo com a declaração da American Heart Association..." (referencing source)
   - BAD: "O guia alimentar brasileiro recomenda..." (referencing document)
   - GOOD: "Quais alimentos devo priorizar para proteger meu coração?" (general, patient perspective)
   - GOOD: "É verdade que reduzir o consumo de sal para menos de 2g por dia ajuda a controlar a pressão arterial?" (no source reference)
   - Present the information as general advice, not as a citation from a specific document{domain_extra}

CRITICAL: The ground_truth is your answer. It MUST be derived entirely from the provided contexts.

ANSWER STYLE RULES:
- Do NOT copy-paste or regurgitate the context text. REWRITE it in a natural, conversational tone.
- BAD: "Você precisa de 1 unidade de cebola, 2 dentes de alho, 1 punhado de mostarda..." (just listing ingredients)
- GOOD: "Para preparar esse molho, você vai precisar de cebola, alho e sementes de mostarda como base, além de vinagre e água. O processo é simples: bata tudo no liquidificador e deixe descansar por dois dias antes de coar." (natural summary)
- Keep specific numbers and key facts (dosages, limits, durations) but express them naturally.
- If the context has a recipe, summarize the steps conversationally — don't list every ingredient like a cookbook.
- If the context has recommendations, explain them as advice, not as a direct quote.
- The answer should read as if a knowledgeable person is explaining it, not as if reading the document aloud.

Do NOT answer from general knowledge — every claim must come from the contexts.

Output must be valid JSON with exactly these fields:
{{{{
    "question": "the question in {language}",
    "ground_truth": "the answer in {language}",
    "type": "single-hop|multi-hop|inference|paraphrase|true-false",
    "difficulty": "easy|medium|hard"
}}}}

For type "true-false":
- The question field should contain an AFFIRMATION/STATEMENT (not a question) that can be verified as true or false based ONLY on the contexts.
- Start the question with "Afirme:" (in Portuguese) or "Assert:" followed by the statement.
- The ground_truth MUST start with "VERDADEIRO." or "FALSO." followed by the evidence from the contexts that proves or disproves the statement.
- Mix true and false statements. For false statements, subtly alter a detail (number, fact, recommendation) from the context so it becomes incorrect.
- Examples of false statements: change a quantity (2g → 5g), swap a recommendation, reverse a finding."""

    def _is_generic_question(self, question: str) -> bool:
        """Heuristic to detect generic/popular-knowledge questions."""
        if len(question) < 20:
            return True

        question_lower = question.lower()

        generic_patterns = [
            r"^what is",
            r"^who is",
            r"^when did",
            r"^where is",
            r"^how to",
            r"^tell me about",
            r"^explain",
            r"^describe",
        ]

        for pattern in generic_patterns:
            if re.match(pattern, question_lower):
                words_after = len(question_lower.split()) - 2
                if words_after < 5:
                    return True

        generic_phrases = [
            "como proteger",
            "o que sao",
            "o que são",
            "como prevenir",
            "como melhorar",
            "como evitar",
            "o que causa",
            "qual a importancia",
            "qual a importância",
            "o que fazer para",
            "como posso mudar minha dieta",
            "como posso integrar exercicios",
            "como posso integrar exercícios",
        ]

        for phrase in generic_phrases:
            if question_lower.startswith(phrase):
                if len(question_lower) < 50:
                    return True

        return False

    def _is_duplicate(self, question: str) -> bool:
        """Check if question is too similar to existing ones."""
        question_normalized = question.lower().strip().rstrip("?")

        if question_normalized in self.generated_questions:
            return True

        # Only check against recent questions, and only block near-identical ones
        recent = list(self.generated_questions)[-self.DEDUP_WINDOW:]
        for existing in recent:
            # Skip if very different length (can't be near-identical)
            if abs(len(existing) - len(question_normalized)) > len(question_normalized) * 0.3:
                continue
            existing_words = set(existing.split())
            question_words = set(question_normalized.split())
            overlap = len(question_words & existing_words) / max(len(question_words), 1)
            if overlap > 0.90:
                return True

        return False

    def _is_ground_truth_grounded(self, ground_truth: str, contexts: List[str]) -> bool:
        """Check that ground truth is grounded in the provided contexts.

        Uses heuristic matching: shared numbers with units, entity overlap,
        or word overlap >= 15%.
        """
        if not ground_truth.strip():
            return False

        import re as _re
        context_text = " ".join(contexts)

        # Check specific quantities from contexts
        ctx_numbers = set(_re.findall(
            r'\b\d+[.,]?\d*\s*(?:mg|g|ml|mmol|%|mcg|kg|lb|cal|kcal|cm|mm)\b',
            context_text.lower()
        ))
        gt_numbers = set(_re.findall(
            r'\b\d+[.,]?\d*\s*(?:mg|g|ml|mmol|%|mcg|kg|lb|cal|kcal|cm|mm)\b',
            ground_truth.lower()
        ))
        if gt_numbers and gt_numbers & ctx_numbers:
            return True

        # Check named entities
        ctx_entities = set(_re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', context_text))
        gt_entities = set(_re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', ground_truth))
        if len(gt_entities & ctx_entities) >= 2:
            return True

        # Fallback: word overlap
        gt_words = set(ground_truth.lower().split())
        context_words = set(context_text.lower().split())
        overlap = len(gt_words & context_words) / max(len(gt_words), 1)
        return overlap >= self.config.grounding_threshold

    def generate_qa_from_section(
        self,
        chunks: List[str],
        topic_path: List[str],
        source: str,
        num_pairs: int = 1,
        full_content: str = "",
        related_chunks: Optional[List[str]] = None,
    ) -> List[QAPair]:
        """Generate QA pairs from a section's chunks.

        Args:
            chunks: List of text chunks from a leaf node
            topic_path: Path in the topic tree (e.g., ["Doc Topics", "Sodium", "Recommended Levels"])
            source: Source document filename
            num_pairs: Number of QA pairs to generate
            full_content: The full segment text (entire content block from slicing)
            related_chunks: Related content from other documents on similar topics

        Returns:
            List of QAPair objects (may be fewer than num_pairs if filtered)
        """
        if not chunks:
            return []

        # Build contexts: primary chunks for grounding + full content + related
        if len(chunks) <= 3:
            contexts = chunks
        else:
            contexts = random.sample(chunks, min(3, len(chunks)))

        # Store full content and related chunks for prompt enrichment
        self._current_full_content = full_content
        self._current_related = related_chunks or []

        if num_pairs == 1:
            pair = self._generate_single(contexts, topic_path, source)
            return [pair] if pair else []
        else:
            return self._generate_batch(contexts, topic_path, source, num_pairs)

    def _generate_single(
        self,
        contexts: List[str],
        topic_path: List[str],
        source: str,
    ) -> Optional[QAPair]:
        """Generate a single QA pair."""
        qa_type = random.choice([t["type"] for t in self.QA_TYPES])
        difficulty = random.choice(self.DIFFICULTIES)

        # For true-false: enforce 50/50 balance by forcing the polarity
        tf_instruction = ""
        if qa_type == "true-false" and self.config.balance_true_false:
            if self.true_false_count["true"] <= self.true_false_count["false"]:
                tf_instruction = "\n\nIMPORTANT: This assertion MUST be TRUE (VERDADEIRO). Generate a statement that is factually correct according to the contexts."
                self.true_false_count["true"] += 1
            else:
                tf_instruction = "\n\nIMPORTANT: This assertion MUST be FALSE (FALSO). Subtly alter a detail from the context (number, fact, recommendation) so the statement becomes incorrect."
                self.true_false_count["false"] += 1
        topic_str = " -> ".join(topic_path)
        example = self.FEW_SHOT_EXAMPLES.get(qa_type, self.FEW_SHOT_EXAMPLES["single-hop"])

        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        # Build supplementary context from full content and related chunks
        supplement = ""
        full_content = getattr(self, "_current_full_content", "")
        related = getattr(self, "_current_related", [])

        if full_content and full_content not in context_text:
            supplement += f"\n=== FULL SECTION TEXT ===\n{full_content}\n"

        if related:
            related_text = "\n\n".join(
                f"[Related - {i+1}]: {chunk}" for i, chunk in enumerate(related)
            )
            supplement += f"\n=== RELATED CONTENT FROM OTHER DOCUMENTS ===\n{related_text}\n"

        # All text the LLM sees (for grounding validation)
        all_source_text = " ".join(contexts)
        if full_content:
            all_source_text += " " + full_content
        for r in related:
            all_source_text += " " + r

        prompt = f"""Generate a QA pair based on the following contexts.

=== TOPIC GUIDANCE ===
Suggested topic angle: {topic_str}
Use this topic as inspiration for the question's angle, but the contexts are the source of truth.
Source document: {source}

=== CONTEXTS (SOURCE OF TRUTH) ===
{context_text}
{supplement}
=== EXAMPLE ({qa_type}, {difficulty}) ===
Contexts:
"""
        prompt += "\n\n".join(example["contexts"])
        prompt += f"""

Question: {example["question"]}
Ground Truth: {example["ground_truth"]}
Type: {qa_type}
Difficulty: {example["difficulty"]}

=== YOUR TASK ===
Generate a NEW {qa_type} question at {difficulty} difficulty level.

CRITICAL REQUIREMENTS:
1. The question MUST be written as a REAL PERSON asking for personal advice (use "eu", "posso", "devo")
2. The question MUST require the SPECIFIC information in the contexts to answer
3. The contexts are the source of truth — ground answers only in what the contexts say
4. Do NOT ask questions answerable from general/popular knowledge
5. Ask about specific details: numbers, names, recommendations, lists from the text
5. The ground_truth MUST cite or closely paraphrase information from the contexts

Output ONLY the JSON object, no additional text:
"""

        for attempt in range(self.config.max_retries):
            if attempt > 0:
                time.sleep(min(2 ** attempt, 16))

            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=1000,
                )

                content = response.choices[0].message.content
                data = self._parse_json_response(content)
                if not data:
                    continue

                question = data.get("question", "").strip()
                ground_truth = data.get("ground_truth", "").strip()

                if not question or not ground_truth:
                    continue
                if self._is_duplicate(question):
                    logger.debug(f"[Rejected] Duplicate question: {question[:60]}...")
                    continue

                self.generated_questions.add(question.lower().strip().rstrip("?"))

                return QAPair(
                    question=question,
                    ground_truth=ground_truth,
                    contexts=contexts,
                    metadata={
                        "type": data.get("type", qa_type),
                        "difficulty": data.get("difficulty", difficulty),
                        "topic_path": topic_path,
                        "source": source,
                    },
                )

            except Exception as e:
                logger.warning(f"[QA rejected] API error: {e}")
                continue

        return None

    def _generate_batch(
        self,
        contexts: List[str],
        topic_path: List[str],
        source: str,
        num_pairs: int,
    ) -> List[QAPair]:
        """Generate multiple QA pairs in a single LLM call."""
        all_types = [t["type"] for t in self.QA_TYPES]
        qa_types = [all_types[i % len(all_types)] for i in range(num_pairs)]
        difficulties = [self.DIFFICULTIES[i % len(self.DIFFICULTIES)] for i in range(num_pairs)]
        topic_str = " -> ".join(topic_path)

        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        # Build supplementary context from full content and related chunks
        supplement = ""
        full_content = getattr(self, "_current_full_content", "")
        related = getattr(self, "_current_related", [])

        if full_content and full_content not in context_text:
            supplement += f"\n=== FULL SECTION TEXT ===\n{full_content}\n"

        if related:
            related_text = "\n\n".join(
                f"[Related - {i+1}]: {chunk}" for i, chunk in enumerate(related)
            )
            supplement += f"\n=== RELATED CONTENT FROM OTHER DOCUMENTS ===\n{related_text}\n"

        # All text the LLM sees (for grounding validation)
        all_source_text = " ".join(contexts)
        if full_content:
            all_source_text += " " + full_content
        for r in related:
            all_source_text += " " + r

        # Build per-pair type/difficulty lines, with true/false polarity instructions
        type_difficulty_lines = []
        for i in range(num_pairs):
            line = f"{i+1}. Type: {qa_types[i]}, Difficulty: {difficulties[i]}"
            if qa_types[i] == "true-false" and self.config.balance_true_false:
                if self.true_false_count["true"] <= self.true_false_count["false"]:
                    line += " — This assertion MUST be TRUE (VERDADEIRO)."
                    self.true_false_count["true"] += 1
                else:
                    line += " — This assertion MUST be FALSE (FALSO). Subtly alter a detail."
                    self.true_false_count["false"] += 1
            type_difficulty_lines.append(line)
        type_difficulty_str = "\n".join(type_difficulty_lines)

        example = self.FEW_SHOT_EXAMPLES[qa_types[0]]

        prompt = f"""Generate {num_pairs} distinct QA pairs based on the following contexts.

=== TOPIC GUIDANCE ===
Suggested topic angle: {topic_str}
Use this topic as inspiration, but the contexts are the source of truth.
Source document: {source}

=== CONTEXTS (SOURCE OF TRUTH) ===
{context_text}
{supplement}
=== TYPES AND DIFFICULTIES FOR EACH PAIR ===
{type_difficulty_str}

=== EXAMPLE ({qa_types[0]}, {difficulties[0]}) ===
Contexts:
"""
        prompt += "\n\n".join(example["contexts"])
        prompt += f"""

Question: {example["question"]}
Ground Truth: {example["ground_truth"]}
Type: {qa_types[0]}
Difficulty: {example["difficulty"]}

=== YOUR TASK ===
Generate {num_pairs} NEW QA pairs, one for each type/difficulty combination.
{"For true-false entries, follow the TRUE/FALSE polarity specified above for each pair." if any(t == "true-false" for t in qa_types) else ""}

CRITICAL REQUIREMENTS:
1. Every question MUST be written as a REAL PERSON asking for personal advice (use "eu", "posso", "devo")
2. Every question MUST require SPECIFIC information from the contexts
3. Ground answers only in what the contexts say
4. Do NOT ask questions answerable from general/popular knowledge
5. Each ground_truth MUST cite or closely paraphrase the contexts
6. Questions must be DISTINCT (no duplicates or near-duplicates)

Output ONLY a JSON array with {num_pairs} objects:
[
  {{"question": "...", "ground_truth": "...", "type": "...", "difficulty": "..."}},
  ...
]
"""

        for attempt in range(self.config.max_retries):
            if attempt > 0:
                time.sleep(min(2 ** attempt, 16))

            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=2000,
                )

                content = response.choices[0].message.content
                data_list = self._parse_json_response(content)

                if not isinstance(data_list, list):
                    continue

                accepted: List[QAPair] = []
                for i, data in enumerate(data_list[:num_pairs]):
                    question = data.get("question", "").strip()
                    ground_truth = data.get("ground_truth", "").strip()

                    if not question or not ground_truth:
                        continue
                    if self._is_duplicate(question):
                        continue

                    self.generated_questions.add(question.lower().strip().rstrip("?"))
                    accepted.append(QAPair(
                        question=question,
                        ground_truth=ground_truth,
                        contexts=contexts,
                        metadata={
                            "type": data.get("type", qa_types[i % len(qa_types)]),
                            "difficulty": data.get("difficulty", difficulties[i % len(difficulties)]),
                            "topic_path": topic_path,
                            "source": source,
                        },
                    ))

                if accepted:
                    logger.info(f"[Batch] Accepted {len(accepted)}/{len(data_list)} pairs from '{source}'")
                return accepted

            except Exception as e:
                logger.warning(f"[Batch] API error: {e}")
                continue

        return []

    @staticmethod
    def _parse_json_response(content: str):
        """Parse JSON from LLM response, handling markdown code blocks."""
        raw = content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return None

    def to_dataset_entry(self, qa_pair: QAPair) -> DatasetEntry:
        """Convert QAPair to RAGAS evaluation format."""
        return DatasetEntry(
            question=qa_pair.question,
            answer=qa_pair.ground_truth,
            ground_truth=qa_pair.ground_truth,
            contexts=qa_pair.contexts,
            source=qa_pair.metadata.get("source", ""),
            metadata=qa_pair.metadata,
        )
