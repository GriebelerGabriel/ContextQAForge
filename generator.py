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
    ]

    # Difficulty levels
    DIFFICULTIES: List[str] = ["easy", "medium", "hard"]

    # Few-shot examples for each type (in Portuguese)
    FEW_SHOT_EXAMPLES = {
        "single-hop": {
            "contexts": [
                "The Python programming language was created by Guido van Rossum and first released in 1991.",
            ],
            "question": "Quem criou o Python e quando ele foi lançado pela primeira vez?",
            "ground_truth": "O Python foi criado por Guido van Rossum e lançado pela primeira vez em 1991.",
            "difficulty": "easy",
        },
        "multi-hop": {
            "contexts": [
                "Python was created by Guido van Rossum and first released in 1991.",
                "Guido van Rossum worked at Google from 2005 to 2012.",
            ],
            "question": "Em qual empresa o criador do Python trabalhou durante os anos 2000?",
            "ground_truth": "Guido van Rossum, o criador do Python, trabalhou na Google de 2005 a 2012.",
            "difficulty": "medium",
        },
        "inference": {
            "contexts": [
                "Python 2.7 was released in 2010 and reached end-of-life in January 2020.",
                "Python 3.0 was released in December 2008 and is actively maintained.",
            ],
            "question": "Considerando a linha do tempo, por que a maioria dos desenvolvedores recomendaria Python 3 em vez de Python 2 para novos projetos iniciados em 2021?",
            "ground_truth": "O Python 2.7 chegou ao fim da vida útil em janeiro de 2020, o que significa que não recebe mais atualizações de segurança ou correções de bugs. O Python 3.0, lançado em 2008, é mantido ativamente. Portanto, para um novo projeto em 2021, o Python 3 seria a escolha segura e suportada.",
            "difficulty": "hard",
        },
        "paraphrase": {
            "contexts": [
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            ],
            "question": "Como os sistemas de IA adquirem conhecimento a partir de informações sem programação explícita?",
            "ground_truth": "Através do aprendizado de máquina (machine learning), um subconjunto da inteligência artificial que permite que sistemas aprendam com dados.",
            "difficulty": "medium",
        },
    }

    # Maximum number of past questions to check for duplicates
    DEDUP_WINDOW = 50

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.generated_questions: Set[str] = set()

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
                "CRITICAL: You MUST write questions as if a REAL PATIENT is asking for PRACTICAL PERSONAL ADVICE:\n\n"
                "   FORBIDDEN (academic/professional perspective):\n"
                '   - "Como os nutricionistas recomendam..." (asking about professionals)\n'
                '   - "Quais são as diretrizes para..." (asking about guidelines)\n'
                '   - "Defina proteínas" (definition for memorization)\n'
                '   - "O que é glicose?" (vocabulary test)\n'
                '   - "Liste os 3 tipos de..." (list memorization)\n'
                '   - "Explique o papel de..." (academic explanation)\n'
                "   \n"
                "   REQUIRED (patient perspective - personal, actionable):\n"
                '   - "O que devo comer para..." (what should I eat)\n'
                '   - "Posso comer ovos todos os dias se tenho colesterol alto?" (can I eat this)\n'
                '   - "Qual é melhor para mim: arroz integral ou branco?" (which is better for me)\n'
                '   - "O que devo evitar antes de uma cirurgia?" (what should I avoid)\n'
                '   - "Como posso preparar uma refeição saudável?" (how can I prepare)\n'
                '   - "Quanto de sal posso consumir por dia?" (how much can I consume)\n'
                "   \n"
                '   The patient asks about THEIR OWN choices, actions, and health. '
                'Use "eu", "posso", "devo", "qual", "como" - first person, seeking guidance for themselves.'
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
4. Ensure question diversity across types: single-hop, multi-hop, inference, paraphrase
5. Control difficulty: easy (direct lookup), medium (some reasoning), hard (complex inference)
6. {lang_instruction}
7. {perspective_instruction}
8. DOCUMENT-SPECIFIC QUESTIONS ONLY:
   - The question MUST require the SPECIFIC information in the provided contexts to answer
   - Do NOT ask questions that anyone could answer from general/popular knowledge
   - BAD: "Como proteger meu coração?" (too generic, common knowledge)
   - BAD: "O que é hipertensão?" (general medical knowledge)
   - GOOD: "Qual dieta específica foi desenvolvida para hipertensos e quais alimentos ela recomenda?" (requires the document)
   - GOOD: "Segundo o manual, qual quantidade diária de azeite extravirgem ajuda a reduzir risco cardiovascular?" (requires specific detail from document)
   - Reference specific numbers, names, lists, recommendations, or recipes found in the contexts
   - The ground_truth MUST include specific details (numbers, names, quantities) that come from the contexts{domain_extra}

CRITICAL: The ground_truth is your answer. It MUST be derived entirely from the provided contexts.
Include specific facts, numbers, or quotes from the context text. If the contexts say "2g of sodium per day",
your ground_truth should mention "2g of sodium per day". Do NOT answer from general knowledge.

Output must be valid JSON with exactly these fields:
{{{{
    "question": "the question in {language}",
    "ground_truth": "the answer in {language}",
    "type": "single-hop|multi-hop|inference|paraphrase",
    "difficulty": "easy|medium|hard"
}}}}"""

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

        recent = list(self.generated_questions)[-self.DEDUP_WINDOW:]
        question_words = set(question_normalized.split())
        for existing in recent:
            existing_words = set(existing.split())
            overlap = len(question_words & existing_words) / max(len(question_words), 1)
            if overlap > 0.65:
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
    ) -> List[QAPair]:
        """Generate QA pairs from a section's chunks.

        Args:
            chunks: List of text chunks from a leaf node
            topic_path: Path in the topic tree (e.g., ["Doc Topics", "Sodium", "Recommended Levels"])
            source: Source document filename
            num_pairs: Number of QA pairs to generate

        Returns:
            List of QAPair objects (may be fewer than num_pairs if filtered)
        """
        if not chunks:
            return []

        # Select 1-3 chunks as context
        if len(chunks) <= 3:
            contexts = chunks
        else:
            # Pick up to 3 diverse chunks
            contexts = random.sample(chunks, min(3, len(chunks)))

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
        topic_str = " -> ".join(topic_path)
        example = self.FEW_SHOT_EXAMPLES.get(qa_type, self.FEW_SHOT_EXAMPLES["single-hop"])

        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""Generate a QA pair based on the following contexts.

=== TOPIC GUIDANCE ===
Suggested topic angle: {topic_str}
Use this topic as inspiration for the question's angle, but the contexts are the source of truth.
Source document: {source}

=== CONTEXTS (SOURCE OF TRUTH) ===
{context_text}

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
1. The question MUST require the SPECIFIC information in the contexts to answer
2. The contexts are the source of truth — ground answers only in what the contexts say
3. Do NOT ask questions answerable from general/popular knowledge
4. Ask about specific details: numbers, names, recommendations, lists from the text
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
                if self._is_generic_question(question):
                    logger.debug(f"[Rejected] Generic question: {question[:60]}...")
                    continue
                if self._is_duplicate(question):
                    logger.debug(f"[Rejected] Duplicate question: {question[:60]}...")
                    continue
                if not self._is_ground_truth_grounded(ground_truth, contexts):
                    logger.debug(f"[Rejected] Ungrounded answer: {ground_truth[:60]}...")
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

        type_difficulty_str = "\n".join([
            f"{i+1}. Type: {qa_types[i]}, Difficulty: {difficulties[i]}"
            for i in range(num_pairs)
        ])

        example = self.FEW_SHOT_EXAMPLES[qa_types[0]]

        prompt = f"""Generate {num_pairs} distinct QA pairs based on the following contexts.

=== TOPIC GUIDANCE ===
Suggested topic angle: {topic_str}
Use this topic as inspiration, but the contexts are the source of truth.
Source document: {source}

=== CONTEXTS (SOURCE OF TRUTH) ===
{context_text}

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

CRITICAL REQUIREMENTS:
1. Every question MUST require SPECIFIC information from the contexts
2. Ground answers only in what the contexts say
3. Do NOT ask questions answerable from general/popular knowledge
4. Each ground_truth MUST cite or closely paraphrase the contexts
5. Questions must be DISTINCT (no duplicates or near-duplicates)

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
                    if self._is_generic_question(question):
                        continue
                    if self._is_duplicate(question):
                        continue
                    if not self._is_ground_truth_grounded(ground_truth, contexts):
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
        )
