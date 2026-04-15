"""Pluto-style QA generation module with topic tree support."""

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
from topic_tree import TopicTree


class QAGenerator:
    """Generates QA pairs using Pluto-style structured prompting."""

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

    def __init__(self, config: PipelineConfig, embedder=None):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.generated_questions: Set[str] = set()
        self.topic_tree: Optional[TopicTree] = None
        self.embedder = embedder  # For semantic grounding check

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

    def set_topic_tree(self, tree: TopicTree) -> None:
        """Set an externally built topic tree."""
        self.topic_tree = tree

    def _build_prompt(
        self,
        contexts: List[str],
        qa_type: str,
        difficulty: str,
        topic_path: Optional[List[str]] = None,
    ) -> str:
        """Build a structured prompt for QA generation with topic guidance."""
        # Get example for this type
        example = self.FEW_SHOT_EXAMPLES.get(qa_type, self.FEW_SHOT_EXAMPLES["single-hop"])

        # Build context section
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        # Build topic string - always use the full path if available
        if topic_path:
            topic_str = " -> ".join(topic_path)
        else:
            topic_str = "General document content"

        prompt = f"""Generate a QA pair based on the following contexts.

=== TOPIC GUIDANCE ===
Suggested topic angle: {topic_str}
Use this topic as inspiration for the question's angle, but the contexts are the source of truth.

=== CONTEXTS (SOURCE OF TRUTH) ===
{context_text}

=== EXAMPLE ({qa_type}, {difficulty}) ===
Contexts:
"""
        prompt += "\n\n".join([f"{ctx}" for ctx in example["contexts"]])
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
4. Ask about specific details: numbers, names, recommendations, lists, or procedures from the text
5. The ground_truth MUST cite or closely paraphrase information from the contexts
6. Use the topic "{topic_str}" as a suggested angle, but prioritize what the contexts actually contain

Output ONLY the JSON object, no additional text:
"""
        return prompt

    def _build_batch_prompt(
        self,
        contexts: List[str],
        num_pairs: int,
        qa_types: List[str],
        difficulties: List[str],
        topic_path: Optional[List[str]] = None,
    ) -> str:
        """Build a structured prompt for batch QA generation with topic guidance."""
        # Build context section
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        # Build topic string
        if topic_path:
            topic_str = " -> ".join(topic_path)
        else:
            topic_str = "General document content"

        # Build type/difficulty assignments
        type_difficulty_str = "\n".join([
            f"{i+1}. Type: {qa_types[i % len(qa_types)]}, Difficulty: {difficulties[i % len(difficulties)]}"
            for i in range(num_pairs)
        ])

        prompt = f"""Generate {num_pairs} distinct QA pairs based on the following contexts.

=== TOPIC GUIDANCE ===
Suggested topic angle: {topic_str}
Use this topic as inspiration for the questions' angles, but the contexts are the source of truth.

=== CONTEXTS (SOURCE OF TRUTH) ===
{context_text}

=== TYPES AND DIFFICULTIES FOR EACH PAIR ===
{type_difficulty_str}

=== EXAMPLE ({qa_types[0]}, {difficulties[0]}) ===
Contexts:
"""
        prompt += "\n\n".join([f"{ctx}" for ctx in self.FEW_SHOT_EXAMPLES[qa_types[0]]["contexts"]])
        prompt += f"""

Question: {self.FEW_SHOT_EXAMPLES[qa_types[0]]["question"]}
Ground Truth: {self.FEW_SHOT_EXAMPLES[qa_types[0]]["ground_truth"]}
Type: {qa_types[0]}
Difficulty: {self.FEW_SHOT_EXAMPLES[qa_types[0]]["difficulty"]}

=== YOUR TASK ===
Generate {num_pairs} NEW QA pairs, one for each type/difficulty combination listed above.

CRITICAL REQUIREMENTS:
1. Every question MUST require the SPECIFIC information in the contexts to answer
2. The contexts are the source of truth — ground answers only in what the contexts say
3. Do NOT ask questions answerable from general/popular knowledge
4. Ask about specific details: numbers, names, recommendations, lists, or procedures from the text
5. Each ground_truth MUST cite or closely paraphrase information from the contexts
6. Questions must be DISTINCT from each other (no duplicates or near-duplicates)
7. Use the topic "{topic_str}" as a suggested angle, but prioritize what the contexts actually contain

Output ONLY a JSON array with {num_pairs} objects, no additional text:
[
  {{"question": "...", "ground_truth": "...", "type": "...", "difficulty": "..."}},
  ...
]
"""
        return prompt

    def _is_generic_question(self, question: str) -> bool:
        """Heuristic to detect generic/popular-knowledge questions."""
        # Too short
        if len(question) < 20:
            return True

        question_lower = question.lower()

        # Generic starters that suggest popular knowledge
        # Note: Removed Portuguese patterns (o que é, quando, onde) as they block valid questions
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
                if words_after < 5:  # Raised from 2 to 5
                    return True

        # Generic question patterns — too broad, answerable from popular knowledge
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
                # Only reject if the question is short (no specific detail added)
                if len(question_lower) < 50:  # Lowered from 80 to 50
                    return True

        return False

    def _is_duplicate(self, question: str) -> bool:
        """Check if question is too similar to existing ones (bounded window)."""
        question_normalized = question.lower().strip().rstrip("?")

        if question_normalized in self.generated_questions:
            return True

        # Only check against a bounded window of recent questions
        recent = list(self.generated_questions)[-self.DEDUP_WINDOW:]
        question_words = set(question_normalized.split())
        for existing in recent:
            existing_words = set(existing.split())
            overlap = len(question_words & existing_words) / max(len(question_words), 1)
            if overlap > 0.65:  # Tightened from 0.8 to catch near-duplicates
                return True

        return False

    def _is_ground_truth_grounded(self, ground_truth: str, contexts: List[str], qa_type: str) -> bool:
        """Check that ground truth is grounded in the provided contexts.

        Uses semantic similarity via embeddings (language-independent).
        Falls back to entity/number matching if no embedder is available.
        """
        if not ground_truth.strip():
            return False

        # Strategy 1: Semantic check using embeddings (works cross-language)
        if self.embedder is not None:
            try:
                import numpy as np
                gt_embedding = self.embedder.embed_query(ground_truth)
                for ctx in contexts:
                    ctx_embedding = self.embedder.embed_query(ctx)
                    similarity = float(np.dot(gt_embedding, ctx_embedding) / (
                        np.linalg.norm(gt_embedding) * np.linalg.norm(ctx_embedding) + 1e-8
                    ))
                    if similarity >= 0.5:
                        return True
                return False
            except Exception:
                pass  # Fall through to entity check

        # Strategy 2: Entity/number extraction (no API calls, language-independent)
        # Check if the ground truth contains specific numbers or entities from the contexts
        import re as _re
        context_text = " ".join(contexts)

        # Extract numbers (with units) from contexts
        ctx_numbers = set(_re.findall(r'\b\d+[.,]?\d*\s*(?:mg|g|ml|mmol|%|mcg|kg|lb|cal|kcal|cm|mm)\b', context_text.lower()))
        gt_numbers = set(_re.findall(r'\b\d+[.,]?\d*\s*(?:mg|g|ml|mmol|%|mcg|kg|lb|cal|kcal|cm|mm)\b', ground_truth.lower()))

        # If ground truth mentions specific quantities from the contexts, it's grounded
        if gt_numbers and gt_numbers & ctx_numbers:
            return True

        # Extract capitalized entities (names, acronyms) from contexts
        ctx_entities = set(_re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', context_text))
        gt_entities = set(_re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', ground_truth))

        # If ground truth mentions named entities from the contexts, it's grounded
        significant_entities = gt_entities & ctx_entities
        if len(significant_entities) >= 2:
            return True

        # Fallback: loose word overlap (lowered to 15%)
        gt_words = set(ground_truth.lower().split())
        context_words = set(context_text.lower().split())
        overlap = len(gt_words & context_words) / max(len(gt_words), 1)
        return overlap >= 0.15

    def generate_qa(
        self,
        contexts: List[str],
        qa_type: Optional[Literal["single-hop", "multi-hop", "inference", "paraphrase"]] = None,
        difficulty: Optional[Literal["easy", "medium", "hard"]] = None,
        iteration: int = 0,
        topic_path: Optional[List[str]] = None,
    ) -> Optional[QAPair]:
        """
        Generate a QA pair from contexts with topic tree guidance.

        Args:
            contexts: List of context strings
            qa_type: Type of question to generate (random if None)
            difficulty: Difficulty level (random if None)
            iteration: Current iteration for logging
            topic_path: Topic path from topic tree (if available)

        Returns:
            QAPair or None if generation failed
        """
        # Randomly select type and difficulty if not specified
        if qa_type is None:
            qa_type = random.choice([t["type"] for t in self.QA_TYPES])
        if difficulty is None:
            difficulty = random.choice(self.DIFFICULTIES)

        # Retry loop with exponential backoff
        for attempt in range(self.config.max_retries):
            if attempt > 0:
                time.sleep(min(2 ** attempt, 16))  # Cap at 16 seconds

            try:
                prompt = self._build_prompt(contexts, qa_type, difficulty, topic_path)

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

                # Parse JSON
                try:
                    # Try to extract JSON from markdown code block
                    if "```json" in content:
                        json_str = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        json_str = content.split("```")[1].split("```")[0]
                    else:
                        json_str = content

                    data = json.loads(json_str.strip())

                    question = data.get("question", "").strip()
                    ground_truth = data.get("ground_truth", "").strip()
                    result_type = data.get("type", qa_type)
                    result_difficulty = data.get("difficulty", difficulty)

                    # Basic validation — both fields must exist
                    if not question or not ground_truth:
                        continue

                    # Check that the answer is grounded in the contexts
                    if not self._is_ground_truth_grounded(ground_truth, contexts, result_type):
                        continue

                    # Store for duplicate detection
                    self.generated_questions.add(question.lower().strip().rstrip("?"))

                    metadata = {
                        "type": result_type,
                        "difficulty": result_difficulty,
                        "attempt": attempt + 1,
                        "iteration": iteration,
                    }

                    if topic_path:
                        metadata["topic_path"] = topic_path

                    return QAPair(
                        question=question,
                        ground_truth=ground_truth,
                        contexts=contexts,
                        metadata=metadata,
                    )

                except json.JSONDecodeError:
                    logger.warning(f"[QA rejected] JSON parse error: {content[:100]}...")
                    continue

            except Exception as e:
                logger.warning(f"[QA rejected] API error: {e}")
                continue

        # Failed after all retries
        return None

    def generate_qa_batch(
        self,
        contexts: List[str],
        num_pairs: int = 3,
        iteration: int = 0,
        topic_path: Optional[List[str]] = None,
    ) -> List[QAPair]:
        """
        Generate multiple QA pairs in a single LLM call.

        Args:
            contexts: List of context strings
            num_pairs: Number of QA pairs to generate
            iteration: Current iteration for logging
            topic_path: Topic path from topic tree (if available)

        Returns:
            List of accepted QAPair objects (may be fewer than num_pairs if filtered)
        """
        # Select diverse types and difficulties for the batch
        all_types = [t["type"] for t in self.QA_TYPES]
        qa_types = [all_types[i % len(all_types)] for i in range(num_pairs)]
        difficulties = [self.DIFFICULTIES[i % len(self.DIFFICULTIES)] for i in range(num_pairs)]

        # Retry loop with exponential backoff
        for attempt in range(self.config.max_retries):
            if attempt > 0:
                time.sleep(min(2 ** attempt, 16))  # Cap at 16 seconds

            try:
                prompt = self._build_batch_prompt(contexts, num_pairs, qa_types, difficulties, topic_path)

                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=2000,  # Increased for multiple pairs
                )

                content = response.choices[0].message.content

                # Parse JSON array
                try:
                    # Try to extract JSON from markdown code block
                    if "```json" in content:
                        json_str = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        json_str = content.split("```")[1].split("```")[0]
                    else:
                        json_str = content

                    data_list = json.loads(json_str.strip())
                    if not isinstance(data_list, list):
                        logger.warning(f"[Batch] LLM returned non-list, got {type(data_list).__name__}")
                        continue

                    accepted_pairs: List[QAPair] = []

                    for i, data in enumerate(data_list):
                        if i >= num_pairs:
                            break

                        question = data.get("question", "").strip()
                        ground_truth = data.get("ground_truth", "").strip()
                        result_type = data.get("type", qa_types[i])
                        result_difficulty = data.get("difficulty", difficulties[i])

                        # Basic validation — both fields must exist
                        if not question or not ground_truth:
                            continue

                        # Check that the answer is grounded in the contexts
                        if not self._is_ground_truth_grounded(ground_truth, contexts, result_type):
                            continue

                        # Store for duplicate detection
                        self.generated_questions.add(question.lower().strip().rstrip("?"))

                        metadata = {
                            "type": result_type,
                            "difficulty": result_difficulty,
                            "attempt": attempt + 1,
                            "iteration": iteration,
                        }

                        if topic_path:
                            metadata["topic_path"] = topic_path

                        accepted_pairs.append(QAPair(
                            question=question,
                            ground_truth=ground_truth,
                            contexts=contexts,
                            metadata=metadata,
                        ))

                    if accepted_pairs:
                        logger.info(f"[Batch] Accepted {len(accepted_pairs)}/{len(data_list)} pairs")
                    else:
                        logger.warning(f"[Batch] All {len(data_list)} pairs rejected by quality filters")

                    return accepted_pairs

                except json.JSONDecodeError:
                    logger.warning(f"[Batch] JSON parse error: {content[:100]}...")
                    continue

            except Exception as e:
                logger.warning(f"[Batch] API error: {e}")
                continue

        # Failed after all retries
        return []

    def to_dataset_entry(self, qa_pair: QAPair) -> DatasetEntry:
        """Convert QAPair to RAGAS evaluation format."""
        return DatasetEntry(
            question=qa_pair.question,
            answer=qa_pair.ground_truth,
            ground_truth=qa_pair.ground_truth,
            contexts=qa_pair.contexts,
        )
