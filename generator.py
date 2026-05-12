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
from collections import deque
from typing import Dict, List, Optional, Set

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

    # Step 1 extraction few-shot examples (contexts -> extracted fact)
    EXTRACT_EXAMPLES = {
        "single-hop": {
            "contexts": [
                "The recommended daily sodium intake for adults is 2g per day (equivalent to 5g of salt).",
            ],
            "extracted_fact": "The recommended daily sodium intake for adults is 2g per day, which is equivalent to 5g of salt.",
        },
        "multi-hop": {
            "contexts": [
                "The DASH diet was developed for people with hypertension and helps control blood pressure.",
                "The DASH diet is rich in fiber, vitamins and minerals, and low in salt, sweets and saturated fat.",
            ],
            "extracted_fact": "The DASH diet was developed specifically for people with hypertension to help control blood pressure. It is rich in fiber, vitamins, and minerals, and low in salt, sweets, and saturated fat.",
        },
        "inference": {
            "contexts": [
                "Reducing sodium intake to <2 g/day was more beneficial for blood pressure than consuming >2 g/day.",
                "Higher sodium intake was associated with higher risk of stroke and coronary heart disease.",
            ],
            "extracted_fact": "Reducing sodium intake to less than 2g per day is more beneficial for blood pressure than reducing but still consuming more than 2g. Higher sodium intake is linked to higher risk of stroke and coronary heart disease.",
        },
        "paraphrase": {
            "contexts": [
                "Replacing butter with plant oils containing predominantly unsaturated fat decreases LDL cholesterol concentrations.",
            ],
            "extracted_fact": "Replacing butter with plant oils that contain predominantly unsaturated fat reduces LDL cholesterol levels.",
        },
        "true-false": {
            "contexts": [
                "Reducing sodium intake to <2 g/day was more beneficial for blood pressure than reducing sodium intake but still consuming >2 g/day.",
            ],
            "extracted_fact": "Reducing sodium intake to less than 2g per day is more beneficial for blood pressure than reducing sodium but still consuming more than 2g per day.",
        },
    }

    # Step 2 question-generation few-shot examples (fact -> question)
    QUESTION_EXAMPLES = {
        "single-hop": {
            "extracted_fact": "The recommended daily sodium intake for adults is 2g per day, which is equivalent to 5g of salt.",
            "question": "Quanto de sal posso consumir por dia sem passar do limite recomendado?",
            "ground_truth": "Você pode consumir no máximo 5g de sal por dia, o que equivale a 2g de sódio.",
            "difficulty": "easy",
        },
        "multi-hop": {
            "extracted_fact": "The DASH diet was developed specifically for people with hypertension to help control blood pressure. It is rich in fiber, vitamins, and minerals, and low in salt, sweets, and saturated fat.",
            "question": "Se eu tenho pressão alta, qual dieta devo seguir e quais alimentos ela inclui?",
            "ground_truth": "Se você tem pressão alta, a dieta Dash foi feita especialmente para isso. Ela inclui alimentos ricos em fibras, vitaminas e minerais (como feijões, grãos integrais, frutas e legumes) e tem pouco sal, doces e gordura saturada.",
            "difficulty": "medium",
        },
        "inference": {
            "extracted_fact": "Reducing sodium intake to less than 2g per day is more beneficial for blood pressure than reducing but still consuming more than 2g. Higher sodium intake is linked to higher risk of stroke and coronary heart disease.",
            "question": "Se eu consigo reduzir meu consumo de sódio para menos de 2g por dia, isso realmente faz diferença na minha pressão arterial?",
            "ground_truth": "Sim, faz muita diferença. Consumir menos de 2g de sódio por dia foi mais eficaz para reduzir a pressão arterial do que apenas diminuir mas continuar acima de 2g. Além disso, consumir mais sódio está ligado a maior risco de derrame e doenças cardíacas.",
            "difficulty": "hard",
        },
        "paraphrase": {
            "extracted_fact": "Replacing butter with plant oils that contain predominantly unsaturated fat reduces LDL cholesterol levels.",
            "question": "É melhor cozinhar com óleo vegetal ou manteiga se eu quiser baixar meu colesterol?",
            "ground_truth": "É melhor usar óleos vegetais, porque eles têm gorduras insaturadas que ajudam a reduzir o colesterol ruim (LDL), ao contrário da manteiga.",
            "difficulty": "medium",
        },
        "true-false": {
            "extracted_fact": "Reducing sodium intake to less than 2g per day is more beneficial for blood pressure than reducing sodium but still consuming more than 2g per day.",
            "question": "Afirme: Consumir menos de 2g de sódio por dia é mais benéfico para a minha pressão arterial do que apenas reduzir mas continuar consumindo mais de 2g.",
            "ground_truth": "VERDADEIRO. Reduzir a ingestão de sódio para menos de 2 g/dia foi mais benéfico para a pressão arterial do que reduzir mas ainda consumir mais de 2 g/dia.",
            "difficulty": "easy",
        },
    }

    # Maximum number of past questions to check for duplicates
    DEDUP_WINDOW = 20
    # Number of candidate facts to extract per Step 1 call
    NUM_CANDIDATE_FACTS = 3

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.generated_questions: Set[str] = set()
        self._recent_questions: deque = deque(maxlen=self.DEDUP_WINDOW)
        self._recent_facts: deque = deque(maxlen=self.DEDUP_WINDOW)
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
                "   PATTERNS TO USE (vary these — do NOT always start with 'Se eu'):\n"
                '   - "Posso...?" / "Devo...?" / "Qual é o melhor...?" / "Quanto...?" / "É verdade que...?"\n'
                '   - "O que [posso/devo] fazer para [objetivo]?"\n'
                '   - "Quais alimentos [ajudam/pioram]...?" / "Qual a diferença entre X e Y?"\n'
                '   - "Tenho [condição], posso [ação]?" / "É seguro [ação] se eu tenho [condição]?"\n'
                '   - Use "Se eu..." AT MOST once per 5 questions. Use other patterns for the rest.\n'
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

        for existing in self._recent_questions:
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

        With the two-step pipeline, grounding is guaranteed by construction
        (facts are extracted directly from contexts in Step 1).
        This is kept as a lightweight sanity check.
        """
        if not ground_truth.strip():
            return False
        return True

    # ── Step 1: Fact extraction ──────────────────────────────────────

    def _get_extract_system_prompt(self) -> str:
        """System prompt for Step 1: extract a key fact from contexts."""
        language = self.config.language
        if language == "pt-BR":
            lang_instruction = (
                "The input contexts may be in ENGLISH, but you MUST output in PORTUGUESE (Brazilian Portuguese)."
            )
        else:
            lang_instruction = f"Output in: {language}."

        return (
            "You are a precise fact extractor. Your job is to extract ONE specific, substantive fact "
            "from the provided document contexts.\n\n"
            f"{lang_instruction}\n\n"
            "RULES:\n"
            "1. Extract ONLY information that is explicitly stated in the contexts.\n"
            "2. Do NOT add outside knowledge or inference.\n"
            "3. Keep specific numbers, quantities, names, and details from the text.\n"
            "4. The fact should be detailed enough that a question can be written about it.\n"
            "5. Write in a clear, complete sentence.\n\n"
            'Output ONLY a JSON object: {"extracted_fact": "the fact here"}'
        )

    def _get_extract_prompt(
        self,
        contexts: List[str],
        topic_path: List[str],
        source: str,
        qa_type: str,
        full_content: str = "",
        related_chunks: Optional[List[str]] = None,
    ) -> str:
        """Build the Step 1 extraction prompt."""
        topic_str = " -> ".join(topic_path)
        context_text = "\n\n".join(f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts))

        example = self.EXTRACT_EXAMPLES.get(qa_type, self.EXTRACT_EXAMPLES["single-hop"])
        example_contexts = "\n\n".join(example["contexts"])

        supplement = ""
        if full_content and full_content not in context_text:
            supplement += f"\n=== FULL SECTION TEXT ===\n{full_content}\n"

        if related_chunks:
            related_text = "\n\n".join(
                f"[Related - {i+1}]: {chunk}" for i, chunk in enumerate(related_chunks)
            )
            supplement += f"\n=== RELATED CONTENT ===\n{related_text}\n"

        # Build avoidance hint from recently extracted facts
        avoidance = ""
        if self._recent_facts:
            recent_samples = list(self._recent_facts)[-3:]
            avoidance = (
                "\n=== FACTS ALREADY USED (pick DIFFERENT facts) ===\n"
                + "\n".join(f"- {f}" for f in recent_samples)
                + "\n"
            )

        return f"""Extract {self.NUM_CANDIDATE_FACTS} distinct key facts from these contexts.

=== TOPIC ANGLE ===
{topic_str}

=== CONTEXTS ===
{context_text}
{supplement}{avoidance}
=== EXAMPLE ===
Contexts:
{example_contexts}

Extracted fact: {example["extracted_fact"]}

=== YOUR TASK ===
Extract {self.NUM_CANDIDATE_FACTS} DIFFERENT facts from the contexts above.
Each fact should be substantive (contain specific details, numbers, or recommendations).
Do NOT repeat or rephrase the same information — each fact must cover a different detail.

Output ONLY a JSON array:
[
  {{"extracted_fact": "fact 1"}},
  {{"extracted_fact": "fact 2"}},
  ...
]"""

    def _step1_extract_fact(
        self,
        contexts: List[str],
        topic_path: List[str],
        source: str,
        qa_type: str,
        full_content: str = "",
        related_chunks: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Step 1: Extract candidate facts from contexts and pick the freshest one."""
        prompt = self._get_extract_prompt(
            contexts, topic_path, source, qa_type, full_content, related_chunks,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self._get_extract_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=800,
            )
            content = response.choices[0].message.content
            data = self._parse_json_response(content)

            # Handle both single object and array responses
            if isinstance(data, dict):
                candidates = [data.get("extracted_fact", "").strip()]
            elif isinstance(data, list):
                candidates = [
                    item.get("extracted_fact", "").strip()
                    for item in data if isinstance(item, dict)
                ]
            else:
                return None

            candidates = [c for c in candidates if c]
            if not candidates:
                return None

            picked = self._pick_freshest_fact(candidates)
            self._recent_facts.append(picked)
            return picked

        except Exception as e:
            logger.warning(f"[Step 1] Extract error: {e}")
            return None

    def _pick_freshest_fact(self, candidates: List[str]) -> str:
        """Pick the candidate with least overlap with recently extracted facts."""
        if not self._recent_facts:
            return random.choice(candidates)

        def overlap_score(fact: str) -> float:
            fact_words = set(fact.lower().split())
            max_overlap = 0.0
            for recent in self._recent_facts:
                recent_words = set(recent.lower().split())
                overlap = len(fact_words & recent_words) / max(len(fact_words), 1)
                max_overlap = max(max_overlap, overlap)
            return max_overlap

        scored = [(c, overlap_score(c)) for c in candidates]
        scored.sort(key=lambda x: x[1])
        # Pick among the lowest-overlap candidates (bottom 40%)
        top_n = max(1, len(scored) * 2 // 5)
        freshest = [s[0] for s in scored[:top_n]]
        return random.choice(freshest)

    # ── Step 2: Question generation ─────────────────────────────────

    def _get_question_prompt(
        self,
        extracted_fact: str,
        contexts: List[str],
        topic_path: List[str],
        source: str,
        qa_type: str,
        difficulty: str,
        tf_polarity: Optional[str] = None,
    ) -> str:
        """Build the Step 2 question-generation prompt."""
        topic_str = " -> ".join(topic_path)
        context_text = "\n\n".join(f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts))

        example = self.QUESTION_EXAMPLES.get(qa_type, self.QUESTION_EXAMPLES["single-hop"])

        polarity_instruction = ""
        if qa_type == "true-false" and tf_polarity:
            if tf_polarity == "true":
                polarity_instruction = (
                    "\nIMPORTANT: This assertion MUST be TRUE (VERDADEIRO). "
                    "Generate a statement that is factually correct according to the extracted fact."
                )
            else:
                polarity_instruction = (
                    "\nIMPORTANT: This assertion MUST be FALSE (FALSO). "
                    "Subtly alter one detail from the extracted fact so the statement becomes incorrect."
                )

        type_hints = {
            "single-hop": "The question should be directly answerable from the extracted fact.",
            "multi-hop": "The question should require connecting the extracted fact with information visible in the contexts.",
            "inference": "The question should require reasoning or inference based on the extracted fact and contexts.",
            "paraphrase": "The question should use different wording than the context — rephrase naturally.",
            "true-false": (
                'The question field should be a STATEMENT starting with "Afirme:" that can be verified as true or false. '
                'The ground_truth MUST start with "VERDADEIRO." or "FALSO." followed by evidence.'
            ),
        }

        return f"""Generate a question that is answered by the extracted fact.

=== EXTRACTED FACT (this is the answer) ===
{extracted_fact}

=== ORIGINAL CONTEXTS (for reference) ===
{context_text}

=== TOPIC ANGLE ===
{topic_str}

=== TYPE: {qa_type} | DIFFICULTY: {difficulty} ===
{type_hints.get(qa_type, type_hints["single-hop"])}
{polarity_instruction}

=== EXAMPLE ({qa_type}) ===
Extracted fact: {example["extracted_fact"]}
Question: {example["question"]}
Ground Truth: {example["ground_truth"]}

=== YOUR TASK ===
Generate a NEW {qa_type} question at {difficulty} difficulty.
The question MUST be written as a REAL PERSON asking for personal advice (use "eu", "posso", "devo").
The ground_truth MUST be based on the extracted fact, written in a natural conversational tone.
Do NOT ask questions answerable from general/popular knowledge.

Output ONLY the JSON object:"""

    def _step2_generate_question(
        self,
        extracted_fact: str,
        contexts: List[str],
        topic_path: List[str],
        source: str,
        qa_type: str,
        difficulty: str,
        tf_polarity: Optional[str] = None,
    ) -> Optional[QAPair]:
        """Step 2: Generate a question for the extracted fact."""
        prompt = self._get_question_prompt(
            extracted_fact, contexts, topic_path, source,
            qa_type, difficulty, tf_polarity,
        )
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
                return None

            question = data.get("question", "").strip()
            ground_truth = data.get("ground_truth", "").strip()
            if not question or not ground_truth:
                return None

            # Validate true-false polarity
            if qa_type == "true-false" and tf_polarity:
                gt_lower = ground_truth.strip().lower()
                if tf_polarity == "true" and gt_lower.startswith("falso"):
                    return None
                if tf_polarity == "false" and gt_lower.startswith("verdadeiro"):
                    return None

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
            logger.warning(f"[Step 2] Question generation error: {e}")
            return None

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

        if num_pairs == 1:
            pair = self._generate_single(contexts, topic_path, source, full_content, related_chunks)
            return [pair] if pair else []
        else:
            return self._generate_batch(contexts, topic_path, source, num_pairs, full_content, related_chunks)

    def _generate_single(
        self,
        contexts: List[str],
        topic_path: List[str],
        source: str,
        full_content: str = "",
        related_chunks: Optional[List[str]] = None,
    ) -> Optional[QAPair]:
        """Generate a single QA pair using the two-step pipeline."""
        qa_type = random.choice([t["type"] for t in self.QA_TYPES])
        difficulty = random.choice(self.DIFFICULTIES)

        tf_polarity = None
        if qa_type == "true-false" and self.config.balance_true_false:
            if self.true_false_count["true"] <= self.true_false_count["false"]:
                tf_polarity = "true"
            else:
                tf_polarity = "false"

        for attempt in range(self.config.max_retries):
            if attempt > 0:
                time.sleep(min(2 ** attempt, 16))

            # Step 1: Extract fact from contexts
            extracted_fact = self._step1_extract_fact(
                contexts, topic_path, source, qa_type, full_content, related_chunks,
            )
            if not extracted_fact:
                continue

            # Step 2: Generate question for the extracted fact
            qa_pair = self._step2_generate_question(
                extracted_fact, contexts, topic_path, source,
                qa_type, difficulty, tf_polarity,
            )
            if not qa_pair:
                continue

            # Validate (grounding is guaranteed by construction)
            if self._is_generic_question(qa_pair.question):
                logger.info(f"[Rejected] Generic question: {qa_pair.question[:60]}...")
                continue
            if self._is_duplicate(qa_pair.question):
                logger.info(f"[Rejected] Duplicate question: {qa_pair.question[:60]}...")
                continue

            normalized = qa_pair.question.lower().strip().rstrip("?")
            self.generated_questions.add(normalized)
            self._recent_questions.append(normalized)
            if tf_polarity:
                self.true_false_count[tf_polarity] += 1

            return qa_pair

        return None

    def _generate_batch(
        self,
        contexts: List[str],
        topic_path: List[str],
        source: str,
        num_pairs: int,
        full_content: str = "",
        related_chunks: Optional[List[str]] = None,
    ) -> List[QAPair]:
        """Generate multiple QA pairs using the two-step pipeline."""
        all_types = [t["type"] for t in self.QA_TYPES]
        qa_types = [all_types[i % len(all_types)] for i in range(num_pairs)]
        difficulties = [self.DIFFICULTIES[i % len(self.DIFFICULTIES)] for i in range(num_pairs)]

        # Track true-false polarities
        tf_polarities: List[Optional[str]] = []
        for qt in qa_types:
            if qt == "true-false" and self.config.balance_true_false:
                if self.true_false_count["true"] <= self.true_false_count["false"]:
                    tf_polarities.append("true")
                else:
                    tf_polarities.append("false")
            else:
                tf_polarities.append(None)

        for attempt in range(self.config.max_retries):
            if attempt > 0:
                time.sleep(min(2 ** attempt, 16))

            # Step 1: Extract N facts
            facts = self._step1_extract_facts_batch(
                contexts, topic_path, source, num_pairs, full_content, related_chunks,
            )
            if not facts:
                continue

            # Step 2: Generate questions for each fact
            qa_pairs = self._step2_generate_questions_batch(
                facts, contexts, topic_path, source, qa_types, difficulties, tf_polarities,
            )
            if not qa_pairs:
                continue

            # Validate and accept
            accepted: List[QAPair] = []
            for i, pair in enumerate(qa_pairs):
                if self._is_generic_question(pair.question):
                    continue
                if self._is_duplicate(pair.question):
                    continue

                normalized = pair.question.lower().strip().rstrip("?")
                self.generated_questions.add(normalized)
                self._recent_questions.append(normalized)
                if i < len(tf_polarities) and tf_polarities[i]:
                    self.true_false_count[tf_polarities[i]] += 1
                accepted.append(pair)

            if accepted:
                logger.info(f"[Batch] Accepted {len(accepted)} pairs from '{source}'")
            return accepted

        return []

    def _step1_extract_facts_batch(
        self,
        contexts: List[str],
        topic_path: List[str],
        source: str,
        num_facts: int,
        full_content: str = "",
        related_chunks: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """Step 1 batch: Extract N facts from contexts in one LLM call."""
        topic_str = " -> ".join(topic_path)
        context_text = "\n\n".join(f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts))

        supplement = ""
        if full_content and full_content not in context_text:
            supplement += f"\n=== FULL SECTION TEXT ===\n{full_content}\n"
        if related_chunks:
            related_text = "\n\n".join(
                f"[Related - {i+1}]: {chunk}" for i, chunk in enumerate(related_chunks)
            )
            supplement += f"\n=== RELATED CONTENT ===\n{related_text}\n"

        example = self.EXTRACT_EXAMPLES["single-hop"]
        example_contexts = "\n\n".join(example["contexts"])

        prompt = f"""Extract {num_facts} distinct key facts from these contexts.

=== TOPIC ANGLE ===
{topic_str}

=== CONTEXTS ===
{context_text}
{supplement}
=== EXAMPLE ===
Contexts:
{example_contexts}

Extracted fact: {example["extracted_fact"]}

=== YOUR TASK ===
Extract {num_facts} DIFFERENT facts from the contexts.
Each fact should be substantive (contain specific details, numbers, or recommendations).
Facts must be distinct — do not repeat the same information.

Output ONLY a JSON array:
[
  {{"extracted_fact": "fact 1"}},
  {{"extracted_fact": "fact 2"}},
  ...
]"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self._get_extract_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=1500,
            )
            content = response.choices[0].message.content
            data_list = self._parse_json_response(content)

            if not isinstance(data_list, list):
                return None

            facts = []
            for item in data_list[:num_facts]:
                fact = item.get("extracted_fact", "").strip() if isinstance(item, dict) else ""
                if fact:
                    facts.append(fact)

            return facts if facts else None
        except Exception as e:
            logger.warning(f"[Step 1 Batch] Extract error: {e}")
            return None

    def _step2_generate_questions_batch(
        self,
        facts: List[str],
        contexts: List[str],
        topic_path: List[str],
        source: str,
        qa_types: List[str],
        difficulties: List[str],
        tf_polarities: List[Optional[str]],
    ) -> Optional[List[QAPair]]:
        """Step 2 batch: Generate questions for N facts in one LLM call."""
        topic_str = " -> ".join(topic_path)
        context_text = "\n\n".join(f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts))

        # Build per-fact instructions
        fact_lines = []
        for i, fact in enumerate(facts):
            line = f'{i+1}. Fact: "{fact}" | Type: {qa_types[i]} | Difficulty: {difficulties[i]}'
            if i < len(tf_polarities) and tf_polarities[i]:
                if tf_polarities[i] == "true":
                    line += " — MUST be TRUE (VERDADEIRO)"
                else:
                    line += " — MUST be FALSE (FALSO). Subtly alter one detail."
            fact_lines.append(line)
        facts_str = "\n".join(fact_lines)

        example = self.QUESTION_EXAMPLES.get(qa_types[0], self.QUESTION_EXAMPLES["single-hop"])

        prompt = f"""Generate a question for each extracted fact.

=== ORIGINAL CONTEXTS (for reference) ===
{context_text}

=== FACTS AND TYPES ===
{facts_str}

=== EXAMPLE ({qa_types[0]}) ===
Extracted fact: {example["extracted_fact"]}
Question: {example["question"]}
Ground Truth: {example["ground_truth"]}

=== YOUR TASK ===
Generate a question for EACH fact above.
Every question MUST be written as a REAL PERSON asking for personal advice (use "eu", "posso", "devo").
The ground_truth MUST be based on the extracted fact, written naturally.
Do NOT ask questions answerable from general/popular knowledge.

Output ONLY a JSON array:
[
  {{"question": "...", "ground_truth": "...", "type": "...", "difficulty": "..."}},
  ...
]"""

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
                return None

            pairs: List[QAPair] = []
            for i, data in enumerate(data_list[:len(facts)]):
                if not isinstance(data, dict):
                    continue
                question = data.get("question", "").strip()
                ground_truth = data.get("ground_truth", "").strip()
                if not question or not ground_truth:
                    continue

                # Validate true-false polarity
                polarity = tf_polarities[i] if i < len(tf_polarities) else None
                if qa_types[i] == "true-false" and polarity:
                    gt_lower = ground_truth.strip().lower()
                    if polarity == "true" and gt_lower.startswith("falso"):
                        continue
                    if polarity == "false" and gt_lower.startswith("verdadeiro"):
                        continue

                pairs.append(QAPair(
                    question=question,
                    ground_truth=ground_truth,
                    contexts=contexts,
                    metadata={
                        "type": data.get("type", qa_types[i]),
                        "difficulty": data.get("difficulty", difficulties[i]),
                        "topic_path": topic_path,
                        "source": source,
                    },
                ))

            return pairs if pairs else None
        except Exception as e:
            logger.warning(f"[Step 2 Batch] Question generation error: {e}")
            return None

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
