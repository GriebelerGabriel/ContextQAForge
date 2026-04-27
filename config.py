"""Configuration module for the RAG QA Pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load .env.local if it exists, otherwise .env
env_path = Path(".env.local")
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Default header/footer patterns for Brazilian healthcare documents
DEFAULT_PDF_REMOVE_PATTERNS: List[str] = [
    "manual de orientações",
    "pacientes e cuidadores",
    "serviço de nutrição",
    "dietética",
    "rua ramiro barcelos",
    "largo eduardo",
    "porto alegre",
    "fones",
    "www.hcpa",
    "hospital de clínicas",
    "educação em saúde",
    "publicação autorizada",
    "vol. 160",
    "vol 160",
    "pes160",
    "pes 160",
    "311721",
    "311421",
    "alimentação e saúde",
    "saúde cardiovascular",
]

DEFAULT_CHUNK_QUALITY_PATTERNS: List[str] = [
    "publicação autorizada",
    "manual de orientações",
    "vol. 160",
    "vol.160",
    "pes160",
    "conselho editorial",
    "serviço de nutrição",
    "educação em saúde",
    "educação\nem saúde",
    "aprovado pelo",
    "pacientes e cuidadores",
    "publicação\nautorizada",
]


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() == "true"


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass
class PipelineConfig:
    """Configuration for the RAG QA generation pipeline.

    All defaults come from environment variables (loaded from .env.local).
    Use PipelineConfig.from_env() or pass explicit values to override.
    """

    # Document processing
    chunk_size: int = 500
    chunk_overlap: int = 100

    # PaddleOCR
    paddleocr_lang: str = "en"
    use_table_recognition: bool = True

    # Generation
    num_samples: int = 100
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_retries: int = 3
    batch_size: int = 5
    grounding_threshold: float = 0.15

    # Domain/Subject Focus
    document_domain: str = "general"

    # Language
    language: str = "pt-BR"

    # Document-specific patterns for filtering
    pdf_remove_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_PDF_REMOVE_PATTERNS))
    chunk_quality_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_CHUNK_QUALITY_PATTERNS))

    # LLM-driven pipeline
    use_llm_pipeline: bool = True
    slicer_model: str = "gpt-4o-mini"
    topic_model: str = "gpt-4o-mini"
    segment_max_chars: int = 15000
    slicer_cache_dir: str = "tree/sliced"

    # QA distribution
    balance_per_document: bool = True
    balance_true_false: bool = True

    # API
    openai_api_key: Optional[str] = None

    # Output
    output_path: str = "dataset.json"
    log_path: str = "pipeline.log"
    tree_path: str = "tree/topic_tree.json"
    parsed_path: str = "tree/parsed"

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key is None:
            raise ValueError(
                "OpenAI API key must be provided either via config or OPENAI_API_KEY environment variable"
            )

    @classmethod
    def from_env(cls, **overrides) -> "PipelineConfig":
        """Create config from environment variables (.env.local).

        Any keyword argument overrides the env value.
        """
        return cls(
            chunk_size=overrides.get("chunk_size", _env_int("CHUNK_SIZE", 500)),
            chunk_overlap=overrides.get("chunk_overlap", _env_int("CHUNK_OVERLAP", 100)),
            paddleocr_lang=overrides.get("paddleocr_lang", _env_str("PADDLEOCR_LANG", "en")),
            use_table_recognition=overrides.get("use_table_recognition", _env_bool("USE_TABLE_RECOGNITION", True)),
            num_samples=overrides.get("num_samples", _env_int("NUM_SAMPLES", 100)),
            model_name=overrides.get("model_name", _env_str("MODEL_NAME", "gpt-4o-mini")),
            temperature=overrides.get("temperature", _env_float("TEMPERATURE", 0.7)),
            max_retries=overrides.get("max_retries", _env_int("MAX_RETRIES", 3)),
            batch_size=overrides.get("batch_size", _env_int("BATCH_SIZE", 5)),
            grounding_threshold=overrides.get("grounding_threshold", _env_float("GROUNDING_THRESHOLD", 0.15)),
            document_domain=overrides.get("document_domain", _env_str("DOCUMENT_DOMAIN", "general")),
            language=overrides.get("language", _env_str("LANGUAGE", "pt-BR")),
            use_llm_pipeline=overrides.get("use_llm_pipeline", _env_bool("USE_LLM_PIPELINE", True)),
            slicer_model=overrides.get("slicer_model", _env_str("SLICER_MODEL", "gpt-4o-mini")),
            topic_model=overrides.get("topic_model", _env_str("TOPIC_MODEL", "gpt-4o-mini")),
            segment_max_chars=overrides.get("segment_max_chars", _env_int("SEGMENT_MAX_CHARS", 15000)),
            slicer_cache_dir=overrides.get("slicer_cache_dir", _env_str("SLICER_CACHE_DIR", "tree/sliced")),
            balance_per_document=overrides.get("balance_per_document", _env_bool("BALANCE_PER_DOCUMENT", True)),
            balance_true_false=overrides.get("balance_true_false", _env_bool("BALANCE_TRUE_FALSE", True)),
            output_path=overrides.get("output_path", _env_str("OUTPUT_PATH", "dataset.json")),
            log_path=overrides.get("log_path", _env_str("LOG_PATH", "pipeline.log")),
            tree_path=overrides.get("tree_path", _env_str("TREE_PATH", "tree/topic_tree.json")),
            parsed_path=overrides.get("parsed_path", _env_str("PARSED_PATH", "tree/parsed")),
        )
