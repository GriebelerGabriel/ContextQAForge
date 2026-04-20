"""Configuration module for the RAG QA Pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load .env.local if it exists
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


@dataclass
class PipelineConfig:
    """Configuration for the RAG QA generation pipeline."""

    # Document processing
    chunk_size: int = 500
    chunk_overlap: int = 100

    # PaddleOCR
    paddleocr_lang: str = "en"  # OCR language: "en", "pt", "latin"
    use_table_recognition: bool = True

    # Generation
    num_samples: int = 100
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_retries: int = 3
    batch_size: int = 3  # QA pairs per LLM call (1-5)
    grounding_threshold: float = 0.25

    # Topic Tree LLM refinement
    topic_model: str = "gpt-4o-mini"  # Cheaper model for tree refinement
    max_section_chars_for_split: int = 1500  # Sections longer than this get LLM-split

    # Domain/Subject Focus
    document_domain: str = "general"

    # Language
    language: str = "pt-BR"  # Output language for generated QA pairs

    # Document-specific patterns for filtering
    pdf_remove_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_PDF_REMOVE_PATTERNS))
    chunk_quality_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_CHUNK_QUALITY_PATTERNS))

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
    def from_env(cls) -> "PipelineConfig":
        """Create config from environment variables."""
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            paddleocr_lang=os.getenv("PADDLEOCR_LANG", "en"),
            use_table_recognition=os.getenv("USE_TABLE_RECOGNITION", "true").lower() == "true",
            num_samples=int(os.getenv("NUM_SAMPLES", "100")),
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            batch_size=int(os.getenv("BATCH_SIZE", "3")),
            grounding_threshold=float(os.getenv("GROUNDING_THRESHOLD", "0.25")),
            topic_model=os.getenv("TOPIC_MODEL", "gpt-4o-mini"),
            max_section_chars_for_split=int(os.getenv("MAX_SECTION_CHARS_FOR_SPLIT", "1500")),
            document_domain=os.getenv("DOCUMENT_DOMAIN", "general"),
            language=os.getenv("LANGUAGE", "pt-BR"),
            output_path=os.getenv("OUTPUT_PATH", "dataset.json"),
            log_path=os.getenv("LOG_PATH", "pipeline.log"),
            tree_path=os.getenv("TREE_PATH", "tree/topic_tree.json"),
            parsed_path=os.getenv("PARSED_PATH", "tree/parsed"),
        )
