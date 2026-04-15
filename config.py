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

    # Vector store
    top_k: int = 5

    # Generation
    num_samples: int = 100
    model_name: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.7
    max_retries: int = 3
    batch_size: int = 3  # QA pairs per LLM call (1-5)
    grounding_threshold: float = 0.25  # Min fraction of GT words found in contexts (was 0.3)

    # Topic Tree
    tree_degree: int = 3  # Children per node
    tree_depth: int = 3   # Levels deep
    topic_model: str = "gpt-4o-mini"  # Model for topic tree generation (cheaper)

    # PDF Processing
    use_llm_for_pdf: bool = False  # Use LLM for table/structure extraction (slower, more accurate)
    pdf_extraction_model: str = "gpt-4o-mini"  # Model for PDF extraction if enabled

    # Domain/Subject Focus
    document_domain: str = "general"  # Domain for QA focus (e.g., nutrition, technology, medicine)

    # Language
    language: str = "pt-BR"  # Output language for generated QA pairs

    # Document-specific patterns for filtering (defaults are for Brazilian healthcare docs)
    pdf_remove_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_PDF_REMOVE_PATTERNS))
    chunk_quality_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_CHUNK_QUALITY_PATTERNS))

    # Chunk deduplication
    dedup_overlap_threshold: float = 0.8  # Word overlap threshold for chunk dedup (was 0.6)

    # API
    openai_api_key: Optional[str] = None

    # Output
    output_path: str = "dataset.json"
    log_path: str = "pipeline.log"

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
            top_k=int(os.getenv("TOP_K", "5")),
            num_samples=int(os.getenv("NUM_SAMPLES", "100")),
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            batch_size=int(os.getenv("BATCH_SIZE", "3")),
            grounding_threshold=float(os.getenv("GROUNDING_THRESHOLD", "0.25")),
            tree_degree=int(os.getenv("TREE_DEGREE", "3")),
            tree_depth=int(os.getenv("TREE_DEPTH", "3")),
            topic_model=os.getenv("TOPIC_MODEL", "gpt-4o-mini"),
            use_llm_for_pdf=os.getenv("USE_LLM_FOR_PDF", "false").lower() == "true",
            pdf_extraction_model=os.getenv("PDF_EXTRACTION_MODEL", "gpt-4o-mini"),
            document_domain=os.getenv("DOCUMENT_DOMAIN", "general"),
            language=os.getenv("LANGUAGE", "pt-BR"),
            output_path=os.getenv("OUTPUT_PATH", "dataset.json"),
            log_path=os.getenv("LOG_PATH", "pipeline.log"),
            dedup_overlap_threshold=float(os.getenv("DEDUP_OVERLAP_THRESHOLD", "0.8")),
        )
