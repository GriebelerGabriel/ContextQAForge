# ContextQAForge

Generate evaluation-ready QA datasets from your documents for RAG systems. Given a folder of PDFs, TXTs, or Markdown files, the pipeline produces RAGAS-compatible question-answer pairs grounded in the actual document content.

## Pipeline Architecture

```
Documents (PDF/TXT/MD)
        |
        v
[1. Load & Parse] ── Digital PDFs: PyMuPDF (text + tables + headings)
        |              Scanned PDFs: PaddleOCR PPStructureV3
        v
[2. LLM Slice] ── gpt-4o-mini splits into substantive content segments
        |           (boilerplate discarded, large docs chunked safely)
        v
[3. Build Topic Tree] ── gpt-4o-mini organizes segments into hierarchy
        |                  (2-3 levels deep, preserves all content)
        v
[4. Chunk Leaves] ── Split leaf node text into 500-char chunks
        |
        v
[5. Generate QA] ── For each leaf node:
        |            a. Pick next leaf from balanced distribution
        |            b. Find related content via embedding similarity
        |            c. LLM generates QA pairs (batch mode)
        |            d. Validate: grounded, non-generic, non-duplicate
        v
   dataset.json (RAGAS format)
```

### How QA generation works

The key principle: **contexts are truth, topic is guidance.**

1. The topic tree suggests what *angle* to ask about
2. Embedding similarity finds related content from other documents
3. The LLM generates questions that can only be answered from those contexts
4. Validation checks: grounding (answer derived from contexts), non-generic (requires document-specific info), deduplication

## Installation

```bash
pip install -r requirements.txt
```

Set your OpenAI API key in `.env.local`:
```
OPENAI_API_KEY=your-api-key
```

## Quick Start

```bash
python main.py ./documents --num-samples 10
```

This generates 10 QA pairs from documents in the `./documents` folder and saves to `dataset.json`.

## Usage

### CLI

```bash
python main.py ./documents \
    --num-samples 50 \
    --batch-size 5 \
    --chunk-size 500 \
    --model gpt-4o-mini \
    --domain health \
    --language pt-BR \
    --output dataset.json
```

### Python API

```python
from config import PipelineConfig
from main import run_pipeline, save_dataset
import logging

config = PipelineConfig(
    num_samples=50,
    document_domain="health",
    language="pt-BR",
    openai_api_key="your-key",
)

logger = logging.getLogger(__name__)
dataset = run_pipeline("./documents", config, logger)
save_dataset(dataset, "output.json", logger)
```

### Environment Variables

All settings can be configured via `.env.local`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `NUM_SAMPLES` | 100 | Target QA pairs |
| `MODEL_NAME` | gpt-4o-mini | LLM for QA generation |
| `SLICER_MODEL` | gpt-4o-mini | LLM for content slicing |
| `TOPIC_MODEL` | gpt-4o-mini | LLM for topic tree building |
| `EMBEDDING_MODEL` | text-embedding-3-small | Embedding model |
| `BATCH_SIZE` | 5 | QA pairs per LLM call |
| `TEMPERATURE` | 0.7 | Generation temperature |
| `MAX_RETRIES` | 3 | Retries per failed generation |
| `DOCUMENT_DOMAIN` | general | Domain focus (health, nutrition, etc.) |
| `LANGUAGE` | pt-BR | Output language for QA pairs |
| `PADDLEOCR_LANG` | en | PaddleOCR language (en, pt, latin) |

### CLI Arguments

```
python main.py ./documents \
    --chunk-size 500 \
    --chunk-overlap 100 \
    --num-samples 100 \
    --model gpt-4o-mini \
    --slicer-model gpt-4o-mini \
    --topic-model gpt-4o-mini \
    --temperature 0.7 \
    --batch-size 5 \
    --domain health \
    --language pt-BR \
    --output dataset.json \
    --tree-path tree/topic_tree.json \
    --paddleocr-lang en \
    --no-llm-pipeline \
    --no-balance-docs \
    --no-balance-tf \
    --fresh
```

## Output Format

RAGAS-compatible JSON:

```json
[
  {
    "question": "Qual a quantidade de sódio diária recomendada para reduzir a pressão arterial?",
    "answer": "Reduzir a ingestão de sódio para menos de 2 g/dia é mais benéfico para a pressão arterial.",
    "contexts": [
      "Reducing sodium intake to <2 g/day was more beneficial for blood pressure...",
      "Higher sodium intake was associated with higher risk of stroke..."
    ],
    "ground_truth": "Reduzir a ingestão de sódio para menos de 2 g/dia é mais benéfico para a pressão arterial."
  }
]
```

- `question` — generated question in the configured language
- `answer` / `ground_truth` — answer extracted from the document contexts
- `contexts` — the retrieved document chunks used to generate the answer

## QA Types

| Type | Description |
|------|-------------|
| single-hop | Direct lookup from one context chunk |
| multi-hop | Connects information across multiple chunks |
| inference | Requires reasoning from the contexts |
| paraphrase | Uses different wording than the source text |
| true-false | Verify a statement as true or false based on contexts |

## Project Structure

```
├── main.py                # Pipeline orchestration & CLI
├── config.py              # All configuration (dataclass + env vars)
├── models.py              # Pydantic models (QAPair, DatasetEntry, SectionNode)
├── loader.py              # Document loading (PDF with PyMuPDF/PaddleOCR, TXT, MD)
├── embedder.py            # OpenAI embeddings with batching
├── vector_store.py        # FAISS vector store with diversity filtering
├── topic_tree.py          # Document-grounded topic tree
├── llm_slicer.py          # LLM content slicing (Pass 1: extract substantive content)
├── llm_topic_builder.py   # LLM topic organization (Pass 2: build hierarchy)
├── generator.py           # QA generation with grounding validation
├── tests/                 # Test suite
├── requirements.txt       # Dependencies
└── .env.local             # API key and settings (not committed)
```
