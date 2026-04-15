# ContextQAForge

Generate evaluation-ready QA datasets from your documents for RAG systems. Given a folder of PDFs, TXTs, or Markdown files, the pipeline produces RAGAS-compatible question-answer pairs grounded in the actual document content.

## Pipeline Architecture

```
Documents (PDF/TXT/MD)
        |
        v
[1. Load & Chunk] ── Split into 500-char chunks with overlap
        |
        v
[2. Embed & Index] ── OpenAI embeddings → FAISS vector store
        |
        v
[3. Build Topic Tree] ── gpt-4o-mini extracts hierarchical topics
        |                   (degree=3, depth=3 → 27 leaf paths)
        v
[4. Generate Query Seeds] ── Expand each topic into rich search terms
        |                       (1 batch call to gpt-4o-mini)
        v
[5. Retrieve & Generate] ─ For each QA pair:
        |                    a. Pick next topic path from tree
        |                    b. Use query seed for vector search → get diverse contexts
        |                    c. LLM generates 3 QA pairs per call (batch)
        |                    d. Grounding check: answer must come from contexts
        v
   dataset.json (RAGAS format)
```

### How QA generation works

The key principle: **contexts are truth, topic is guidance.**

1. The topic tree suggests what *angle* to ask about
2. The query seed retrieves the most relevant document chunks
3. The LLM generates questions that can only be answered from those chunks
4. A grounding check verifies the answer is semantically derived from the contexts (not the LLM's general knowledge)

### How context diversity works

The pipeline tracks which chunks have been used and excludes them from subsequent retrievals. This ensures each QA pair covers a different part of the document. When 70% of chunks have been used, the pool resets.

## Installation

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

Or copy `.env.example` to `.env` and fill in your key.

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
    --batch-size 3 \
    --chunk-size 500 \
    --top-k 5 \
    --model gpt-4o \
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

All settings can be configured via `.env` file (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `TOP_K` | 5 | Chunks retrieved per QA pair |
| `NUM_SAMPLES` | 100 | Target QA pairs |
| `MODEL_NAME` | gpt-4o | LLM for QA generation |
| `TOPIC_MODEL` | gpt-4o-mini | LLM for topic tree + query seeds (cheaper) |
| `EMBEDDING_MODEL` | text-embedding-3-small | Embedding model |
| `BATCH_SIZE` | 3 | QA pairs per LLM call |
| `TREE_DEGREE` | 3 | Children per topic node |
| `TREE_DEPTH` | 3 | Topic tree depth |
| `TEMPERATURE` | 0.7 | Generation temperature |
| `MAX_RETRIES` | 3 | Retries per failed generation |
| `DOCUMENT_DOMAIN` | general | Domain focus (health, nutrition, etc.) |
| `LANGUAGE` | pt-BR | Output language for QA pairs |
| `DEDUP_OVERLAP_THRESHOLD` | 0.8 | Chunk dedup word-overlap threshold |

### CLI Arguments

```
python main.py ./documents \
    --chunk-size 500 \
    --chunk-overlap 100 \
    --top-k 5 \
    --num-samples 100 \
    --model gpt-4o \
    --embedding-model text-embedding-3-small \
    --temperature 0.7 \
    --batch-size 3 \
    --tree-degree 3 \
    --tree-depth 3 \
    --output dataset.json
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

## Project Structure

```
├── main.py            # Pipeline orchestration & CLI
├── config.py          # All configuration (dataclass + env vars)
├── models.py          # Pydantic models (QAPair, DatasetEntry)
├── loader.py          # Document loading (PDF, TXT, MD)
├── chunker.py         # Text chunking with quality filtering
├── embedder.py        # OpenAI embeddings + EmbeddingCache
├── vector_store.py    # FAISS vector store with diversity filtering
├── topic_tree.py      # Hierarchical topic tree + query seed generation
├── generator.py       # Pluto-style batch QA generation + grounding check
├── tests/             # Test suite
├── requirements.txt   # Dependencies
└── .env.example       # Example environment configuration
```
