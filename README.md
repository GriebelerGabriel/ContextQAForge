# RAG QA Dataset Generator

A production-ready Python pipeline for generating high-quality QA datasets for evaluating Retrieval-Augmented Generation (RAG) systems using Pluto-style structured prompt engineering.

## Features

- **Multi-format document loading**: Supports TXT, PDF, and Markdown files
- **Intelligent chunking**: Configurable chunk size and overlap with sentence boundary detection
- **FAISS vector store**: Efficient similarity search for context retrieval
- **Pluto-style QA generation**: Structured prompting with few-shot examples and diversity controls
- **RAGAS-compatible output**: JSON format ready for RAG evaluation
- **Quality controls**: Duplicate detection, retry logic, and generic question filtering

## Installation

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

Or copy `.env.example` to `.env` and fill in your credentials.

## Quick Start

```bash
python main.py /path/to/documents --num-samples 50 --output dataset.json
```

## Usage

### Basic Usage

```bash
python main.py ./documents
```

### Advanced Options

```bash
python main.py ./documents \
    --chunk-size 1500 \
    --chunk-overlap 300 \
    --top-k 7 \
    --num-samples 200 \
    --model gpt-4.1 \
    --output custom_dataset.json
```

### Python API

```python
from config import PipelineConfig
from main import run_pipeline, save_dataset
import logging

config = PipelineConfig(
    chunk_size=1000,
    chunk_overlap=200,
    top_k=5,
    num_samples=50,
    model_name="gpt-4o",
    openai_api_key="your-api-key"
)

logger = logging.getLogger(__name__)
dataset = run_pipeline("./documents", config, logger)
save_dataset(dataset, "output.json", logger)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Characters per chunk |
| `chunk_overlap` | 200 | Overlap between chunks |
| `top_k` | 5 | Number of chunks to retrieve |
| `num_samples` | 100 | QA pairs to generate |
| `model_name` | gpt-4o | OpenAI chat model |
| `embedding_model` | text-embedding-3-small | OpenAI embedding model |
| `temperature` | 0.7 | Generation temperature |

## Output Format

```json
[
  {
    "question": "What company did the creator of Python work for during the 2000s?",
    "ground_truth": "Guido van Rossum, the creator of Python, worked at Google from 2005 to 2012.",
    "contexts": [
      "Python was created by Guido van Rossum...",
      "Guido van Rossum worked at Google..."
    ],
    "metadata": {
      "type": "multi-hop",
      "difficulty": "medium"
    }
  }
]
```

## QA Types

- **single-hop**: Direct lookup from one context
- **multi-hop**: Requires connecting multiple contexts
- **inference**: Requires reasoning from contexts
- **unanswerable**: Cannot be answered from contexts
- **paraphrase**: Uses different wording than context

## Project Structure

```
.
├── config.py          # Configuration management
├── models.py          # Pydantic data models
├── loader.py          # Document loading (txt, pdf, md)
├── chunker.py         # Text chunking
├── embedder.py        # OpenAI embeddings
├── vector_store.py    # FAISS storage
├── generator.py       # Pluto-style QA generation
├── main.py            # Pipeline orchestration
└── requirements.txt   # Dependencies
```
