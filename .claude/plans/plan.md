# Plan: Restructure Pipeline — PaddleOCR + Document-Grounded Tree + QA Generation

## Problem

Current pipeline has three major issues:
1. **PDF parsing** uses `pdfplumber` (text-only, no layout awareness, loses tables/structure)
2. **Topic tree** is 100% LLM-generated — not grounded in actual document structure, may miss/bias topics
3. **Vector-store retrieval** biases toward first documents; no guarantee of full document coverage

## Goal

Redesign the pipeline to:
- Use **PaddleOCR PPStructureV3** for PDF parsing with layout detection, table recognition, and heading hierarchy
- Build a **document-grounded JSON topic tree** extracted from actual document headings/sections (not LLM)
- Associate **chunks directly with tree nodes** (content from each section)
- **Guarantee coverage of ALL documents** via round-robin iteration across tree leaf nodes

---

## New Pipeline Architecture

```
Step 1: PARSE (PaddleOCR PPStructureV3)
  PDFs → structured Markdown with headings (# H1, ## H2, ### H3), tables, paragraphs
  TXT/MD → used directly
  Supports both English and Portuguese (configurable via paddleocr_lang)

Step 2: BUILD TREE (heading skeleton + LLM refinement)
  2a. Extract heading hierarchy from structured Markdown → initial skeleton tree
  2b. LLM refines the tree: splits long/vague sections into sub-topics based on actual content,
      validates headings match content, enriches where structure is thin
  2c. Each leaf node holds its section's text content

Step 3: CHUNK PER LEAF NODE
  For each leaf node, split its section content into smaller chunks (e.g., 500-1000 chars)
  Chunks are scoped per leaf node (not per whole document)
  Multiple small chunks per leaf → used as contexts for RAGAS

Step 4: GENERATE QA (LLM)
  Round-robin through ALL leaf nodes across ALL documents
  Pick 1-3 chunks from the current leaf node's chunk pool as context
  Distribute num_samples evenly across leaf nodes, cycle if needed
```

---

## Detailed Steps

### Step 1: Replace PDF loader with PaddleOCR (`loader.py`)

**Changes to `loader.py`:**
- Replace `pdfplumber` with `paddleocr.PPStructureV3`
- New `_load_pdf_with_paddleocr()` function:
  - Initialize `PPStructureV3(lang=config.paddleocr_lang, use_table_recognition=True)`
  - `paddleocr_lang` supports `"en"`, `"pt"`, `"latin"` — configurable per run (default: `"en"`)
  - Call `pipe.predict(pdf_path)` to get structured results per page
  - Use `restructure_pages(merge_tables=True, relevel_titles=True, concatenate_pages=True)` to get full doc
  - Call `save_to_markdown()` to get structured Markdown output
  - Return structured Markdown as `Document.content`
- Keep `_load_text_file()` unchanged for TXT/MD
- Keep `_clean_pdf_text()` for post-processing if needed
- Cache parsed Markdown to `tree/parsed/` directory (avoid re-parsing on reruns)

**New dependencies in `requirements.txt`:**
- `paddleocr` (or `paddleocr[doc-parser]`)
- `paddlepaddle` (CPU version)

### Step 2: Document-grounded topic tree (`topic_tree.py` — major rewrite)

**Hybrid approach: heading skeleton + LLM refinement**

#### Phase 2a: Extract heading skeleton (no LLM)

1. **Parse heading hierarchy** from each document's structured Markdown:
   - `# Title` → depth 0
   - `## Section` → depth 1
   - `### Subsection` → depth 2
   - etc.
   - Content between headings belongs to the preceding heading node
   - This gives us an initial tree structure grounded in actual document layout

#### Phase 2b: LLM refinement (enriches the skeleton)

2. **LLM refines each section that needs it:**
   - **Long sections without subheadings**: LLM reads the content and splits into 2-5 sub-topics. Each sub-topic gets its own node with the relevant content excerpt.
   - **Vague headings**: LLM reads the section content and rewrites the heading to better reflect the actual content.
   - **Missing granularity**: If a section covers multiple distinct topics, LLM identifies and creates child nodes for each.
   - Uses the cheaper `topic_model` (gpt-4o-mini) for this step.
   - **Preserves the heading skeleton** — only enriches, never removes or restructures the base tree.

3. **Build JSON tree structure:**
   ```json
   {
     "name": "Document Topics",
     "children": [
       {
         "name": "Dietary Sodium Intake",
         "source": "9789241504836_eng.pdf",
         "children": [
           {
             "name": "Recommended Sodium Levels",
             "source": "9789241504836_eng.pdf",
             "content": "Actual text content from this section...",
             "children": []
           }
         ]
       },
       {
         "name": "Cardiovascular Health Guidelines",
         "source": "lichtenstein-et-al-2026.pdf",
         "children": [...]
       }
     ]
   }
   ```

4. **Key design:**
   - Root node groups all documents
   - Each document contributes its section hierarchy as subtrees
   - Leaf nodes contain the raw section text (can be large)
   - Track `source` (filename) at each node for provenance
   - Heading extraction is free (no API cost), LLM refinement is targeted (only sections that need it)
   - Tree is saved as JSON for inspection and reuse

5. **Tree iteration for QA generation:**
   - `get_all_leaf_nodes()` → returns all leaf nodes with their content, grouped by source
   - `get_qa_distribution(num_samples)` → distributes QA count across leaves evenly
     - e.g., 30 leaves + 10 QA = 1 QA per 3 leaves (round-robin through all)
     - e.g., 5 leaves + 10 QA = 2 QA per leaf
   - Guarantees every document section gets covered

### Step 3: Chunk per leaf node (`chunker.py` — keep and adapt)

**Changes:**
- `chunker.py` stays but chunks are scoped **per leaf node**, not per whole document
- Each leaf node's section content is split into smaller chunks using the same character-based approach
- Default: 500-1000 chars with overlap (configurable)
- Multiple chunks per leaf → stored in the leaf node
- These chunks become the `contexts` in RAGAS dataset entries

**Why not the whole section?**
- A section can span multiple pages (5000+ chars) — too big for LLM context and RAGAS evaluation
- Smaller chunks = better QA specificity, better RAGAS context relevance scores
- Scoping per leaf ensures chunks stay topically coherent

### Step 4: QA Generation (`generator.py` — adapt)

**Changes:**
- Remove dependency on topic tree's `get_next_topic_path()` + vector store retrieval
- Instead, accept a leaf node's chunks directly as context
- Add `generate_qa_from_section()` method:
  - Takes leaf node's chunks + topic path (from tree) + source document name
  - Picks 1-3 chunks from the leaf's pool as context for each QA pair
  - Generates QA pairs grounded in those specific chunks
  - Uses same quality filters (generic detection, grounding check)
- Distribution logic moves to `main.py` (round-robin across leaves)

### Step 5: Simplify pipeline (`main.py` — restructure)

**New pipeline flow:**
```python
def run_pipeline(documents_folder, config):
    # 1. Load & parse documents (PaddleOCR for PDFs)
    documents = load_documents(documents_folder, ...)

    # 2. Build document-grounded topic tree (heading skeleton + LLM refinement)
    tree = DocumentTopicTree()
    tree.build_from_documents(documents)
    tree.save("tree/topic_tree.json")  # persist for inspection

    # 3. Chunk per leaf node (section content → smaller chunks)
    tree.chunk_leaves(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

    # 4. Get QA distribution across leaf nodes
    distribution = tree.get_qa_distribution(config.num_samples)

    # 5. Generate QA pairs (round-robin across all leaves)
    generator = QAGenerator(config)
    dataset = []
    for leaf, count in distribution:
        qa_pairs = generator.generate_qa_from_section(
            chunks=leaf.chunks,         # small chunks from this leaf
            topic_path=leaf.path,
            source=leaf.source,
            num_pairs=count
        )
        dataset.extend(qa_pairs)

    return dataset
```

**What gets removed:**
- `embedder.py` — no longer needed (no vector store)
- `vector_store.py` — no longer needed (tree-grounded chunks replace vector retrieval)
- LLM-based topic generation in `topic_tree.py`

### Step 5: Update config, models, tests

**`config.py`:**
- Add `paddleocr_lang` config (default: `"en"`, supports `"pt"` for Portuguese)
- Add `use_table_recognition` config (default: True)
- Remove vector-store-related configs (`top_k`, `embedding_model`, `dedup_overlap_threshold`)
- Keep `topic_model` for LLM refinement of tree (gpt-4o-mini)
- Keep QA generation configs (`model_name`, `num_samples`, `batch_size`, etc.)

**`models.py`:**
- Add `SectionNode` model for tree JSON structure:
  ```python
  class SectionNode(BaseModel):
      name: str
      source: str
      depth: int
      content: Optional[str] = None  # raw section text (leaf nodes)
      chunks: List[str] = []          # smaller chunks after chunking step
      children: List["SectionNode"] = []
  ```

**Tests:**
- Update `test_loader.py` for PaddleOCR (mock PPStructureV3)
- Rewrite `test_topic_tree.py` for document-structure extraction
- Update `test_generator.py` for section-based generation
- Remove `test_vector_store.py`

---

## File Change Summary

| File | Action |
|------|--------|
| `loader.py` | **Major rewrite** — PaddleOCR PPStructureV3 for PDFs |
| `topic_tree.py` | **Major rewrite** — heading skeleton + LLM refinement, JSON tree |
| `chunker.py` | **Keep & adapt** — chunks scoped per leaf node instead of per document |
| `main.py` | **Rewrite pipeline** — remove vector store, use tree-based distribution |
| `generator.py` | **Adapt** — section-based QA generation using leaf chunks |
| `config.py` | **Update** — add PaddleOCR configs, remove vector store configs |
| `models.py` | **Update** — add SectionNode model with chunks field |
| `requirements.txt` | **Update** — add paddleocr, remove faiss-cpu |
| `embedder.py` | **Remove** (no longer needed) |
| `vector_store.py` | **Remove** (no longer needed) |
| Tests | **Update** to match new architecture |

---

## Implementation Order

1. **`models.py`** — Add `SectionNode` model
2. **`loader.py`** — PaddleOCR integration, structured Markdown output
3. **`topic_tree.py`** — Rewrite as `DocumentTopicTree` (heading extraction → JSON tree)
4. **`config.py`** — Update configs
5. **`generator.py`** — Adapt for section-based QA generation
6. **`main.py`** — New pipeline orchestration
7. **`requirements.txt`** — Update dependencies
8. **Tests** — Update all test files

---

## Visual Flow Example

Tracing a single document through the pipeline end-to-end.

---

### Input: `guia_cardiovascular.pdf`

A Portuguese PDF about cardiovascular nutrition guidelines.

---

### Step 1 — PaddleOCR Parsing

**Input:** `guia_cardiovascular.pdf` (12 pages, mixed text + tables)

**PaddleOCR PPStructureV3** detects layout, headings, tables, paragraphs and produces:

**Output:** Structured Markdown (`tree/parsed/guia_cardiovascular.md`)

```markdown
# Guia de Nutrição Cardiovascular

## Introdução

As doenças cardiovasculares são a principal causa de morte no Brasil.
Estudos recentes mostram que a dieta desempenha um papel fundamental
na prevenção e no manejo dessas condições. Este guia apresenta
recomendações baseadas em evidências para profissionais de saúde.

## Recomendações Nutricionais

### Redução de Sódio

A Organização Mundial de Saúde recomenda que o consumo de sódio
não ultrapasse 2g por dia para adultos. No Brasil, o consumo médio
é de 4.1g por dia, o dobro do recomendado.

| Alimento        | Sódio (mg/porção) |
|-----------------|-------------------|
| Pão francês     | 540               |
| Queijo mussarela| 420               |
| Presunto        | 980               |

A redução do sódio pode diminuir a pressão arterial em 5-6 mmHg.

### Gorduras e Colesterol

O consumo de gorduras saturadas deve ser limitado a menos de 10%
das calorias totais diárias. Recomenda-se substituir por gorduras
insaturadas presentes em azeite, abacate e peixes.

### Consumo de Fibras

A ingestão diária recomendada de fibras é de 25-30g. Estudos mostram
que cada 7g de fibra adicional reduz o risco cardiovascular em 9%.

## Populações Especiais

### Pacientes com Hipertensão

Para pacientes hipertensos, a dieta DASH é recomendada como primeira
linha de tratamento. Esta dieta enfatiza frutas, vegetais, laticínios
com baixo teor de gordura e redução de sódio para 1.5g/dia.

### Pacientes com Diabetes

Pacientes diabéticos devem seguir as recomendações gerais com atenção
especial ao índice glicêmico dos alimentos e ao controle de porções.
```

---

### Step 2a — Heading Skeleton (no LLM, free)

Parser extracts heading hierarchy and assigns content to each node:

```
Root: Document Topics
├── Guia de Nutrição Cardiovascular           ← source: guia_cardiovascular.pdf
│   ├── Introdução                            ← content: "As doenças cardiovasculares..."
│   ├── Recomendações Nutricionais
│   │   ├── Redução de Sódio                  ← content: "A OMS recomenda... 5-6 mmHg."
│   │   ├── Gorduras e Colesterol             ← content: "O consumo de gorduras..."
│   │   └── Consumo de Fibras                 ← content: "A ingestão diária..."
│   └── Populações Especiais
│       ├── Pacientes com Hipertensão         ← content: "Para pacientes hipertensos..."
│       └── Pacientes com Diabetes            ← content: "Pacientes diabéticos..."
```

**Leaf nodes** (will hold content): `Introdução`, `Redução de Sódio`, `Gorduras e Colesterol`, `Consumo de Fibras`, `Pacientes com Hipertensão`, `Pacientes com Diabetes`

---

### Step 2b — LLM Refinement (gpt-4o-mini)

LLM reviews each leaf node. Example for **Redução de Sódio** (section has a table + recommendations — could be split):

```
BEFORE (single leaf):
  Redução de Sódio  →  [all 400 chars of content including table + recommendation]

AFTER (LLM splits into sub-topics):
  Redução de Sódio
  ├── Recomendações de Ingestão de Sódio  →  "A OMS recomenda... dobro do recomendado."
  ├── Tabela de Sódio por Alimento        →  [table content]
  └── Impacto na Pressão Arterial         →  "A redução do sódio pode diminuir..."
```

**Tree after refinement:**

```
Root: Document Topics
├── Guia de Nutrição Cardiovascular
│   ├── Introdução                            ← 250 chars
│   ├── Recomendações Nutricionais
│   │   ├── Redução de Sódio
│   │   │   ├── Recomendações de Ingestão     ← 180 chars
│   │   │   ├── Tabela de Sódio por Alimento  ← 120 chars
│   │   │   └── Impacto na Pressão Arterial   ← 100 chars
│   │   ├── Gorduras e Colesterol             ← 200 chars
│   │   └── Consumo de Fibras                 ← 180 chars
│   └── Populações Especiais
│       ├── Pacientes com Hipertensão         ← 220 chars
│       └── Pacientes com Diabetes            ← 190 chars
```

9 leaf nodes total, each with manageable content.

---

### Step 3 — Chunk per Leaf Node

Leaf nodes with content > `chunk_size` get split. Small leaves stay as one chunk.

```
Introdução (250 chars)              → 1 chunk
Recomendações de Ingestão (180)     → 1 chunk
Tabela de Sódio por Alimento (120)  → 1 chunk
Impacto na Pressão Arterial (100)   → 1 chunk
Gorduras e Colesterol (200)         → 1 chunk
Consumo de Fibras (180)             → 1 chunk
Pacientes com Hipertensão (220)     → 1 chunk
Pacientes com Diabetes (190)        → 1 chunk
```

(With a larger document, sections could be 2000+ chars → split into 3-4 chunks each)

**Tree JSON saved to `tree/topic_tree.json`:**

```json
{
  "name": "Document Topics",
  "children": [
    {
      "name": "Guia de Nutrição Cardiovascular",
      "source": "guia_cardiovascular.pdf",
      "depth": 0,
      "children": [
        {
          "name": "Introdução",
          "source": "guia_cardiovascular.pdf",
          "depth": 1,
          "content": "As doenças cardiovasculares são a principal causa de morte no Brasil. Estudos recentes mostram que a dieta desempenha um papel fundamental na prevenção e no manejo dessas condições. Este guia apresenta recomendações baseadas em evidências para profissionais de saúde.",
          "chunks": [
            "As doenças cardiovasculares são a principal causa de morte no Brasil. Estudos recentes mostram que a dieta desempenha um papel fundamental na prevenção e no manejo dessas condições. Este guia apresenta recomendações baseadas em evidências para profissionais de saúde."
          ],
          "children": []
        },
        {
          "name": "Recomendações Nutricionais",
          "source": "guia_cardiovascular.pdf",
          "depth": 1,
          "children": [
            {
              "name": "Redução de Sódio",
              "source": "guia_cardiovascular.pdf",
              "depth": 2,
              "children": [
                {
                  "name": "Recomendações de Ingestão de Sódio",
                  "source": "guia_cardiovascular.pdf",
                  "depth": 3,
                  "content": "A Organização Mundial de Saúde recomenda que o consumo de sódio não ultrapasse 2g por dia para adultos. No Brasil, o consumo médio é de 4.1g por dia, o dobro do recomendado.",
                  "chunks": ["A Organização Mundial de Saúde recomenda que o consumo de sódio não ultrapasse 2g por dia para adultos. No Brasil, o consumo médio é de 4.1g por dia, o dobro do recomendado."],
                  "children": []
                },
                {
                  "name": "Tabela de Sódio por Alimento",
                  "source": "guia_cardiovascular.pdf",
                  "depth": 3,
                  "content": "| Alimento | Sódio (mg/porção) | |---|---| | Pão francês | 540 | | Queijo mussarela | 420 | | Presunto | 980 |",
                  "chunks": ["| Alimento | Sódio (mg/porção) | ..."],
                  "children": []
                },
                {
                  "name": "Impacto na Pressão Arterial",
                  "source": "guia_cardiovascular.pdf",
                  "depth": 3,
                  "content": "A redução do sódio pode diminuir a pressão arterial em 5-6 mmHg.",
                  "chunks": ["A redução do sódio pode diminuir a pressão arterial em 5-6 mmHg."],
                  "children": []
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

---

### Step 4 — QA Generation (LLM)

**User requests `--num-samples 10`**

9 leaf nodes, 10 QA pairs requested → round-robin distribution:

```
Leaf #1  Introdução                      → 2 QA pairs
Leaf #2  Recomendações de Ingestão       → 1 QA pair
Leaf #3  Tabela de Sódio                 → 1 QA pair
Leaf #4  Impacto na Pressão              → 1 QA pair
Leaf #5  Gorduras e Colesterol           → 1 QA pair
Leaf #6  Consumo de Fibras               → 1 QA pair
Leaf #7  Pacientes com Hipertensão       → 1 QA pair
Leaf #8  Pacientes com Diabetes          → 1 QA pair
Leaf #9  (extra QA cycles back to #1)    → 1 QA pair
                                         ─────────
                                         10 total
```

**Example QA pair from leaf "Recomendações de Ingestão de Sódio":**

```json
{
  "question": "Qual é o consumo médio de sódio no Brasil e como ele se compara ao recomendado pela OMS?",
  "answer": "O consumo médio de sódio no Brasil é de 4.1g por dia, o dobro do limite de 2g por dia recomendado pela Organização Mundial de Saúde para adultos.",
  "contexts": [
    "A Organização Mundial de Saúde recomenda que o consumo de sódio não ultrapasse 2g por dia para adultos. No Brasil, o consumo médio é de 4.1g por dia, o dobro do recomendado."
  ],
  "ground_truth": "O consumo médio de sódio no Brasil é de 4.1g por dia, o dobro do limite de 2g por dia recomendado pela Organização Mundial de Saúde para adultos."
}
```

**Example QA pair from leaf "Pacientes com Hipertensão":**

```json
{
  "question": "Qual dieta é recomendada como primeira linha de tratamento para pacientes hipertensos?",
  "answer": "A dieta DASH é recomendada como primeira linha de tratamento para pacientes hipertensos. Ela enfatiza frutas, vegetais, laticínios com baixo teor de gordura e redução de sódio para 1.5g por dia.",
  "contexts": [
    "Para pacientes hipertensos, a dieta DASH é recomendada como primeira linha de tratamento. Esta dieta enfatiza frutas, vegetais, laticínios com baixo teor de gordura e redução de sódio para 1.5g/dia."
  ],
  "ground_truth": "A dieta DASH é recomendada como primeira linha de tratamento para pacientes hipertensos. Ela enfatiza frutas, vegetais, laticínios com baixo teor de gordura e redução de sódio para 1.5g por dia."
}
```

---

### Final Output: `dataset.json`

All 10 QA entries saved, each grounded in content from different sections. Every section of the document was covered. Ready for RAGAS evaluation.

---

### With Multiple Documents

If there are 3 documents with 9, 12, and 8 leaf nodes respectively (29 total):

```
--num-samples 30  →  1 QA per leaf node, cycling back to #1 for the 30th
--num-samples 58  →  2 QA per leaf node
--num-samples 10  →  1 QA every ~3rd leaf, still covering all 3 documents
```

Every document contributes proportionally. No document gets skipped.

---

## Key Guarantees

- **Full document coverage**: Round-robin through ALL leaf nodes across ALL documents. No document gets skipped.
- **Even distribution**: `num_samples` distributed across leaf nodes. If 5 leaves + 10 QA → 2 per leaf. If 30 leaves + 10 QA → cycle through every 3rd leaf.
- **Document-grounded tree**: Topics come from actual headings in documents, refined by LLM where the structure is thin or vague. Best of both worlds.
- **Chunk association**: Each leaf node has its section content directly — no separate chunking step needed.
- **Multi-language**: PaddleOCR supports `lang="pt"` for Portuguese documents. Configurable per run.
