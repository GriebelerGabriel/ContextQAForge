"""Main pipeline orchestration for RAG QA dataset generation.

Pipeline: Load → Build Tree → Chunk Leaves → Distribute QA → Generate → Save
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import PipelineConfig
from embedder import Embedder
from generator import QAGenerator
from loader import load_documents
from models import DatasetEntry, SectionNode
from topic_tree import DocumentTopicTree


def setup_logging(log_path: str = "pipeline.log") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def run_pipeline(
    documents_folder: str,
    config: PipelineConfig,
    logger: logging.Logger,
    qa_only: bool = False,
) -> List[DatasetEntry]:
    """
    Run the complete RAG QA generation pipeline.

    Args:
        documents_folder: Path to folder containing documents
        config: Pipeline configuration
        logger: Logger instance
        qa_only: If True, skip document loading and tree building, load cached tree only.

    Returns:
        List of dataset entries
    """
    logger.info("=" * 60)
    logger.info("RAG QA Dataset Generation Pipeline")
    logger.info("=" * 60)

    tree = DocumentTopicTree(config)

    if qa_only:
        # QA-only mode: load cached tree, skip document loading and tree building
        if not Path(config.tree_path).exists():
            raise FileNotFoundError(
                f"No cached tree found at {config.tree_path}. "
                "Run the full pipeline first to build the tree."
            )
        logger.info(f"QA-only mode: loading cached tree from {config.tree_path}")
        tree.load(config.tree_path)
    else:
        # Step 1: Load & parse documents (PaddleOCR for PDFs)
        logger.info(f"Loading documents from: {documents_folder}")
        logger.info(f"PaddleOCR language: {config.paddleocr_lang}")
        documents = load_documents(
            documents_folder,
            paddleocr_lang=config.paddleocr_lang,
            use_table_recognition=config.use_table_recognition,
            pdf_remove_patterns=config.pdf_remove_patterns,
            parsed_cache_dir=config.parsed_path,
        )
        logger.info(f"Loaded {len(documents)} documents")

        if not documents:
            raise ValueError("No documents found in the specified folder")

        for doc in documents:
            logger.info(f"  - {doc.source}: {len(doc.content)} chars ({doc.doc_type})")

        # Step 2: Build document-grounded topic tree (heading skeleton + LLM refinement)
        logger.info("Building document-grounded topic tree")
        tree.build_from_documents(documents)

        # Visualize before chunking
        tree_viz = tree.visualize()
        logger.info(tree_viz)

        # Step 3: Chunk per leaf node
        logger.info(
            f"Chunking leaf nodes (size={config.chunk_size}, overlap={config.chunk_overlap})"
        )
        tree.chunk_leaves(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

        leaves = tree.get_all_leaf_nodes()
        total_chunks = sum(len(leaf.chunks) for leaf in leaves)
        logger.info(f"Tree has {len(leaves)} leaf nodes, {total_chunks} total chunks")

        # Save tree JSON (after chunking so chunks are persisted)
        tree.save(config.tree_path)
        logger.info(f"Tree saved to {config.tree_path}")

    leaves = tree.get_all_leaf_nodes()
    total_chunks = sum(len(leaf.chunks) for leaf in leaves)
    logger.info(f"Tree has {len(leaves)} leaf nodes, {total_chunks} total chunks")

    # Save tree JSON (after chunking so chunks are persisted)
    tree.save(config.tree_path)
    logger.info(f"Tree saved to {config.tree_path}")

    # Step 4: Distribute QA pairs across documents, then leaves
    all_leaves = tree.get_all_leaf_nodes()

    # Group leaves by source document
    doc_leaves: dict = {}
    for leaf in all_leaves:
        doc_leaves.setdefault(leaf.source, []).append(leaf)

    num_docs = len(doc_leaves)
    doc_list = list(doc_leaves.keys())

    if getattr(config, "balance_per_document", True) and num_docs > 0:
        # Balanced mode: equal share per document, remainder via round-robin
        per_doc = config.num_samples // num_docs
        doc_remainder = config.num_samples % num_docs

        distribution: List[Tuple] = []
        for doc_idx, source in enumerate(doc_list):
            leaves = doc_leaves[source]
            target = per_doc + (1 if doc_idx < doc_remainder else 0)
            if not leaves or target <= 0:
                continue
            # Distribute within document's leaves weighted by content length
            content_weights = [max(len(leaf.content or ""), 100) for leaf in leaves]
            total_weight = sum(content_weights)
            for leaf_idx, leaf in enumerate(leaves):
                weight = content_weights[leaf_idx]
                count = max(1, round(target * weight / total_weight))
                distribution.append((leaf, count))
    else:
        # Standard mode: round-robin across all leaves (original behavior)
        distribution = tree.get_qa_distribution(config.num_samples)

    logger.info(
        f"Distributing {config.num_samples} QA pairs across {len(doc_list)} documents, "
        f"{len(distribution)} leaf nodes"
    )
    for source in doc_list:
        leaves = doc_leaves[source]
        doc_total = sum(c for l, c in distribution if l.source == source)
        logger.info(f"  {source[:60]}: {len(leaves)} leaves, {doc_total} QA pairs")

    # Step 5: Generate QA pairs
    logger.info(f"Generating QA pairs using {config.model_name}...")
    generator = QAGenerator(config)

    # Pre-compute embeddings for embedding-based related-chunk matching
    embedder = Embedder(config)
    leaf_titles = [leaf.name for leaf in all_leaves]
    title_embeddings = embedder.embed_chunks(leaf_titles) if leaf_titles else np.array([])

    dataset: List[DatasetEntry] = []

    # Pre-build leaf index lookup and production counter
    leaf_to_idx = {id(leaf): i for i, leaf in enumerate(all_leaves)}
    leaf_produced: dict = {}  # (source, name) -> count

    # Keep generating until we hit the target (or exhaust max_total_attempts)
    max_total_attempts = config.num_samples * 5
    total_attempts = 0
    consecutive_stall_passes = 0
    max_stall_passes = 3  # break if 3 full passes produce zero new pairs

    while len(dataset) < config.num_samples and total_attempts < max_total_attempts:
        progress_before_pass = len(dataset)

        for leaf, count in distribution:
            if len(dataset) >= config.num_samples:
                break

            if not leaf.content and not leaf.chunks:
                continue

            leaf_key = (leaf.source, leaf.name)
            produced = leaf_produced.get(leaf_key, 0)
            remaining_for_leaf = count - produced
            if remaining_for_leaf <= 0:
                continue

            topic_path = tree.get_leaf_path(leaf)

            leaf_idx = leaf_to_idx[id(leaf)]
            related_chunks = _find_related_chunks_embedding(
                leaf_idx, all_leaves, title_embeddings, max_chunks=3,
            )
            full_content = leaf.content or ""
            contexts = [full_content] if full_content else leaf.chunks
            if not contexts:
                continue

            if remaining_for_leaf > 1 and config.batch_size > 1:
                batch_count = min(remaining_for_leaf, config.batch_size)
                qa_pairs = generator.generate_qa_from_section(
                    chunks=contexts,
                    topic_path=topic_path,
                    source=leaf.source,
                    num_pairs=batch_count,
                    full_content=full_content,
                    related_chunks=related_chunks,
                )
                for qa_pair in qa_pairs:
                    if len(dataset) < config.num_samples:
                        entry = generator.to_dataset_entry(qa_pair)
                        dataset.append(entry)
                        leaf_produced[leaf_key] = leaf_produced.get(leaf_key, 0) + 1
            else:
                qa_pairs = generator.generate_qa_from_section(
                    chunks=contexts,
                    topic_path=topic_path,
                    source=leaf.source,
                    num_pairs=1,
                    full_content=full_content,
                    related_chunks=related_chunks,
                )
                for qa_pair in qa_pairs:
                    if len(dataset) < config.num_samples:
                        entry = generator.to_dataset_entry(qa_pair)
                        dataset.append(entry)
                        leaf_produced[leaf_key] = leaf_produced.get(leaf_key, 0) + 1

            total_attempts += 1

            if len(dataset) % 5 == 0 or len(dataset) >= config.num_samples:
                logger.info(f"  Progress: {len(dataset)}/{config.num_samples} QA pairs (attempts: {total_attempts})")

        # Stall detection: if a full pass produced nothing, count it
        if len(dataset) == progress_before_pass:
            consecutive_stall_passes += 1
            logger.info(
                f"  Stall: no progress this pass ({consecutive_stall_passes}/{max_stall_passes})"
            )
            if consecutive_stall_passes >= max_stall_passes:
                logger.info(
                    f"  Breaking: {max_stall_passes} consecutive passes with no new pairs"
                )
                break
        else:
            consecutive_stall_passes = 0

    logger.info(f"Successfully generated {len(dataset)} QA pairs")
    return dataset


def _find_related_chunks_embedding(
    leaf_idx: int,
    all_leaves: List[SectionNode],
    title_embeddings: np.ndarray,
    max_chunks: int = 3,
    similarity_threshold: float = 0.5,
) -> List[str]:
    """Find related content from other documents using embedding cosine similarity.

    Converts leaf titles to vectors and checks cosine similarity.
    This handles synonyms (e.g., "Vitamin C" matches "Ascorbic Acid")
    because the embedding model understands semantics.
    """
    if title_embeddings.size == 0 or leaf_idx >= len(title_embeddings):
        return []

    query_emb = title_embeddings[leaf_idx]
    query_norm = np.linalg.norm(query_emb)
    if query_norm == 0:
        return []
    query_emb = query_emb / query_norm

    related: List[str] = []
    scored: List[Tuple[float, int]] = []

    for i, other in enumerate(all_leaves):
        if i == leaf_idx:
            continue
        if other.source == all_leaves[leaf_idx].source:
            continue  # skip same document
        if not other.content:
            continue

        other_emb = title_embeddings[i]
        other_norm = np.linalg.norm(other_emb)
        if other_norm == 0:
            continue
        other_emb = other_emb / other_norm

        similarity = float(np.dot(query_emb, other_emb))
        if similarity >= similarity_threshold:
            scored.append((similarity, i))

    # Sort by similarity descending, take top max_chunks
    scored.sort(key=lambda x: -x[0])
    for _, idx in scored[:max_chunks]:
        related.append(all_leaves[idx].content)

    return related


def save_dataset(dataset: List[DatasetEntry], output_path: str, logger: logging.Logger) -> None:
    """Save dataset to JSON file."""
    output = [entry.model_dump() for entry in dataset]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Dataset saved to: {output_path}")


def _esc(text: str) -> str:
    """HTML-escape text."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def save_dataset_html(dataset: List[DatasetEntry], output_path: str, logger: logging.Logger) -> None:
    """Save dataset as a human-readable HTML file."""
    html_path = Path(output_path).with_suffix(".html")

    # Stats
    by_source = {}
    by_type = {}
    by_diff = {}
    for e in dataset:
        s = Path(e.source).stem
        by_source[s] = by_source.get(s, 0) + 1
        t = e.metadata.get("type", "-")
        by_type[t] = by_type.get(t, 0) + 1
        d = e.metadata.get("difficulty", "-")
        by_diff[d] = by_diff.get(d, 0) + 1

    # Type badge colors
    type_colors = {
        "single-hop": "#3b82f6", "multi-hop": "#8b5cf6", "inference": "#f59e0b",
        "paraphrase": "#10b981", "true-false": "#ef4444",
    }
    diff_colors = {"easy": "#22c55e", "medium": "#f59e0b", "hard": "#ef4444"}

    cards = ""
    for i, entry in enumerate(dataset, 1):
        src = Path(entry.source).stem
        t = entry.metadata.get("type", "-")
        d = entry.metadata.get("difficulty", "-")
        topic = " > ".join(entry.metadata.get("topic_path", [])[-3:])
        tc = type_colors.get(t, "#6b7280")
        dc = diff_colors.get(d, "#6b7280")

        ctx_html = ""
        if entry.contexts:
            ctx_items = "".join(
                f'<div class="ctx-item"><strong>Context {j}:</strong><p>{_esc(ctx)}</p></div>'
                for j, ctx in enumerate(entry.contexts, 1)
            )
            ctx_html = f'<details><summary>Show contexts ({len(entry.contexts)})</summary><div class="ctx-list">{ctx_items}</div></details>'

        cards += f"""
        <div class="card">
          <div class="card-header">
            <span class="idx">#{i}</span>
            <span class="badge" style="background:{tc}">{t}</span>
            <span class="badge" style="background:{dc}">{d}</span>
            <span class="source">{_esc(src)}</span>
          </div>
          <div class="topic">{_esc(topic)}</div>
          <div class="field"><label>Question</label><p>{_esc(entry.question)}</p></div>
          <div class="field"><label>Answer</label><p>{_esc(entry.answer)}</p></div>
          <div class="field"><label>Ground Truth</label><p>{_esc(entry.ground_truth)}</p></div>
          {ctx_html}
        </div>"""

    # Summary table rows
    summary_rows = "".join(
        f'<tr><td>{_esc(s)}</td><td>{c}</td></tr>' for s, c in sorted(by_source.items(), key=lambda x: -x[1])
    )
    type_rows = "".join(
        f'<tr><td><span class="badge-sm" style="background:{type_colors.get(t, "#6b7280")}">{t}</span></td><td>{c}</td></tr>'
        for t, c in sorted(by_type.items(), key=lambda x: -x[1])
    )
    diff_rows = "".join(
        f'<tr><td><span class="badge-sm" style="background:{diff_colors.get(d, "#6b7280")}">{d}</span></td><td>{c}</td></tr>'
        for d, c in sorted(by_diff.items(), key=lambda x: -x[1])
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QA Dataset — {len(dataset)} pairs</title>
<style>
  :root {{ --bg:#0f172a; --card:#1e293b; --border:#334155; --text:#e2e8f0; --muted:#94a3b8; --accent:#3b82f6; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:var(--bg); color:var(--text); padding:2rem; max-width:960px; margin:0 auto; }}
  h1 {{ font-size:1.5rem; margin-bottom:1rem; }}
  .stats {{ display:flex; gap:1rem; margin-bottom:2rem; flex-wrap:wrap; }}
  .stat-box {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:1rem; flex:1; min-width:180px; }}
  .stat-box h3 {{ font-size:0.75rem; text-transform:uppercase; color:var(--muted); margin-bottom:0.5rem; }}
  .stat-box table {{ width:100%; font-size:0.85rem; }}
  .stat-box td {{ padding:2px 0; }}
  .stat-box td:last-child {{ text-align:right; color:var(--muted); }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:1.25rem; margin-bottom:1rem; }}
  .card-header {{ display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem; flex-wrap:wrap; }}
  .idx {{ font-weight:700; color:var(--accent); font-size:0.9rem; }}
  .badge {{ color:#fff; font-size:0.7rem; padding:2px 8px; border-radius:99px; font-weight:600; }}
  .badge-sm {{ color:#fff; font-size:0.7rem; padding:1px 6px; border-radius:99px; font-weight:600; }}
  .source {{ font-size:0.8rem; color:var(--muted); margin-left:auto; }}
  .topic {{ font-size:0.8rem; color:var(--muted); margin-bottom:0.75rem; }}
  .field {{ margin-bottom:0.75rem; }}
  .field label {{ font-size:0.7rem; text-transform:uppercase; color:var(--accent); font-weight:600; display:block; margin-bottom:2px; }}
  .field p {{ font-size:0.9rem; line-height:1.5; }}
  details {{ margin-top:0.5rem; }}
  summary {{ cursor:pointer; font-size:0.8rem; color:var(--muted); }}
  .ctx-list {{ margin-top:0.5rem; }}
  .ctx-item {{ background:var(--bg); border-radius:6px; padding:0.75rem; margin-bottom:0.5rem; }}
  .ctx-item strong {{ font-size:0.75rem; color:var(--accent); }}
  .ctx-item p {{ font-size:0.82rem; color:var(--muted); line-height:1.5; margin-top:2px; }}
</style></head><body>
<h1>QA Dataset — {len(dataset)} pairs</h1>
<div class="stats">
  <div class="stat-box"><h3>By Document</h3><table>{summary_rows}</table></div>
  <div class="stat-box"><h3>By Type</h3><table>{type_rows}</table></div>
  <div class="stat-box"><h3>By Difficulty</h3><table>{diff_rows}</table></div>
</div>
{cards}
</body></html>"""

    html_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML dataset saved to: {html_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate RAG QA dataset from documents"
    )
    parser.add_argument(
        "documents_folder",
        type=str,
        help="Path to folder containing documents (txt, pdf, md)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap in characters (default: 100)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of QA pairs to generate (default: 100)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for QA generation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset.json",
        help="Output file path (default: dataset.json)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="QA pairs per LLM call (default: 5)",
    )
    parser.add_argument(
        "--paddleocr-lang",
        type=str,
        default="en",
        help="PaddleOCR language: en, pt, latin (default: en)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="general",
        help="Document domain for prompt framing (default: general)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="pt-BR",
        help="Output language for QA pairs (default: pt-BR)",
    )
    parser.add_argument(
        "--tree-path",
        type=str,
        default="tree/topic_tree.json",
        help="Path to save topic tree JSON (default: tree/topic_tree.json)",
    )
    parser.add_argument(
        "--no-llm-pipeline",
        action="store_true",
        default=False,
        help="Use heading-based pipeline instead of LLM-driven content slicing",
    )
    parser.add_argument(
        "--slicer-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for content slicing (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--topic-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for topic organization (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-balance-docs",
        action="store_true",
        default=False,
        help="Disable balanced per-document QA distribution (use round-robin across all leaves)",
    )
    parser.add_argument(
        "--no-balance-tf",
        action="store_true",
        default=False,
        help="Disable 50/50 true/false balance for true-false assertions",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="Clear all caches (parsed OCR, sliced segments, topic tree) before running",
    )
    parser.add_argument(
        "--qa-only",
        action="store_true",
        default=False,
        help="Skip document loading and tree building. Load cached tree and generate QA only.",
    )

    args = parser.parse_args()

    # Setup — use from_env() to respect .env.local, with CLI overrides
    config = PipelineConfig.from_env(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        num_samples=args.num_samples,
        model_name=args.model,
        temperature=args.temperature,
        batch_size=args.batch_size,
        paddleocr_lang=args.paddleocr_lang,
        document_domain=args.domain,
        language=args.language,
        output_path=args.output,
        tree_path=args.tree_path,
        use_llm_pipeline=not args.no_llm_pipeline,
        slicer_model=args.slicer_model,
        topic_model=args.topic_model,
        balance_per_document=not args.no_balance_docs,
        balance_true_false=not args.no_balance_tf,
    )

    logger = setup_logging(config.log_path)

    # Clear caches if --fresh
    if args.fresh:
        import shutil
        cache_paths = [
            Path(config.slicer_cache_dir),
            Path(config.tree_path),
            Path(config.parsed_path),
        ]
        for p in cache_paths:
            if p.is_dir():
                shutil.rmtree(p)
                logger.info(f"Cleared cache directory: {p}")
            elif p.is_file():
                p.unlink()
                logger.info(f"Cleared cache file: {p}")

    # Run pipeline
    try:
        dataset = run_pipeline(args.documents_folder, config, logger, qa_only=args.qa_only)
        save_dataset(dataset, args.output, logger)
        save_dataset_html(dataset, args.output, logger)
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
