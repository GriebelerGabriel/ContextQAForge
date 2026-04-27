"""Main pipeline orchestration for RAG QA dataset generation.

Pipeline: Load → Build Tree → Chunk Leaves → Distribute QA → Generate → Save
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

from config import PipelineConfig
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
) -> List[DatasetEntry]:
    """
    Run the complete RAG QA generation pipeline.

    Args:
        documents_folder: Path to folder containing documents
        config: Pipeline configuration
        logger: Logger instance

    Returns:
        List of dataset entries
    """
    logger.info("=" * 60)
    logger.info("RAG QA Dataset Generation Pipeline")
    logger.info("=" * 60)

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
    tree = DocumentTopicTree(config)
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
            # Distribute within document's leaves round-robin
            per_leaf = target // len(leaves)
            leaf_remainder = target % len(leaves)
            for leaf_idx, leaf in enumerate(leaves):
                count = per_leaf + (1 if leaf_idx < leaf_remainder else 0)
                if count > 0:
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
    dataset: List[DatasetEntry] = []

    # Keep generating until we hit the target (or exhaust max_total_attempts)
    max_total_attempts = config.num_samples * 2  # safety limit: 2x target
    total_attempts = 0

    while len(dataset) < config.num_samples and total_attempts < max_total_attempts:
        for leaf, count in distribution:
            if len(dataset) >= config.num_samples:
                break

            # Use full content (entire segment text) as primary context
            if not leaf.content and not leaf.chunks:
                continue

            # Check how many this leaf has produced so far
            leaf_key = (leaf.source, leaf.name)
            produced = sum(1 for e in dataset if (e.source, e.metadata.get("topic_path", [""])[-1]) == leaf_key)
            remaining_for_leaf = count - produced
            if remaining_for_leaf <= 0:
                continue

            topic_path = tree.get_leaf_path(leaf)

            # Use the full segment content as context (not tiny 500-char chunks)
            # Also gather related content from other documents on similar topics
            related_chunks = _find_related_chunks(leaf, all_leaves, max_chunks=3)
            full_content = leaf.content or ""
            # Use full content as the primary context for the LLM
            contexts = [full_content] if full_content else leaf.chunks
            if not contexts:
                continue

            # Use batch generation if count > 1 and batch_size allows
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

            total_attempts += 1

            # Progress
            if len(dataset) % 5 == 0 or len(dataset) >= config.num_samples:
                logger.info(f"  Progress: {len(dataset)}/{config.num_samples} QA pairs (attempts: {total_attempts})")

    logger.info(f"Successfully generated {len(dataset)} QA pairs")
    return dataset


def _find_related_chunks(
    leaf: SectionNode,
    all_leaves: List[SectionNode],
    max_chunks: int = 5,
) -> List[str]:
    """Find related content from other documents on similar topics.

    Uses simple keyword overlap between leaf names to find related leaves,
    then returns chunks from those leaves as additional context.
    """
    # Extract keywords from the leaf name
    name_words = set(leaf.name.lower().split())

    related: List[str] = []
    for other in all_leaves:
        if other is leaf:
            continue
        if other.source == leaf.source:
            continue  # skip same document

        other_words = set(other.name.lower().split())
        overlap = name_words & other_words - {"and", "or", "the", "of", "in", "to", "for", "a", "de", "e", "da", "do", "em"}

        if overlap and other.content:
            related.append(other.content)

    return related[:max_chunks]


def save_dataset(dataset: List[DatasetEntry], output_path: str, logger: logging.Logger) -> None:
    """Save dataset to JSON file."""
    output = [entry.model_dump() for entry in dataset]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Dataset saved to: {output_path}")


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

    args = parser.parse_args()

    # Setup
    config = PipelineConfig(
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
        dataset = run_pipeline(args.documents_folder, config, logger)
        save_dataset(dataset, args.output, logger)
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
