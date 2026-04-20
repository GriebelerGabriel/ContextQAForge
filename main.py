"""Main pipeline orchestration for RAG QA dataset generation.

Pipeline: Load → Build Tree → Chunk Leaves → Distribute QA → Generate → Save
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from config import PipelineConfig
from generator import QAGenerator
from loader import load_documents
from models import DatasetEntry
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

    # Save tree JSON
    tree.save(config.tree_path)
    logger.info(f"Tree saved to {config.tree_path}")

    # Step 3: Chunk per leaf node
    logger.info(
        f"Chunking leaf nodes (size={config.chunk_size}, overlap={config.chunk_overlap})"
    )
    tree.chunk_leaves(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

    leaves = tree.get_all_leaf_nodes()
    total_chunks = sum(len(leaf.chunks) for leaf in leaves)
    logger.info(f"Tree has {len(leaves)} leaf nodes, {total_chunks} total chunks")

    # Step 4: Get QA distribution across leaf nodes
    distribution = tree.get_qa_distribution(config.num_samples)
    logger.info(f"Distributing {config.num_samples} QA pairs across {len(distribution)} leaf nodes")

    # Step 5: Generate QA pairs (round-robin across all leaves)
    logger.info(f"Generating QA pairs using {config.model_name}...")
    generator = QAGenerator(config)
    dataset: List[DatasetEntry] = []

    for leaf, count in distribution:
        if not leaf.chunks:
            logger.warning(f"Skipping leaf '{leaf.name}' — no chunks")
            continue

        topic_path = tree.get_leaf_path(leaf)

        # Use batch generation if count > 1 and batch_size allows
        if count > 1 and config.batch_size > 1:
            batch_count = min(count, config.batch_size)
            remaining = count
            while remaining > 0:
                this_batch = min(batch_count, remaining)
                qa_pairs = generator.generate_qa_from_section(
                    chunks=leaf.chunks,
                    topic_path=topic_path,
                    source=leaf.source,
                    num_pairs=this_batch,
                )
                for qa_pair in qa_pairs:
                    entry = generator.to_dataset_entry(qa_pair)
                    dataset.append(entry)
                remaining -= len(qa_pairs)
                if not qa_pairs:
                    break  # LLM failed, move to next leaf
        else:
            for _ in range(count):
                qa_pair = generator.generate_qa_from_section(
                    chunks=leaf.chunks,
                    topic_path=topic_path,
                    source=leaf.source,
                    num_pairs=1,
                )
                if qa_pair:
                    for pair in qa_pair:
                        entry = generator.to_dataset_entry(pair)
                        dataset.append(entry)

        # Progress
        if len(dataset) % 5 == 0 or len(dataset) == config.num_samples:
            logger.info(f"  Progress: {len(dataset)}/{config.num_samples} QA pairs")

    logger.info(f"Successfully generated {len(dataset)} QA pairs")
    return dataset


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
        default="gpt-4o",
        help="OpenAI model for QA generation (default: gpt-4o)",
    )
    parser.add_argument(
        "--topic-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for tree refinement (default: gpt-4o-mini)",
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
        default=3,
        help="QA pairs per LLM call (default: 3)",
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

    args = parser.parse_args()

    # Setup
    config = PipelineConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        num_samples=args.num_samples,
        model_name=args.model,
        topic_model=args.topic_model,
        temperature=args.temperature,
        batch_size=args.batch_size,
        paddleocr_lang=args.paddleocr_lang,
        document_domain=args.domain,
        language=args.language,
        output_path=args.output,
        tree_path=args.tree_path,
    )

    logger = setup_logging(config.log_path)

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
