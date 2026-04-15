"""Main pipeline orchestration for RAG QA dataset generation."""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List

import faiss

from config import PipelineConfig
from chunker import chunk_documents
from embedder import Embedder, EmbeddingCache
from generator import QAGenerator
from loader import load_documents
from models import DatasetEntry
from topic_tree import TopicTree
from vector_store import VectorStore


def _deduplicate_chunks(chunks: list, overlap_threshold: float = 0.6) -> list:
    """Remove chunks with high word overlap with already-seen chunks."""
    unique = []
    for chunk in chunks:
        words = set(chunk.content.lower().split())
        if not words:
            continue
        is_dup = False
        for existing in unique:
            existing_words = set(existing.content.lower().split())
            overlap = len(words & existing_words) / max(len(words), 1)
            if overlap > overlap_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(chunk)
    return unique


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

    # Step 1: Load documents (with optional LLM PDF extraction)
    logger.info(f"Loading documents from: {documents_folder}")
    if config.use_llm_for_pdf:
        logger.info("LLM PDF extraction enabled (slower, more accurate)")
    documents = load_documents(
        documents_folder,
        use_llm_for_pdf=config.use_llm_for_pdf,
        pdf_extraction_model=config.pdf_extraction_model,
        api_key=config.openai_api_key,
        pdf_remove_patterns=config.pdf_remove_patterns,
    )
    logger.info(f"Loaded {len(documents)} documents")

    if not documents:
        raise ValueError("No documents found in the specified folder")

    # Step 2: Chunk documents
    logger.info(
        f"Chunking with size={config.chunk_size}, overlap={config.chunk_overlap}"
    )
    chunks = chunk_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        quality_patterns=config.chunk_quality_patterns,
    )
    logger.info(f"Created {len(chunks)} chunks")

    # Step 2b: Deduplicate chunks with high text overlap
    # Skip dedup for small chunk sets to avoid over-pruning
    if len(chunks) > 50:
        chunks = _deduplicate_chunks(chunks, overlap_threshold=config.dedup_overlap_threshold)
        logger.info(f"After deduplication: {len(chunks)} unique chunks")
    else:
        logger.info(f"Skipping deduplication: {len(chunks)} chunks (too few to deduplicate)")

    # Step 3: Create embeddings
    logger.info(f"Generating embeddings using {config.embedding_model}")
    embedder = Embedder(config)
    chunk_texts = [chunk.content for chunk in chunks]
    embeddings = embedder.embed_chunks(chunk_texts)
    logger.info(f"Generated embeddings: {embeddings.shape}")

    # Step 4: Store in vector database
    logger.info("Building FAISS vector store")
    vector_store = VectorStore(dimension=embedder.dimension, metric=faiss.METRIC_INNER_PRODUCT)
    vector_store.add_documents(chunks, embeddings)
    logger.info(f"Vector store contains {len(vector_store)} chunks")

    # Step 5: Build topic tree for progressive specificity
    logger.info("Building topic tree for progressive question generation")
    topic_tree = TopicTree(
        config=config,
        root_topic="Document Topics",
        tree_degree=config.tree_degree,
        tree_depth=config.tree_depth,
    )
    topic_tree.build_from_chunks(chunk_texts)
    logger.info(f"Topic tree built with {len(topic_tree.leaf_paths)} leaf paths")

    # Display tree structure
    topic_tree.visualize(logger)

    # Generate query seeds and pre-compute their embeddings
    topic_tree.generate_query_seeds()
    logger.info(f"Generated query seeds for {len(topic_tree.query_seeds)} topic paths")

    embedding_cache = EmbeddingCache(embedder)
    embedding_cache.precompute(list(topic_tree.query_seeds.values()))
    logger.info(f"Pre-computed {len(topic_tree.query_seeds)} query seed embeddings")

    # Step 6: Generate QA pairs with topic tree guidance
    logger.info(f"Generating {config.num_samples} QA pairs...")
    generator = QAGenerator(config, embedder=embedder)
    generator.set_topic_tree(topic_tree)
    dataset: List[DatasetEntry] = []
    used_chunk_indices: set = set()
    total_chunks = len(vector_store)

    max_attempts = config.num_samples * 10  # Hard ceiling to avoid infinite loops
    attempt = 0
    while len(dataset) < config.num_samples and attempt < max_attempts:
        attempt += 1

        # Reset used chunks when most have been used (allows new combinations)
        if len(used_chunk_indices) > total_chunks * 0.7:
            logger.info("Resetting chunk pool — most chunks have been used")
            used_chunk_indices.clear()

        # Get topic path FIRST to guide retrieval
        topic_path = topic_tree.get_next_topic_path()

        # Retrieval: use query seed (rich search terms) for vector search
        if topic_path:
            query_seed = topic_tree.get_query_seed(topic_path)
            seed_embedding = embedding_cache.embed_query(query_seed)
            retrieved_chunks, scores, chunk_indices = vector_store.search(
                seed_embedding, top_k=config.top_k, exclude_indices=used_chunk_indices
            )
            used_chunk_indices.update(chunk_indices)
        else:
            # No topic path: use random retrieval
            retrieved_chunks = vector_store.get_random_chunks(n=config.top_k)
            scores = [0.0] * len(retrieved_chunks)

        if not retrieved_chunks:
            continue

        contexts = [chunk.content for chunk in retrieved_chunks]

        # Generate QA pairs — use batch if config.batch_size > 1
        if config.batch_size > 1:
            remaining = config.num_samples - len(dataset)
            batch_count = min(config.batch_size, remaining)
            qa_pairs = generator.generate_qa_batch(
                contexts, num_pairs=batch_count, iteration=attempt, topic_path=topic_path
            )
            for qa_pair in qa_pairs:
                entry = generator.to_dataset_entry(qa_pair)
                dataset.append(entry)
        else:
            # Single-pair generation (backward compatibility)
            qa_pair = generator.generate_qa(contexts, iteration=attempt, topic_path=topic_path)
            if qa_pair is None:
                continue
            entry = generator.to_dataset_entry(qa_pair)
            dataset.append(entry)

        # Simple progress counter (every 10 samples)
        if len(dataset) % 10 == 0 or len(dataset) == config.num_samples:
            logger.info(f"  Progress: {len(dataset)}/{config.num_samples} QA pairs generated (attempt {attempt})")

    logger.info(f"Successfully generated {len(dataset)} QA pairs after {attempt} attempts")
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
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
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
        help="OpenAI model name (default: gpt-4o)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model (default: text-embedding-3-small)",
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
        "--tree-degree",
        type=int,
        default=3,
        help="Topic tree branching factor (default: 3)",
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=3,
        help="Topic tree depth (default: 3)",
    )

    args = parser.parse_args()

    # Setup
    config = PipelineConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        num_samples=args.num_samples,
        model_name=args.model,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        batch_size=args.batch_size,
        tree_degree=args.tree_degree,
        tree_depth=args.tree_depth,
        output_path=args.output,
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
