"""
Main execution script for the paper knowledge graph system.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from config import DEFAULT_LLM_MODEL
from knowledge_graph_builder import KnowledgeGraphPipeline


def configure_console_encoding() -> None:
    """Avoid Windows console crashes when the script prints Unicode symbols."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def setup_logging(log_level: str = "INFO") -> None:
    """Set up console and file logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "knowledge_graph.log", encoding="utf-8"),
        ],
    )


def print_ingestion_warnings(warnings: list[str]) -> None:
    """Print skipped-record warnings without flooding the console."""
    if not warnings:
        return

    print("\nWarnings:")
    for warning in warnings[:10]:
        print(f"   - {warning}")
    if len(warnings) > 10:
        print(f"   - ... {len(warnings) - 10} more warning(s)")


def non_negative_int(value: str) -> int:
    """Argparse helper for non-negative integer arguments."""
    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("Value must be greater than or equal to 0.")
    return parsed_value


def process_document(
    file_path: str,
    llm_provider: str = "openai",
    llm_model: str = DEFAULT_LLM_MODEL,
    clear_database: bool = False,
    start: int = 0,
    count: int | None = None,
):
    """Process a paper source and build a knowledge graph."""
    print("Starting paper knowledge graph ingestion")
    print(f"Source file: {file_path}")
    print(f"LLM: {llm_provider}/{llm_model}")
    if start or count is not None:
        requested_count = "all remaining papers" if count is None else str(count)
        print(f"Batch selection: start={start}, count={requested_count}")

    pipeline = KnowledgeGraphPipeline(llm_provider, llm_model)

    try:
        result = pipeline.process_document(
            file_path,
            clear_database=clear_database,
            start=start,
            count=count,
        )

        print("\nKnowledge graph construction completed.")
        print("Run summary:")
        print(f"   - File: {result['file_name']}")
        print(f"   - Input records: {result.get('total_input_records', result.get('processed_papers', 0))}")
        if result.get("source_total_papers", 0) != result.get("batch_selected_papers", result.get("processed_papers", 0)):
            print(
                "   - Selected papers: "
                f"{result.get('batch_start', 0) + 1}-{result.get('batch_end', 0)} "
                f"of {result.get('source_total_papers', 0)}"
            )
        print(f"   - Papers processed: {result.get('processed_papers', result.get('total_papers', 0))}")
        print(f"   - Papers skipped: {result.get('skipped_papers', 0)}")
        print(f"   - Chunks: {result['total_chunks']}")
        print(f"   - Tokens: {result['total_tokens']}")
        print(f"   - Entities written: {result['total_entities_written']}")
        print(f"   - Relations written: {result['total_relations_written']}")

        stats = result["database_stats"]
        print("\nDatabase statistics:")
        print(f"   - Total Papers: {stats.get('total_papers', 0)}")
        print(f"   - Total Entities: {stats.get('total_entities', 0)}")
        print(f"   - Total Relations: {stats.get('total_relations', 0)}")

        print_ingestion_warnings(result.get("ingestion_warnings") or [])
        return result

    except Exception as exc:
        print(f"Error processing document: {exc}")
        logging.error("Document processing failed: %s", exc)
        return None


def interactive_mode() -> None:
    """Run in interactive mode for querying the knowledge graph."""
    print("Starting interactive paper knowledge base explorer")
    print("Launching the Streamlit interface.")

    try:
        query_script = Path(__file__).with_name("query_interface.py")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(query_script)],
            check=True,
            cwd=str(query_script.parent),
        )
    except Exception as exc:
        print(f"Error starting web interface: {exc}")
        logging.error("Web interface failed: %s", exc)


def test_system_components() -> None:
    """Test core system components."""
    print("Testing document processor...")
    try:
        from document_processor import DocumentProcessor

        DocumentProcessor()
        print("Document processor: OK")
    except Exception as exc:
        print(f"Document processor: {exc}")

    print("Testing Neo4j connection...")
    try:
        from neo4j_manager import Neo4jManager

        manager = Neo4jManager()
        if manager.connect():
            print("Neo4j connection: OK")
            manager.close()
        else:
            print("Neo4j connection: Failed")
    except Exception as exc:
        print(f"Neo4j connection: {exc}")

    print("Testing embedding system...")
    try:
        from embedding_manager import EmbeddingManager

        embedding_manager = EmbeddingManager()
        test_embedding = embedding_manager.generate_single_embedding("test")
        print(f"Embedding system: OK (dimension: {len(test_embedding)})")
    except Exception as exc:
        print(f"Embedding system: {exc}")

    print("Testing LLM extractor...")
    try:
        from llm_extractor import LLMExtractor

        LLMExtractor()
        print("LLM extractor: OK")
    except Exception as exc:
        print(f"LLM extractor: {exc}")

    print("Testing knowledge graph pipeline...")
    try:
        KnowledgeGraphPipeline()
        print("Knowledge graph pipeline: OK")
    except Exception as exc:
        print(f"Knowledge graph pipeline: {exc}")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Paper knowledge graph system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a paper CSV file
  python main.py process sample_papers.csv --clear-database

  # Process a larger file in batches through the existing CLI
  python main.py process data/papers_bootstrap_240.csv --provider dashscope --model qwen3-max --start 0 --count 40

  # Process JSONL with a specific LLM
  python main.py process papers.jsonl --provider anthropic --model claude-3-sonnet-20240229

  # Process with Qwen through DashScope's OpenAI-compatible API
  python main.py process sample_papers.csv --provider dashscope --model qwen3-max --clear-database

  # Process a legacy single-paper .docx file
  python main.py process single_paper.docx

  # Launch the web interface
  python main.py web

  # Process and then launch the web interface
  python main.py process sample_papers.csv --web
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    process_parser = subparsers.add_parser("process", help="Process a paper source and build the graph")
    process_parser.add_argument("file_path", help="Path to a .csv, .jsonl, .docx, or .txt source file")
    process_parser.add_argument(
        "--provider",
        choices=["openai", "dashscope", "anthropic"],
        default="openai",
        help="LLM provider to use",
    )
    process_parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="LLM model to use")
    process_parser.add_argument(
        "--clear-database",
        action="store_true",
        help="Clear the database before processing",
    )
    process_parser.add_argument(
        "--start",
        type=non_negative_int,
        default=0,
        help="Start processing from this zero-based paper index within the source file",
    )
    process_parser.add_argument(
        "--count",
        type=non_negative_int,
        default=None,
        help="Process at most this many papers from the starting index",
    )
    process_parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the web interface after processing",
    )
    process_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    web_parser = subparsers.add_parser("web", help="Launch the web interface")
    web_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    test_parser = subparsers.add_parser("test", help="Test system components")
    test_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser


def main() -> None:
    """Main entry point for the application."""
    configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup_logging(args.log_level)

    if args.command == "process":
        if not Path(args.file_path).exists():
            print(f"Error: file '{args.file_path}' not found")
            return

        result = process_document(
            args.file_path,
            args.provider,
            args.model,
            args.clear_database,
            args.start,
            args.count,
        )

        if result and args.web:
            print("\nLaunching web interface...")
            interactive_mode()

    elif args.command == "web":
        interactive_mode()

    elif args.command == "test":
        print("Testing system components...")
        test_system_components()


if __name__ == "__main__":
    main()
