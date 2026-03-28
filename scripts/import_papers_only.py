"""
Import paper metadata into Neo4j without running LLM extraction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from document_processor import DocumentProcessor
from neo4j_manager import Neo4jManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Import paper records as Paper nodes only.")
    parser.add_argument("file_path", help="Path to the paper CSV or JSONL file")
    parser.add_argument("--clear-database", action="store_true", help="Clear the database before import")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for imported records")
    args = parser.parse_args()

    file_path = Path(args.file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    processor = DocumentProcessor()
    payload = processor.process_document(str(file_path))
    papers = payload["papers"]
    if args.limit > 0:
        papers = papers[: args.limit]

    manager = Neo4jManager()
    if not manager.connect():
        raise ConnectionError("Failed to connect to Neo4j database")

    try:
        manager.setup_schema()
        if args.clear_database:
            manager.clear_database()

        for index, paper in enumerate(papers, start=1):
            manager.upsert_paper(paper)
            if index % 25 == 0 or index == len(papers):
                print(f"Imported {index}/{len(papers)} papers")

        stats = manager.get_graph_statistics()
        print("Paper-only import completed.")
        print(f"Imported papers: {len(papers)}")
        print(f"Total papers in database: {stats.get('total_papers', 0)}")
        print(f"Total entities in database: {stats.get('total_entities', 0)}")
        print(f"Total relations in database: {stats.get('total_relations', 0)}")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
