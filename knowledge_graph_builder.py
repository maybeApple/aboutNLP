"""
Paper-centric knowledge graph construction pipeline.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from config import DEFAULT_LLM_MODEL
from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager, SemanticSearchEngine
from llm_extractor import LLMExtractor
from neo4j_manager import Neo4jManager, PAPER_RELATION_TYPES


class PaperKnowledgeGraphBuilder:
    """Build a normalized paper graph from LLM extraction output."""

    def __init__(self, llm_extractor: LLMExtractor):
        self.llm_extractor = llm_extractor
        self.logger = logging.getLogger(__name__)

    def build_paper_graph(self, paper_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a stable graph payload for one paper."""
        extraction = self.llm_extractor.extract_paper_knowledge(paper_record)
        derived_relations = []
        for entity in extraction["entities"]:
            relation_type = PAPER_RELATION_TYPES.get(entity["type"])
            if relation_type is None:
                continue
            derived_relations.append(
                {
                    "source": "paper",
                    "target": entity["name"],
                    "type": relation_type,
                }
            )

        deduped_relations: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for relation in derived_relations + (extraction.get("relations") or []):
            key = (
                relation["source"].lower(),
                relation["type"],
                relation["target"].lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped_relations.append(relation)

        return {
            "paper": extraction["paper"],
            "entities": extraction["entities"],
            "relations": deduped_relations,
        }


class KnowledgeGraphPipeline:
    """Main pipeline for building and querying a paper knowledge graph."""

    def __init__(self, llm_provider: str = "openai", llm_model: str = DEFAULT_LLM_MODEL):
        self.logger = logging.getLogger(__name__)
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        self.document_processor = DocumentProcessor()
        self.neo4j_manager = Neo4jManager()
        self.embedding_manager = EmbeddingManager()
        self.semantic_search_engine = SemanticSearchEngine(self.embedding_manager)

        self.llm_extractor: Optional[LLMExtractor] = None
        self.kg_builder: Optional[PaperKnowledgeGraphBuilder] = None

    def _ensure_graph_builder(self) -> None:
        if self.llm_extractor is None:
            self.llm_extractor = LLMExtractor(model_name=self.llm_model, provider=self.llm_provider)
        if self.kg_builder is None:
            self.kg_builder = PaperKnowledgeGraphBuilder(self.llm_extractor)

    def process_paper_records(
        self,
        paper_records: List[Dict[str, Any]],
        clear_database: bool = False,
        source_name: str = "paper_records",
    ) -> Dict[str, Any]:
        """Process in-memory paper records into Neo4j."""
        if not self.neo4j_manager.connect():
            raise ConnectionError("Failed to connect to Neo4j database")

        try:
            self.neo4j_manager.setup_schema()
            if clear_database:
                self.neo4j_manager.clear_database()

            total_entities = 0
            total_relations = 0
            if paper_records:
                self._ensure_graph_builder()
                for index, paper_record in enumerate(paper_records, start=1):
                    self.logger.info("Processing paper %s/%s: %s", index, len(paper_records), paper_record["title"])
                    paper_graph = self.kg_builder.build_paper_graph(paper_record)
                    stored_counts = self.neo4j_manager.upsert_paper_graph(paper_graph)
                    total_entities += stored_counts["entities"]
                    total_relations += stored_counts["relations"]

            stats = self.neo4j_manager.get_graph_statistics()
            return {
                "source_name": source_name,
                "total_papers": len(paper_records),
                "processed_papers": len(paper_records),
                "total_entities_written": total_entities,
                "total_relations_written": total_relations,
                "database_stats": stats,
                "processing_successful": True,
            }
        finally:
            self.neo4j_manager.close()

    def process_document(
        self,
        file_path: str,
        clear_database: bool = False,
        start: int = 0,
        count: int | None = None,
    ) -> Dict[str, Any]:
        """Process a CSV/JSONL paper source or a single legacy document."""
        payload = self.document_processor.process_document(file_path)
        paper_records = payload["papers"]
        source_total_papers = len(paper_records)

        if start > source_total_papers:
            raise ValueError(
                f"Start index {start} is out of range for {source_total_papers} paper(s)."
            )

        batch_start = start
        batch_end = source_total_papers if count is None else min(source_total_papers, start + count)
        selected_records = paper_records[batch_start:batch_end]

        if source_total_papers and not selected_records:
            raise ValueError(
                f"No papers selected from index {batch_start} with count={count}."
            )

        selected_tokens = sum(
            chunk["tokens"]
            for paper in selected_records
            for chunk in paper["chunks"]
        )
        selected_chunks = sum(len(paper["chunks"]) for paper in selected_records)
        result = self.process_paper_records(
            paper_records=selected_records,
            clear_database=clear_database,
            source_name=payload["file_name"],
        )
        result.update(
            {
                "file_name": payload["file_name"],
                "total_chunks": selected_chunks,
                "total_tokens": selected_tokens,
                "processed_papers": len(selected_records),
                "skipped_papers": payload["skipped_papers"],
                "total_input_records": payload["total_input_records"],
                "ingestion_warnings": payload["warnings"],
                "source_total_papers": source_total_papers,
                "batch_start": batch_start,
                "batch_end": batch_end,
                "batch_selected_papers": len(selected_records),
            }
        )
        return result

    def search_entities(
        self,
        query: str,
        limit: int = 10,
        entity_type: str | None = None,
    ) -> List[Dict[str, Any]]:
        if not self.neo4j_manager.connect():
            raise ConnectionError("Failed to connect to Neo4j database")
        try:
            return self.neo4j_manager.search_entities(query, limit=limit, entity_type=entity_type)
        finally:
            self.neo4j_manager.close()

    def get_entity_context(
        self,
        entity_name: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        if not self.neo4j_manager.connect():
            raise ConnectionError("Failed to connect to Neo4j database")
        try:
            return self.neo4j_manager.get_entity_context(entity_name, entity_type=entity_type, limit=limit)
        finally:
            self.neo4j_manager.close()

    def get_relation_triples(self, relation_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.neo4j_manager.connect():
            raise ConnectionError("Failed to connect to Neo4j database")
        try:
            return self.neo4j_manager.get_relation_triples(relation_type, limit=limit)
        finally:
            self.neo4j_manager.close()

    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.neo4j_manager.connect():
            raise ConnectionError("Failed to connect to Neo4j database")
        try:
            return self.neo4j_manager.semantic_search(query, limit=limit)
        finally:
            self.neo4j_manager.close()

    def get_graph_statistics(self) -> Dict[str, Any]:
        if not self.neo4j_manager.connect():
            raise ConnectionError("Failed to connect to Neo4j database")
        try:
            return self.neo4j_manager.get_graph_statistics()
        finally:
            self.neo4j_manager.close()

    def get_available_relation_types(self) -> List[str]:
        return self.neo4j_manager.get_available_relation_types()

    def find_entity_paths(
        self,
        entity1: str,
        entity2: str,
        max_depth: int = 3,
        entity1_type: str | None = None,
        entity2_type: str | None = None,
    ) -> List[Dict[str, Any]]:
        if not self.neo4j_manager.connect():
            raise ConnectionError("Failed to connect to Neo4j database")
        try:
            return self.neo4j_manager.find_entity_paths(
                entity1,
                entity2,
                max_depth=max_depth,
                entity1_type=entity1_type,
                entity2_type=entity2_type,
            )
        finally:
            self.neo4j_manager.close()


class GraphRAGQueryEngine:
    """Lightweight paper-context query helper."""

    def __init__(self, pipeline: KnowledgeGraphPipeline):
        self.pipeline = pipeline
        self.logger = logging.getLogger(__name__)

    def _candidate_queries(self, question: str) -> List[str]:
        cleaned_question = question.strip()
        if not cleaned_question:
            return []

        candidates = [cleaned_question]
        quoted_phrases = re.findall(r'"([^"]+)"|\'([^\']+)\'', cleaned_question)
        for left, right in quoted_phrases:
            phrase = left or right
            if phrase:
                candidates.append(phrase)

        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9/+\-_.]*", cleaned_question)
        stopwords = {
            "what",
            "which",
            "papers",
            "paper",
            "methods",
            "method",
            "used",
            "use",
            "for",
            "with",
            "the",
            "are",
            "is",
            "on",
            "in",
            "of",
            "by",
        }
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
        max_ngram = min(4, len(filtered_tokens))
        for size in range(max_ngram, 0, -1):
            for index in range(len(filtered_tokens) - size + 1):
                candidates.append(" ".join(filtered_tokens[index : index + size]))

        deduped: List[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = candidate.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(candidate.strip())
        return deduped

    def query_with_context(self, question: str, context_limit: int = 5) -> Dict[str, Any]:
        candidate_queries = self._candidate_queries(question)

        relevant_papers: List[Dict[str, Any]] = []
        seen_papers: set[str] = set()
        for candidate in candidate_queries:
            for paper in self.pipeline.semantic_search(candidate, limit=context_limit):
                title = paper.get("title", "")
                if title in seen_papers:
                    continue
                seen_papers.add(title)
                relevant_papers.append(paper)
                if len(relevant_papers) >= context_limit:
                    break
            if len(relevant_papers) >= context_limit:
                break

        related_entities: List[Dict[str, Any]] = []
        seen_entities: set[tuple[str, str]] = set()
        for candidate in candidate_queries:
            for entity in self.pipeline.search_entities(candidate, limit=5):
                key = (entity["type"], entity["name"].lower())
                if key in seen_entities:
                    continue
                seen_entities.add(key)
                related_entities.append(entity)
                if len(related_entities) >= 8:
                    break
            if len(related_entities) >= 8:
                break

        supporting_triples: List[Dict[str, Any]] = []
        seen_triples: set[tuple[str, str, str]] = set()
        for entity in related_entities[:3]:
            context = self.pipeline.get_entity_context(entity["name"], entity_type=entity["type"], limit=5)
            for paper in context["papers"]:
                key = (paper["title"], paper["relation_type"], entity["name"])
                if key in seen_triples:
                    continue
                seen_triples.add(key)
                supporting_triples.append(
                    {
                        "source": paper["title"],
                        "source_type": "PAPER",
                        "relation_type": paper["relation_type"],
                        "target": entity["name"],
                        "target_type": entity["type"],
                    }
                )

        context_text = "\n\n".join(
            f"{paper['title']} ({paper.get('year') or 'Unknown'}): {paper.get('abstract', '')}"
            for paper in relevant_papers
        )
        return {
            "question": question,
            "relevant_papers": relevant_papers,
            "related_entities": related_entities,
            "supporting_triples": supporting_triples,
            "context_text": context_text,
        }


if __name__ == "__main__":
    pipeline = KnowledgeGraphPipeline()
    print("KnowledgeGraphPipeline ready for paper-centric ingestion.")
