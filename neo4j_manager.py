"""
Neo4j manager for a paper-centric academic knowledge graph.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME


ENTITY_LABELS = {
    "TASK": "Task",
    "METHOD": "Method",
    "DATASET": "Dataset",
    "METRIC": "Metric",
    "RESULT": "Result",
}
LABEL_TO_ENTITY_TYPE = {label: entity_type for entity_type, label in ENTITY_LABELS.items()}
PAPER_RELATION_TYPES = {
    "TASK": "STUDIES",
    "METHOD": "PROPOSES",
    "DATASET": "USES_DATASET",
    "METRIC": "EVALUATED_BY",
}
ENTITY_RELATION_TYPES = {"APPLIED_TO", "TESTED_ON", "ACHIEVES"}
ALL_RELATION_TYPES = set(PAPER_RELATION_TYPES.values()) | ENTITY_RELATION_TYPES
ENTITY_ALIASES = {
    "DATASET": {
        "cnn / dailymail": "CNN/DailyMail",
        "cnn/dailymail": "CNN/DailyMail",
        "cnn dailymail": "CNN/DailyMail",
        "conll 2003": "CoNLL-2003",
        "conll2003": "CoNLL-2003",
        "ms marco": "MS MARCO",
        "msmarco": "MS MARCO",
    },
    "METRIC": {
        "f 1": "F1",
        "f-1": "F1",
        "f1": "F1",
        "rouge l": "ROUGE-L",
        "rouge-l": "ROUGE-L",
        "em": "EM",
    },
}


class Neo4jManager:
    """Manages a paper-focused Neo4j graph with stable labels and relations."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        username: str = NEO4J_USERNAME,
        password: str = NEO4J_PASSWORD,
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        """Establish a Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.logger.info("Successfully connected to Neo4j database")
            return True
        except Exception as exc:
            self.logger.error("Failed to connect to Neo4j: %s", exc)
            return False

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver is not None:
            self.driver.close()
            self.driver = None
            self.logger.info("Neo4j connection closed")

    def _normalize_lookup_key(self, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value).strip())
        text = text.replace("–", "-").replace("—", "-")
        text = re.sub(r"\s*/\s*", "/", text)
        text = re.sub(r"\s*-\s*", "-", text)
        return text.lower()

    def normalize_entity(self, name: str, entity_type: str) -> Dict[str, str]:
        """Normalize entity text for MERGE and query stability."""
        normalized_type = entity_type.upper().strip()
        display_name = re.sub(r"\s+", " ", str(name).strip())
        alias_map = ENTITY_ALIASES.get(normalized_type, {})
        alias_key = self._normalize_lookup_key(display_name)
        canonical_name = alias_map.get(alias_key, display_name)
        return {
            "name": canonical_name,
            "name_normalized": self._normalize_lookup_key(canonical_name),
        }

    def _paper_title_normalized(self, title: str) -> str:
        return self._normalize_lookup_key(title)

    def _entity_identity(
        self,
        entity_type: str,
        entity_name: str,
        paper_title_normalized: str | None = None,
    ) -> Dict[str, str]:
        normalized = self.normalize_entity(entity_name, entity_type)
        if entity_type == "RESULT":
            if not paper_title_normalized:
                raise ValueError("Result nodes require paper_title_normalized for unique identity.")
            return {
                "key_name": "result_key",
                "key_value": f"{paper_title_normalized}::{normalized['name_normalized']}",
                "display_name": normalized["name"],
                "display_key": normalized["name_normalized"],
            }
        return {
            "key_name": "name_normalized",
            "key_value": normalized["name_normalized"],
            "display_name": normalized["name"],
            "display_key": normalized["name_normalized"],
        }

    def setup_schema(self) -> None:
        """Create paper-graph constraints and indexes."""
        schema_queries = [
            "CREATE CONSTRAINT paper_title_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.title_normalized IS UNIQUE",
            "CREATE CONSTRAINT task_name_unique IF NOT EXISTS FOR (n:Task) REQUIRE n.name_normalized IS UNIQUE",
            "CREATE CONSTRAINT method_name_unique IF NOT EXISTS FOR (n:Method) REQUIRE n.name_normalized IS UNIQUE",
            "CREATE CONSTRAINT dataset_name_unique IF NOT EXISTS FOR (n:Dataset) REQUIRE n.name_normalized IS UNIQUE",
            "CREATE CONSTRAINT metric_name_unique IF NOT EXISTS FOR (n:Metric) REQUIRE n.name_normalized IS UNIQUE",
            "CREATE CONSTRAINT result_key_unique IF NOT EXISTS FOR (n:Result) REQUIRE n.result_key IS UNIQUE",
            "CREATE INDEX paper_year_index IF NOT EXISTS FOR (p:Paper) ON (p.year)",
            "CREATE INDEX task_name_index IF NOT EXISTS FOR (n:Task) ON (n.name_normalized)",
            "CREATE INDEX method_name_index IF NOT EXISTS FOR (n:Method) ON (n.name_normalized)",
            "CREATE INDEX dataset_name_index IF NOT EXISTS FOR (n:Dataset) ON (n.name_normalized)",
            "CREATE INDEX metric_name_index IF NOT EXISTS FOR (n:Metric) ON (n.name_normalized)",
            "CREATE INDEX result_name_index IF NOT EXISTS FOR (n:Result) ON (n.name_normalized)",
            "CREATE FULLTEXT INDEX paper_text_search IF NOT EXISTS FOR (p:Paper) ON EACH [p.title, p.abstract, p.text]",
        ]

        with self.driver.session() as session:
            for query in schema_queries:
                try:
                    session.run(query)
                except Exception as exc:
                    self.logger.warning("Schema query failed: %s", exc)

    def clear_database(self) -> None:
        """Clear all nodes and relationships."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            self.logger.info("Database cleared")

    def upsert_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update a Paper node."""
        title = str(paper.get("title", "")).strip()
        if not title:
            raise ValueError("Paper title is required.")

        params = {
            "paper_id": paper.get("paper_id"),
            "title": title,
            "title_normalized": self._paper_title_normalized(title),
            "year": paper.get("year"),
            "authors": paper.get("authors") or [],
            "abstract": paper.get("abstract", ""),
            "text": paper.get("text", ""),
        }
        query = """
        MERGE (p:Paper {title_normalized: $title_normalized})
        ON CREATE SET p.created_at = datetime()
        SET p.paper_id = $paper_id,
            p.title = $title,
            p.year = $year,
            p.authors = $authors,
            p.abstract = $abstract,
            p.text = $text,
            p.updated_at = datetime()
        RETURN p.title as title, p.title_normalized as title_normalized
        """
        with self.driver.session() as session:
            record = session.run(query, **params).single()
        return dict(record)

    def upsert_entity(self, entity: Dict[str, Any], paper_title_normalized: str) -> Dict[str, Any]:
        """Create or update a typed entity node."""
        entity_type = str(entity.get("type", "")).upper().strip()
        label = ENTITY_LABELS.get(entity_type)
        if label is None:
            raise ValueError(f"Unsupported entity type: {entity_type}")

        identity = self._entity_identity(entity_type, str(entity.get("name", "")), paper_title_normalized)
        params = {
            "name": identity["display_name"],
            "name_normalized": identity["display_key"],
            "description": str(entity.get("description", "")).strip(),
            "paper_title_normalized": paper_title_normalized,
            "merge_key": identity["key_value"],
        }
        if entity_type == "RESULT":
            query = f"""
            MERGE (n:{label} {{result_key: $merge_key}})
            ON CREATE SET n.created_at = datetime()
            SET n.name = $name,
                n.name_normalized = $name_normalized,
                n.paper_title_normalized = $paper_title_normalized,
                n.description = $description,
                n.updated_at = datetime()
            RETURN n.name as name, '{entity_type}' as type
            """
        else:
            query = f"""
            MERGE (n:{label} {{name_normalized: $merge_key}})
            ON CREATE SET n.created_at = datetime()
            SET n.name = $name,
                n.name_normalized = $name_normalized,
                n.description = $description,
                n.updated_at = datetime()
            RETURN n.name as name, '{entity_type}' as type
            """

        with self.driver.session() as session:
            record = session.run(query, **params).single()
        return dict(record)

    def merge_paper_relation(self, paper_title: str,
                             relation_type: str,
                             target_type: str,
                             target_name: str) -> None:
        """Create a Paper -> Entity relation."""
        if relation_type not in ALL_RELATION_TYPES:
            raise ValueError(f"Unsupported relation type: {relation_type}")
        label = ENTITY_LABELS.get(target_type)
        if label is None:
            raise ValueError(f"Unsupported target type: {target_type}")

        paper_title_normalized = self._paper_title_normalized(paper_title)
        identity = self._entity_identity(target_type, target_name, paper_title_normalized)
        query = f"""
        MATCH (p:Paper {{title_normalized: $paper_title_normalized}})
        MATCH (target:{label} {{{identity['key_name']}: $target_key}})
        MERGE (p)-[r:{relation_type}]->(target)
        ON CREATE SET r.created_at = datetime()
        SET r.updated_at = datetime()
        """
        with self.driver.session() as session:
            session.run(
                query,
                paper_title_normalized=paper_title_normalized,
                target_key=identity["key_value"],
            )

    def merge_entity_relation(
        self,
        source_type: str,
        source_name: str,
        relation_type: str,
        target_type: str,
        target_name: str,
        paper_title: str,
    ) -> None:
        """Create a typed entity -> typed entity relation."""
        if relation_type not in ALL_RELATION_TYPES:
            raise ValueError(f"Unsupported relation type: {relation_type}")

        source_label = ENTITY_LABELS.get(source_type)
        target_label = ENTITY_LABELS.get(target_type)
        if source_label is None or target_label is None:
            raise ValueError("Unsupported entity label in relation.")

        paper_title_normalized = self._paper_title_normalized(paper_title)
        source_identity = self._entity_identity(source_type, source_name, paper_title_normalized)
        target_identity = self._entity_identity(target_type, target_name, paper_title_normalized)
        query = f"""
        MATCH (source:{source_label} {{{source_identity['key_name']}: $source_key}})
        MATCH (target:{target_label} {{{target_identity['key_name']}: $target_key}})
        MERGE (source)-[r:{relation_type}]->(target)
        ON CREATE SET r.created_at = datetime()
        SET r.updated_at = datetime()
        """
        with self.driver.session() as session:
            session.run(query, source_key=source_identity["key_value"], target_key=target_identity["key_value"])

    def upsert_paper_graph(self, paper_graph: Dict[str, Any]) -> Dict[str, int]:
        """Persist one extracted paper graph using the fixed academic schema."""
        paper = paper_graph.get("paper") or {}
        stored_paper = self.upsert_paper(paper)
        paper_title = stored_paper["title"]
        paper_title_normalized = stored_paper["title_normalized"]

        stored_entities: List[Dict[str, Any]] = []
        entity_type_by_name: Dict[str, str] = {}
        for entity in paper_graph.get("entities") or []:
            stored = self.upsert_entity(entity, paper_title_normalized)
            stored_entities.append(stored)
            entity_type_by_name[self._normalize_lookup_key(stored["name"])] = stored["type"]

        relation_keys: set[tuple[str, str, str]] = set()
        relation_count = 0

        for entity in stored_entities:
            relation_type = PAPER_RELATION_TYPES.get(entity["type"])
            if relation_type is None:
                continue
            key = ("paper", relation_type, self._normalize_lookup_key(entity["name"]))
            if key in relation_keys:
                continue
            self.merge_paper_relation(paper_title, relation_type, entity["type"], entity["name"])
            relation_keys.add(key)
            relation_count += 1

        for relation in paper_graph.get("relations") or []:
            relation_type = relation["type"]
            target_name = relation["target"]
            target_type = entity_type_by_name.get(self._normalize_lookup_key(target_name))
            if target_type is None:
                continue

            source_name = relation["source"]
            key = (
                self._normalize_lookup_key(source_name),
                relation_type,
                self._normalize_lookup_key(target_name),
            )
            if key in relation_keys:
                continue

            if source_name.lower() == "paper":
                self.merge_paper_relation(paper_title, relation_type, target_type, target_name)
            else:
                source_type = entity_type_by_name.get(self._normalize_lookup_key(source_name))
                if source_type is None:
                    continue
                self.merge_entity_relation(
                    source_type=source_type,
                    source_name=source_name,
                    relation_type=relation_type,
                    target_type=target_type,
                    target_name=target_name,
                    paper_title=paper_title,
                )
            relation_keys.add(key)
            relation_count += 1

        return {
            "papers": 1,
            "entities": len(stored_entities),
            "relations": relation_count,
        }

    def _fetch_entity_node(self, entity_name: str, entity_type: str | None = None) -> Optional[Dict[str, Any]]:
        normalized_query = self._normalize_lookup_key(entity_name)
        search_types = [entity_type.upper()] if entity_type else list(ENTITY_LABELS.keys())

        with self.driver.session() as session:
            for current_type in search_types:
                label = ENTITY_LABELS[current_type]
                query = f"""
                MATCH (n:{label})
                WHERE n.name_normalized = $search_value
                RETURN elementId(n) as element_id, n.name as name, '{current_type}' as type,
                       coalesce(n.description, '') as description
                LIMIT 1
                """
                record = session.run(query, search_value=normalized_query).single()
                if record:
                    return dict(record)
        return None

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Return paper-centric graph statistics."""
        counts: Dict[str, int] = {}
        with self.driver.session() as session:
            counts["total_papers"] = session.run("MATCH (p:Paper) RETURN count(p) as count").single()["count"]
            counts["total_relations"] = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            for entity_type, label in ENTITY_LABELS.items():
                counts[entity_type.lower()] = session.run(
                    f"MATCH (n:{label}) RETURN count(n) as count"
                ).single()["count"]
            papers_by_year = [
                dict(record)
                for record in session.run(
                    """
                    MATCH (p:Paper)
                    RETURN coalesce(toString(p.year), 'Unknown') as year, count(*) as count
                    ORDER BY year DESC
                    """
                )
            ]

        total_entities = sum(counts[key] for key in ("task", "method", "dataset", "metric", "result"))
        entity_breakdown = [
            {"type": "TASK", "count": counts["task"]},
            {"type": "METHOD", "count": counts["method"]},
            {"type": "DATASET", "count": counts["dataset"]},
            {"type": "METRIC", "count": counts["metric"]},
            {"type": "RESULT", "count": counts["result"]},
        ]
        return {
            "total_papers": counts["total_papers"],
            "total_entities": total_entities,
            "total_relations": counts["total_relations"],
            "entity_breakdown": entity_breakdown,
            "papers_by_year": papers_by_year,
        }

    def search_entities(
        self,
        search_term: str,
        limit: int = 10,
        entity_type: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Search entity nodes by normalized name."""
        normalized_query = self._normalize_lookup_key(search_term)
        if not normalized_query:
            return []

        search_types = [entity_type.upper()] if entity_type else list(ENTITY_LABELS.keys())
        results: List[Dict[str, Any]] = []

        with self.driver.session() as session:
            per_label_limit = max(limit, 10)
            for current_type in search_types:
                label = ENTITY_LABELS.get(current_type)
                if label is None:
                    continue
                query = f"""
                MATCH (n:{label})
                WHERE n.name_normalized CONTAINS $search_value
                OPTIONAL MATCH (p:Paper)-[]->(n)
                RETURN elementId(n) as element_id, n.name as name, '{current_type}' as type,
                       coalesce(n.description, '') as description, count(DISTINCT p) as paper_count
                ORDER BY paper_count DESC, n.name ASC
                LIMIT $per_label_limit
                """
                results.extend(
                    dict(record)
                    for record in session.run(
                        query,
                        search_value=normalized_query,
                        per_label_limit=per_label_limit,
                    )
                )

        results.sort(
            key=lambda item: (
                0 if item["name"].lower() == search_term.lower() else 1,
                -item.get("paper_count", 0),
                item["type"],
                item["name"].lower(),
            )
        )
        deduped: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for result in results:
            key = (result["type"], result["name"].lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
            if len(deduped) >= limit:
                break
        return deduped

    def get_entity_context(
        self,
        entity_name: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Return papers and neighboring entities for one entity."""
        entity = self._fetch_entity_node(entity_name, entity_type=entity_type)
        if entity is None:
            return {"entity": None, "papers": [], "neighbors": []}

        with self.driver.session() as session:
            papers = [
                dict(record)
                for record in session.run(
                    """
                    MATCH (n) WHERE elementId(n) = $element_id
                    OPTIONAL MATCH (p:Paper)-[r]->(n)
                    RETURN DISTINCT p.title as title, p.year as year, p.authors as authors, type(r) as relation_type
                    ORDER BY p.year DESC, p.title ASC
                    LIMIT $limit
                    """,
                    element_id=entity["element_id"],
                    limit=limit,
                )
                if record["title"]
            ]
            direct_neighbors = [
                dict(record)
                for record in session.run(
                    """
                    MATCH (n) WHERE elementId(n) = $element_id
                    MATCH (n)-[r]-(neighbor)
                    WHERE NOT neighbor:Paper
                    RETURN DISTINCT neighbor.name as name, head(labels(neighbor)) as label, type(r) as relation_type,
                           null as via_paper
                    LIMIT $limit
                    """,
                    element_id=entity["element_id"],
                    limit=limit,
                )
            ]
            paper_neighbors = [
                dict(record)
                for record in session.run(
                    """
                    MATCH (n) WHERE elementId(n) = $element_id
                    MATCH (p:Paper)-[]->(n)
                    MATCH (p)-[r]->(neighbor)
                    WHERE neighbor <> n AND NOT neighbor:Paper
                    RETURN DISTINCT neighbor.name as name, head(labels(neighbor)) as label, type(r) as relation_type,
                           p.title as via_paper
                    LIMIT $limit
                    """,
                    element_id=entity["element_id"],
                    limit=limit,
                )
            ]

        neighbors: List[Dict[str, Any]] = []
        seen_neighbors: set[tuple[str, str, str, str | None]] = set()
        for row in direct_neighbors + paper_neighbors:
            neighbor_type = LABEL_TO_ENTITY_TYPE.get(row["label"], row["label"])
            key = (row["name"].lower(), neighbor_type, row["relation_type"], row.get("via_paper"))
            if key in seen_neighbors:
                continue
            seen_neighbors.add(key)
            neighbors.append(
                {
                    "name": row["name"],
                    "type": neighbor_type,
                    "relation_type": row["relation_type"],
                    "via_paper": row.get("via_paper"),
                }
            )
            if len(neighbors) >= limit:
                break

        return {"entity": entity, "papers": papers, "neighbors": neighbors}

    def get_relation_triples(self, relation_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Return triples for one fixed relation type."""
        normalized_relation = relation_type.upper().strip()
        if normalized_relation not in ALL_RELATION_TYPES:
            raise ValueError(f"Unsupported relation type: {relation_type}")

        query = f"""
        MATCH (source)-[r:{normalized_relation}]->(target)
        RETURN coalesce(source.title, source.name) as source,
               head(labels(source)) as source_label,
               type(r) as relation_type,
               coalesce(target.title, target.name) as target,
               head(labels(target)) as target_label
        ORDER BY source ASC, target ASC
        LIMIT $limit
        """
        with self.driver.session() as session:
            records = [dict(record) for record in session.run(query, limit=limit)]

        for record in records:
            record["source_type"] = LABEL_TO_ENTITY_TYPE.get(record.pop("source_label"), "PAPER")
            record["target_type"] = LABEL_TO_ENTITY_TYPE.get(record.pop("target_label"), "PAPER")
        return records

    def semantic_search(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search papers by title and abstract."""
        query_text = query_text.strip()
        if not query_text:
            return []

        with self.driver.session() as session:
            try:
                records = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('paper_text_search', $query_text)
                    YIELD node, score
                    RETURN node.paper_id as paper_id,
                           node.title as title,
                           node.year as year,
                           node.abstract as abstract,
                           node.authors as authors,
                           node.text as text,
                           score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    query_text=query_text,
                    limit=limit,
                )
                return [dict(record) for record in records]
            except Exception:
                records = session.run(
                    """
                    MATCH (p:Paper)
                    WITH p,
                         CASE
                           WHEN toLower(p.title) CONTAINS toLower($query_text) THEN 3.0
                           WHEN toLower(p.abstract) CONTAINS toLower($query_text) THEN 2.0
                           WHEN toLower(p.text) CONTAINS toLower($query_text) THEN 1.0
                           ELSE 0.0
                         END as score
                    WHERE score > 0
                    RETURN p.paper_id as paper_id,
                           p.title as title,
                           p.year as year,
                           p.abstract as abstract,
                           p.authors as authors,
                           p.text as text,
                           score
                    ORDER BY score DESC, p.year DESC
                    LIMIT $limit
                    """,
                    query_text=query_text,
                    limit=limit,
                )
                return [dict(record) for record in records]

    def find_entity_paths(
        self,
        entity1: str,
        entity2: str,
        max_depth: int = 3,
        entity1_type: str | None = None,
        entity2_type: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Find shortest paths between two entities."""
        source = self._fetch_entity_node(entity1, entity_type=entity1_type)
        target = self._fetch_entity_node(entity2, entity_type=entity2_type)
        if source is None or target is None:
            return []

        query = f"""
        MATCH (source) WHERE elementId(source) = $source_id
        MATCH (target) WHERE elementId(target) = $target_id
        MATCH path = shortestPath((source)-[*..{max_depth}]-(target))
        RETURN [n IN nodes(path) | {{
                  name: coalesce(n.title, n.name),
                  label: head(labels(n))
               }}] as nodes,
               [r IN relationships(path) | type(r)] as relationships,
               length(path) as path_length
        """
        with self.driver.session() as session:
            records = [dict(record) for record in session.run(query, source_id=source["element_id"], target_id=target["element_id"])]

        paths: List[Dict[str, Any]] = []
        for record in records:
            paths.append(
                {
                    "length": record["path_length"],
                    "nodes": [
                        {
                            "name": node["name"],
                            "type": LABEL_TO_ENTITY_TYPE.get(node["label"], "PAPER"),
                        }
                        for node in record["nodes"]
                    ],
                    "relationships": record["relationships"],
                }
            )
        return paths

    def get_available_relation_types(self) -> List[str]:
        return sorted(ALL_RELATION_TYPES)


if __name__ == "__main__":
    manager = Neo4jManager()
    if manager.connect():
        manager.setup_schema()
        print(manager.get_graph_statistics())
        manager.close()
