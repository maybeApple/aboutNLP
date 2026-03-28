"""
Streamlit interface for the paper knowledge base.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from config import DEFAULT_LLM_MODEL
from knowledge_graph_builder import GraphRAGQueryEngine, KnowledgeGraphPipeline
from neo4j_manager import ENTITY_LABELS, Neo4jManager


ENTITY_OPTIONS = ["AUTO"] + list(ENTITY_LABELS.keys())


class KnowledgeGraphQueryInterface:
    """Query wrapper for the paper knowledge base."""

    def __init__(self):
        self.pipeline: Optional[KnowledgeGraphPipeline] = None
        self.rag_engine: Optional[GraphRAGQueryEngine] = None
        self.logger = logging.getLogger(__name__)

    def initialize_pipeline(self, llm_provider: str = "openai", llm_model: str = DEFAULT_LLM_MODEL) -> bool:
        try:
            self.pipeline = KnowledgeGraphPipeline(llm_provider, llm_model)
            self.rag_engine = GraphRAGQueryEngine(self.pipeline)
            return True
        except Exception as exc:
            self.logger.error("Failed to initialize pipeline: %s", exc)
            return False

    def search_entities(self, query: str, entity_type: str | None = None, limit: int = 10) -> List[Dict[str, Any]]:
        if self.pipeline is None:
            raise ValueError("Pipeline not initialized")
        return self.pipeline.search_entities(query, limit=limit, entity_type=entity_type)

    def get_entity_context(
        self,
        entity_name: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        if self.pipeline is None:
            raise ValueError("Pipeline not initialized")
        return self.pipeline.get_entity_context(entity_name, entity_type=entity_type, limit=limit)

    def get_relation_triples(self, relation_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        if self.pipeline is None:
            raise ValueError("Pipeline not initialized")
        return self.pipeline.get_relation_triples(relation_type, limit=limit)

    def query_with_context(self, question: str, context_limit: int = 5) -> Dict[str, Any]:
        if self.rag_engine is None:
            raise ValueError("RAG engine not initialized")
        return self.rag_engine.query_with_context(question, context_limit)

    def get_graph_statistics(self) -> Dict[str, Any]:
        if self.pipeline is None:
            raise ValueError("Pipeline not initialized")
        return self.pipeline.get_graph_statistics()

    def get_available_relation_types(self) -> List[str]:
        if self.pipeline is None:
            raise ValueError("Pipeline not initialized")
        return self.pipeline.get_available_relation_types()


class StreamlitWebInterface:
    """Paper knowledge base web UI."""

    def __init__(self):
        self.interface = KnowledgeGraphQueryInterface()
        self.setup_page_config()

    def setup_page_config(self) -> None:
        st.set_page_config(
            page_title="Paper Knowledge Base",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def run(self) -> None:
        st.title("Paper Knowledge Base")
        st.caption("Query papers, methods, tasks, datasets, metrics, and results stored in Neo4j.")

        self.render_sidebar()
        if self.interface.pipeline is None and not self.interface.initialize_pipeline():
            st.error("Failed to initialize the paper knowledge base pipeline.")
            return

        self.render_overview()
        tab1, tab2, tab3 = st.tabs(
            [
                "Entity Query",
                "Relation Query",
                "Natural Language Query",
            ]
        )
        with tab1:
            self.render_entity_query()
        with tab2:
            self.render_relation_query()
        with tab3:
            self.render_natural_language_query()

    def render_sidebar(self) -> None:
        with st.sidebar:
            st.header("System")
            provider = st.selectbox("LLM Provider", ["openai", "dashscope", "anthropic"], index=0)
            if provider == "openai":
                model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"], index=0)
            elif provider == "dashscope":
                model = st.selectbox("Model", ["qwen3-max", "qwen-plus", "qwen-flash"], index=0)
            else:
                model = st.selectbox(
                    "Model",
                    ["claude-3-sonnet-20240229", "claude-3-opus-20240229"],
                    index=0,
                )

            if st.button("Initialize", type="primary"):
                if self.interface.initialize_pipeline(provider, model):
                    st.success("Pipeline initialized.")
                else:
                    st.error("Pipeline initialization failed.")

            st.divider()
            st.subheader("Database Status")
            manager = Neo4jManager()
            if manager.connect():
                stats = manager.get_graph_statistics()
                manager.close()
                st.metric("Papers", stats.get("total_papers", 0))
                st.metric("Entities", stats.get("total_entities", 0))
                st.metric("Relations", stats.get("total_relations", 0))
            else:
                st.error("Cannot connect to Neo4j.")

    def render_overview(self) -> None:
        try:
            stats = self.interface.get_graph_statistics()
        except Exception as exc:
            st.error(f"Failed to load graph statistics: {exc}")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Papers", stats.get("total_papers", 0))
        col2.metric("Total Entities", stats.get("total_entities", 0))
        col3.metric("Total Relations", stats.get("total_relations", 0))

        breakdown = stats.get("entity_breakdown") or []
        if breakdown:
            st.subheader("Entity Breakdown")
            st.dataframe(pd.DataFrame(breakdown), hide_index=True, use_container_width=True)

    def render_entity_query(self) -> None:
        st.header("Entity-Centric Query")
        st.write("Input a task, method, dataset, metric, or result name to inspect related papers and neighbors.")

        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            query = st.text_input("Entity Name", placeholder="e.g. text summarization")
        with col2:
            entity_type = st.selectbox("Entity Type", ENTITY_OPTIONS, index=0)
        with col3:
            limit = st.number_input("Limit", min_value=5, max_value=50, value=10)

        if st.button("Search Entity", key="entity_search_button"):
            if not query.strip():
                st.warning("Enter an entity name first.")
                return

            resolved_type = None if entity_type == "AUTO" else entity_type
            matches = self.interface.search_entities(query, entity_type=resolved_type, limit=int(limit))
            if not matches:
                st.info("No matching entities found.")
                return

            st.subheader("Matches")
            matches_df = pd.DataFrame(matches)
            st.dataframe(matches_df[["name", "type", "paper_count", "description"]], hide_index=True, use_container_width=True)

            selected = matches[0]
            context = self.interface.get_entity_context(
                selected["name"],
                entity_type=selected["type"],
                limit=int(limit),
            )

            st.subheader(f"Related Papers for {selected['name']}")
            papers = context.get("papers") or []
            if papers:
                st.dataframe(pd.DataFrame(papers), hide_index=True, use_container_width=True)
            else:
                st.info("No directly linked papers found.")

            st.subheader("Neighboring Entities")
            neighbors = context.get("neighbors") or []
            if neighbors:
                st.dataframe(pd.DataFrame(neighbors), hide_index=True, use_container_width=True)
            else:
                st.info("No neighboring entities found.")

    def render_relation_query(self) -> None:
        st.header("Relation Query")
        st.write("Browse fixed graph relations such as PROPOSES, USES_DATASET, and EVALUATED_BY.")

        relation_types = self.interface.get_available_relation_types()
        col1, col2 = st.columns([2, 1])
        with col1:
            relation_type = st.selectbox("Relation Type", relation_types, index=0)
        with col2:
            limit = st.number_input("Triples Limit", min_value=10, max_value=200, value=50, key="relation_limit")

        triples = self.interface.get_relation_triples(relation_type, limit=int(limit))
        if not triples:
            st.info("No triples found for this relation type.")
            return

        triples_df = pd.DataFrame(triples)
        st.dataframe(triples_df, hide_index=True, use_container_width=True)

    def render_natural_language_query(self) -> None:
        st.header("Natural Language Query")
        st.write("Ask free-form questions over the paper knowledge base. This view retrieves relevant papers, entities, and supporting graph context.")

        question = st.text_area(
            "Question",
            placeholder="What methods are used for text summarization?",
            height=120,
        )
        context_limit = st.slider("Relevant Papers", min_value=3, max_value=10, value=5)

        if st.button("Run Query", key="natural_language_button"):
            if not question.strip():
                st.warning("Enter a natural language question first.")
                return

            result = self.interface.query_with_context(question, context_limit=context_limit)

            st.subheader("Relevant Papers")
            papers = result.get("relevant_papers") or []
            if papers:
                paper_rows = [
                    {
                        "title": paper.get("title"),
                        "year": paper.get("year"),
                        "score": paper.get("score"),
                        "authors": ", ".join(paper.get("authors") or []),
                        "abstract": paper.get("abstract"),
                    }
                    for paper in papers
                ]
                st.dataframe(pd.DataFrame(paper_rows), hide_index=True, use_container_width=True)
            else:
                st.info("No relevant papers found.")

            st.subheader("Related Entities")
            entities = result.get("related_entities") or []
            if entities:
                st.dataframe(pd.DataFrame(entities), hide_index=True, use_container_width=True)
            else:
                st.info("No related entities found.")

            st.subheader("Supporting Triples")
            triples = result.get("supporting_triples") or []
            if triples:
                st.dataframe(pd.DataFrame(triples), hide_index=True, use_container_width=True)
            else:
                st.info("No supporting triples found.")

            st.subheader("Context Preview")
            st.text_area("Context", value=result.get("context_text", ""), height=220, disabled=True)


def run_web_interface() -> None:
    interface = StreamlitWebInterface()
    interface.run()


if __name__ == "__main__":
    run_web_interface()
