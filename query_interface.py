"""
Streamlit interface for the paper knowledge base.
"""
from __future__ import annotations

import html
import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from config import DEFAULT_LLM_MODEL
from knowledge_graph_builder import GraphRAGQueryEngine, KnowledgeGraphPipeline
from neo4j_manager import ENTITY_LABELS, Neo4jManager


ENTITY_OPTIONS = ["AUTO"] + list(ENTITY_LABELS.keys())
MODEL_OPTIONS = {
    "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
    "dashscope": ["qwen3-max", "qwen-plus", "qwen-flash"],
    "anthropic": ["claude-3-sonnet-20240229", "claude-3-opus-20240229"],
}


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


@st.cache_resource(show_spinner=False)
def get_initialized_interface(
    llm_provider: str,
    llm_model: str,
    refresh_token: int,
) -> KnowledgeGraphQueryInterface:
    """Cache the query interface per provider/model until an explicit refresh."""
    _ = refresh_token
    interface = KnowledgeGraphQueryInterface()
    if not interface.initialize_pipeline(llm_provider, llm_model):
        raise RuntimeError(f"Failed to initialize pipeline for {llm_provider}/{llm_model}.")
    return interface


class StreamlitWebInterface:
    """Paper knowledge base web UI."""

    def __init__(self):
        self.interface: Optional[KnowledgeGraphQueryInterface] = None
        self.setup_page_config()
        self.initialize_session_state()

    def setup_page_config(self) -> None:
        st.set_page_config(
            page_title="Paper Knowledge Base",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def initialize_session_state(self) -> None:
        default_model = DEFAULT_LLM_MODEL if DEFAULT_LLM_MODEL in MODEL_OPTIONS["openai"] else MODEL_OPTIONS["openai"][0]
        defaults = {
            "selected_provider": "openai",
            "selected_model": default_model,
            "pipeline_refresh": 0,
            "entity_search_results": [],
            "entity_selected_key": None,
            "entity_context_limit": 10,
            "relation_query_result": [],
            "relation_query_type": None,
            "natural_language_result": None,
        }
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)

    def run(self) -> None:
        self.inject_styles()
        st.title("Paper Knowledge Base")
        st.caption("Graph-backed paper search for papers, methods, tasks, datasets, metrics, and results stored in Neo4j.")

        st.markdown(
            """
            <div class="kb-panel kb-panel-hero">
              <div class="kb-kicker">Graph-backed retrieval workspace</div>
              <div class="kb-hero">Search by entity, browse fixed relations, or ask a natural-language question and inspect the exact evidence that was retrieved.</div>
              <div class="kb-copy">The interface now keeps your active model selection stable across reruns and makes the context window readable instead of burying it in one long text box.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not self.render_sidebar():
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

    def inject_styles(self) -> None:
        st.markdown(
            """
            <style>
            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(125, 181, 225, 0.18), transparent 22rem),
                    linear-gradient(180deg, #f5fbff 0%, #ffffff 28%, #f7f2e9 100%);
            }
            html, body, [class*="css"] {
                font-family: "Segoe UI", "Trebuchet MS", sans-serif;
            }
            h1, h2, h3, .kb-hero {
                font-family: Georgia, "Palatino Linotype", serif;
                color: #10243d;
                letter-spacing: -0.02em;
            }
            .block-container {
                padding-top: 1.8rem;
                padding-bottom: 3rem;
            }
            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.76);
                border: 1px solid #d9e4ee;
                border-radius: 18px;
                padding: 0.75rem 0.9rem;
                box-shadow: 0 14px 28px rgba(16, 36, 61, 0.06);
            }
            div[data-baseweb="tab-list"] {
                gap: 0.35rem;
            }
            button[role="tab"] {
                border-radius: 999px;
                border: 1px solid #d7e2eb;
                background: rgba(255, 255, 255, 0.85);
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .kb-panel {
                background: rgba(255, 255, 255, 0.86);
                border: 1px solid #dce5ee;
                border-radius: 20px;
                padding: 1rem 1.15rem;
                margin: 0.35rem 0 1rem 0;
                box-shadow: 0 18px 32px rgba(16, 36, 61, 0.05);
            }
            .kb-panel-hero {
                background:
                    linear-gradient(135deg, rgba(232, 243, 255, 0.98), rgba(255, 250, 241, 0.98));
            }
            .kb-kicker {
                color: #406b92;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                margin-bottom: 0.4rem;
                text-transform: uppercase;
            }
            .kb-hero {
                font-size: 1.35rem;
                line-height: 1.25;
                margin-bottom: 0.45rem;
            }
            .kb-copy {
                color: #43576b;
                line-height: 1.5;
            }
            .kb-chip-row {
                margin-top: 0.25rem;
            }
            .kb-chip {
                display: inline-block;
                padding: 0.22rem 0.65rem;
                margin: 0 0.4rem 0.4rem 0;
                border-radius: 999px;
                background: #edf5ff;
                border: 1px solid #cfe0f4;
                color: #1f5380;
                font-size: 0.82rem;
                font-weight: 600;
            }
            .context-card {
                border: 1px solid #dbe5ee;
                border-radius: 20px;
                background: rgba(255, 255, 255, 0.9);
                padding: 1rem 1.05rem;
                margin-bottom: 0.85rem;
                box-shadow: 0 16px 30px rgba(16, 36, 61, 0.05);
            }
            .context-card__eyebrow {
                color: #5d7388;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                margin-bottom: 0.4rem;
                text-transform: uppercase;
            }
            .context-card__title {
                color: #10243d;
                font-family: Georgia, "Palatino Linotype", serif;
                font-size: 1.1rem;
                line-height: 1.25;
                margin-bottom: 0.45rem;
            }
            .context-card__meta {
                color: #4e6173;
                font-size: 0.9rem;
                margin-bottom: 0.55rem;
            }
            .context-card__body {
                color: #29384a;
                line-height: 1.55;
            }
            .kb-raw-context {
                white-space: pre-wrap;
                word-break: break-word;
                background: #0d1d31;
                color: #eef5ff;
                border-radius: 18px;
                padding: 1rem;
                margin-top: 0.6rem;
                max-height: 28rem;
                overflow: auto;
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def render_sidebar(self) -> bool:
        with st.sidebar:
            st.header("System")
            provider = st.selectbox("LLM Provider", list(MODEL_OPTIONS.keys()), key="selected_provider")
            available_models = MODEL_OPTIONS[provider]
            if st.session_state.get("selected_model") not in available_models:
                st.session_state["selected_model"] = (
                    DEFAULT_LLM_MODEL if DEFAULT_LLM_MODEL in available_models else available_models[0]
                )
            model = st.selectbox("Model", available_models, key="selected_model")

            if st.button("Reconnect Pipeline", type="primary"):
                st.session_state["pipeline_refresh"] += 1

            try:
                self.interface = get_initialized_interface(
                    provider,
                    model,
                    st.session_state["pipeline_refresh"],
                )
                st.caption(f"Active pipeline: {provider}/{model}")
                st.caption("The selected provider/model now stays stable across reruns.")
                initialized = True
            except Exception as exc:
                self.interface = None
                st.error(f"Pipeline initialization failed: {exc}")
                initialized = False

            st.divider()
            st.subheader("Database Status")
            stats = self.load_database_stats()
            if stats:
                st.metric("Papers", stats.get("total_papers", 0))
                st.metric("Entities", stats.get("total_entities", 0))
                st.metric("Relations", stats.get("total_relations", 0))
            else:
                st.error("Cannot connect to Neo4j.")

        return initialized

    def load_database_stats(self) -> Dict[str, Any] | None:
        manager = Neo4jManager()
        try:
            if not manager.connect():
                return None
            return manager.get_graph_statistics()
        except Exception:
            return None
        finally:
            try:
                manager.close()
            except Exception:
                pass

    def render_overview(self) -> None:
        if self.interface is None:
            st.error("Pipeline is not ready.")
            return

        try:
            stats = self.interface.get_graph_statistics()
        except Exception as exc:
            st.error(f"Failed to load graph statistics: {exc}")
            return

        provider = st.session_state.get("selected_provider", "openai")
        model = st.session_state.get("selected_model", DEFAULT_LLM_MODEL)
        st.markdown(
            f"""
            <div class="kb-panel">
              <div class="kb-kicker">Session overview</div>
              <div class="kb-copy">Active retrieval stack: <strong>{html.escape(provider)}/{html.escape(model)}</strong>. Use the tabs below to move between graph browsing and question-driven context retrieval without losing state.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Papers", stats.get("total_papers", 0))
        col2.metric("Total Entities", stats.get("total_entities", 0))
        col3.metric("Total Relations", stats.get("total_relations", 0))

        chart_left, chart_right = st.columns([1.05, 0.95])
        breakdown = pd.DataFrame(stats.get("entity_breakdown") or [])
        with chart_left:
            st.subheader("Entity Breakdown")
            if breakdown.empty:
                st.info("No entity counts available.")
            else:
                st.bar_chart(breakdown.set_index("type")["count"], use_container_width=True)
                st.dataframe(breakdown, hide_index=True, use_container_width=True)

        papers_by_year = pd.DataFrame(stats.get("papers_by_year") or [])
        with chart_right:
            st.subheader("Papers by Year")
            if papers_by_year.empty:
                st.info("No paper timeline available.")
            else:
                papers_by_year["sort_year"] = pd.to_numeric(papers_by_year["year"], errors="coerce").fillna(-1)
                chart_df = papers_by_year.sort_values("sort_year").drop(columns=["sort_year"]).set_index("year")
                st.bar_chart(chart_df["count"], use_container_width=True)
                st.dataframe(chart_df.reset_index(), hide_index=True, use_container_width=True)

    def render_entity_query(self) -> None:
        if self.interface is None:
            return

        st.header("Entity-Centric Query")
        st.write("Search for a task, method, dataset, metric, or result, then inspect related papers and neighboring entities.")

        with st.form("entity_query_form"):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                query = st.text_input("Entity Name", placeholder="e.g. text summarization")
            with col2:
                entity_type = st.selectbox("Entity Type", ENTITY_OPTIONS, index=0)
            with col3:
                limit = st.number_input("Limit", min_value=5, max_value=50, value=10)
            submitted = st.form_submit_button("Search Entity", type="primary")

        if submitted:
            if not query.strip():
                st.warning("Enter an entity name first.")
            else:
                resolved_type = None if entity_type == "AUTO" else entity_type
                with st.spinner("Searching entities and graph context..."):
                    matches = self.interface.search_entities(query, entity_type=resolved_type, limit=int(limit))
                st.session_state["entity_search_results"] = matches
                st.session_state["entity_context_limit"] = int(limit)
                st.session_state["entity_selected_key"] = self.match_key(matches[0]) if matches else None

        matches = st.session_state.get("entity_search_results") or []
        if not matches:
            self.render_hint_panel(
                "No entity selected yet.",
                "Run a search to inspect the graph neighborhood around a task, method, dataset, metric, or result.",
            )
            return

        options_by_key = {self.match_key(match): match for match in matches}
        option_keys = list(options_by_key.keys())
        if st.session_state.get("entity_selected_key") not in options_by_key:
            st.session_state["entity_selected_key"] = option_keys[0]

        st.subheader("Matches")
        st.dataframe(
            self.records_to_dataframe(matches, ["name", "type", "paper_count", "description"]),
            hide_index=True,
            use_container_width=True,
        )

        selected_key = st.selectbox(
            "Inspect a matched entity",
            option_keys,
            format_func=lambda key: self.format_entity_match(options_by_key[key]),
            key="entity_selected_key",
        )
        selected = options_by_key[selected_key]
        context_limit = int(st.session_state.get("entity_context_limit", 10))
        context = self.interface.get_entity_context(
            selected["name"],
            entity_type=selected["type"],
            limit=context_limit,
        )

        papers = context.get("papers") or []
        neighbors = context.get("neighbors") or []
        stat1, stat2, stat3 = st.columns(3)
        stat1.metric("Matches", len(matches))
        stat2.metric("Linked Papers", len(papers))
        stat3.metric("Neighbors", len(neighbors))

        description = self.safe_text(selected.get("description"), "No description available.")
        selected_type = self.safe_text(selected.get("type"), "ENTITY")
        selected_name = self.safe_text(selected.get("name"), "Unknown entity")
        st.markdown(
            f"""
            <div class="kb-panel">
              <div class="kb-kicker">{html.escape(selected_type)}</div>
              <div class="kb-hero">{html.escape(selected_name)}</div>
              <div class="kb-copy">{html.escape(description)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        papers_tab, neighbors_tab = st.tabs(["Related Papers", "Neighboring Entities"])
        with papers_tab:
            if papers:
                paper_rows = [
                    {
                        "title": paper.get("title"),
                        "year": paper.get("year"),
                        "relation_type": paper.get("relation_type"),
                        "authors": self.format_authors(paper.get("authors")),
                    }
                    for paper in papers
                ]
                st.dataframe(pd.DataFrame(paper_rows), hide_index=True, use_container_width=True)
            else:
                st.info("No directly linked papers found.")

        with neighbors_tab:
            if neighbors:
                st.dataframe(
                    self.records_to_dataframe(neighbors, ["name", "type", "relation_type", "via_paper"]),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No neighboring entities found.")

    def render_relation_query(self) -> None:
        if self.interface is None:
            return

        st.header("Relation Query")
        st.write("Browse fixed graph relations such as PROPOSES, USES_DATASET, and EVALUATED_BY.")

        relation_types = self.interface.get_available_relation_types()
        if not relation_types:
            st.info("No relation types are available.")
            return

        with st.form("relation_query_form"):
            col1, col2 = st.columns([2, 1])
            with col1:
                relation_type = st.selectbox("Relation Type", relation_types, index=0)
            with col2:
                limit = st.number_input("Triples Limit", min_value=10, max_value=200, value=50, key="relation_limit")
            submitted = st.form_submit_button("Fetch Triples", type="primary")

        if submitted:
            with st.spinner("Loading relation triples..."):
                triples = self.interface.get_relation_triples(relation_type, limit=int(limit))
            st.session_state["relation_query_result"] = triples
            st.session_state["relation_query_type"] = relation_type

        triples = st.session_state.get("relation_query_result") or []
        active_relation = st.session_state.get("relation_query_type") or relation_type
        if not triples:
            self.render_hint_panel(
                "No relation triples loaded yet.",
                "Choose a relation type and fetch triples to inspect the graph as source-relation-target records.",
            )
            return

        source_count = len({(row.get("source"), row.get("source_type")) for row in triples})
        target_count = len({(row.get("target"), row.get("target_type")) for row in triples})
        stat1, stat2, stat3 = st.columns(3)
        stat1.metric("Relation Type", active_relation)
        stat2.metric("Unique Sources", source_count)
        stat3.metric("Unique Targets", target_count)

        st.dataframe(
            self.records_to_dataframe(triples, ["source", "source_type", "relation_type", "target", "target_type"]),
            hide_index=True,
            use_container_width=True,
        )

    def render_natural_language_query(self) -> None:
        if self.interface is None:
            return

        st.header("Natural Language Query")
        st.write("Ask a free-form question, then inspect the retrieved papers, graph entities, supporting triples, and the exact context text used for answer grounding.")

        with st.form("natural_language_query_form"):
            question = st.text_area(
                "Question",
                placeholder="What methods are used for text summarization?",
                height=120,
            )
            context_limit = st.slider("Relevant Papers", min_value=3, max_value=10, value=5)
            submitted = st.form_submit_button("Run Query", type="primary")

        if submitted:
            if not question.strip():
                st.warning("Enter a natural language question first.")
            else:
                with st.spinner("Retrieving papers, entities, and graph context..."):
                    result = self.interface.query_with_context(question, context_limit=context_limit)
                st.session_state["natural_language_result"] = result

        result = st.session_state.get("natural_language_result")
        if not result:
            self.render_hint_panel(
                "No natural-language query run yet.",
                "Ask a question to see retrieved evidence, structured entities, triples, and a readable context preview.",
            )
            return

        papers = result.get("relevant_papers") or []
        entities = result.get("related_entities") or []
        triples = result.get("supporting_triples") or []
        question_text = self.safe_text(result.get("question"), "Unknown question")

        stat1, stat2, stat3 = st.columns(3)
        stat1.metric("Relevant Papers", len(papers))
        stat2.metric("Related Entities", len(entities))
        stat3.metric("Supporting Triples", len(triples))

        overview_tab, papers_tab, entities_tab, triples_tab, raw_tab = st.tabs(
            ["Overview", "Papers", "Entities", "Triples", "Raw Context"]
        )

        with overview_tab:
            left, right = st.columns([1.15, 0.85])
            with left:
                st.markdown(
                    f"""
                    <div class="kb-panel">
                      <div class="kb-kicker">Question</div>
                      <div class="kb-hero">{html.escape(question_text)}</div>
                      <div class="kb-copy">Top retrieved entities and source papers are summarized below. Use the tabs to switch from the visual digest to raw evidence tables.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("**Entity Focus**")
                self.render_chip_cloud(
                    [f"{entity.get('type', 'ENTITY')}: {entity.get('name', 'Unknown')}" for entity in entities[:8]],
                    empty_message="No entity matches were retrieved.",
                )

            with right:
                relation_counter = Counter(triple.get("relation_type", "UNKNOWN") for triple in triples if triple.get("relation_type"))
                st.markdown("**Relation Signals**")
                self.render_chip_cloud(
                    [f"{relation} x{count}" for relation, count in relation_counter.most_common(6)],
                    empty_message="No supporting relations were retrieved.",
                )
                year_values = [
                    int(year)
                    for year in pd.to_numeric([paper.get("year") for paper in papers], errors="coerce")
                    if pd.notna(year)
                ]
                year_summary = f"{min(year_values)} to {max(year_values)}" if year_values else "Unknown"
                st.markdown(
                    f"""
                    <div class="kb-panel">
                      <div class="kb-kicker">Context Window</div>
                      <div class="kb-copy">Retrieved paper span: <strong>{html.escape(str(year_summary))}</strong><br/>Top source count: <strong>{len(papers)}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.subheader("Context Preview")
            self.render_paper_cards(papers, max_cards=3)
            if len(papers) > 3:
                st.caption(f"{len(papers) - 3} more papers are available in the Papers tab.")

        with papers_tab:
            if papers:
                paper_rows = [
                    {
                        "title": paper.get("title"),
                        "year": paper.get("year"),
                        "score": self.format_score(paper.get("score")),
                        "authors": self.format_authors(paper.get("authors")),
                        "abstract_preview": self.truncate_text(paper.get("abstract"), limit=180),
                    }
                    for paper in papers
                ]
                st.dataframe(pd.DataFrame(paper_rows), hide_index=True, use_container_width=True)
                st.markdown("**Paper Details**")
                self.render_paper_cards(papers)
            else:
                st.info("No relevant papers found.")

        with entities_tab:
            if entities:
                st.markdown("**Entity Digest**")
                self.render_chip_cloud(
                    [f"{entity.get('type', 'ENTITY')}: {entity.get('name', 'Unknown')}" for entity in entities],
                    empty_message=None,
                )
                st.dataframe(
                    self.records_to_dataframe(entities, ["name", "type", "paper_count", "description"]),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No related entities found.")

        with triples_tab:
            if triples:
                st.dataframe(
                    self.records_to_dataframe(
                        triples,
                        ["source", "source_type", "relation_type", "target", "target_type"],
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No supporting triples found.")

        with raw_tab:
            raw_context = self.safe_text(result.get("context_text"), "No context text available.")
            st.markdown(
                """
                <div class="kb-panel">
                  <div class="kb-kicker">Prompt-ready context</div>
                  <div class="kb-copy">This is the exact stitched context string produced from the retrieved papers. Keep this view when you need the raw retrieval window instead of the visual digest.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"<pre class='kb-raw-context'>{html.escape(raw_context)}</pre>", unsafe_allow_html=True)

    def render_hint_panel(self, title: str, message: str) -> None:
        st.markdown(
            f"""
            <div class="kb-panel">
              <div class="kb-kicker">Waiting for input</div>
              <div class="kb-hero">{html.escape(title)}</div>
              <div class="kb-copy">{html.escape(message)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def render_chip_cloud(self, items: List[str], empty_message: str | None = None) -> None:
        deduped_items: List[str] = []
        seen: set[str] = set()
        for item in items:
            label = self.safe_text(item)
            if not label:
                continue
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped_items.append(label)

        if not deduped_items:
            if empty_message:
                st.caption(empty_message)
            return

        markup = "".join(f"<span class='kb-chip'>{html.escape(item)}</span>" for item in deduped_items)
        st.markdown(f"<div class='kb-chip-row'>{markup}</div>", unsafe_allow_html=True)

    def render_paper_cards(self, papers: List[Dict[str, Any]], max_cards: int | None = None) -> None:
        if not papers:
            st.info("No paper context is available.")
            return

        visible_papers = papers[:max_cards] if max_cards is not None else papers
        for index, paper in enumerate(visible_papers, start=1):
            title = self.safe_text(paper.get("title"), "Untitled paper")
            year = self.safe_text(paper.get("year"), "Unknown year")
            score = self.format_score(paper.get("score"))
            authors = self.format_authors(paper.get("authors"))
            abstract = self.safe_text(paper.get("abstract"), "No abstract available.")
            preview = self.truncate_text(abstract, limit=360)

            st.markdown(
                f"""
                <div class="context-card">
                  <div class="context-card__eyebrow">Source {index} <span class="kb-chip">{html.escape(year)}</span> <span class="kb-chip">score {html.escape(score)}</span></div>
                  <div class="context-card__title">{html.escape(title)}</div>
                  <div class="context-card__meta">{html.escape(authors)}</div>
                  <div class="context-card__body">{html.escape(preview)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if abstract != "No abstract available.":
                with st.expander(f"Open full abstract for {title}"):
                    st.write(abstract)

    @staticmethod
    def safe_text(value: Any, fallback: str = "") -> str:
        if value is None:
            return fallback
        text = str(value).strip()
        return text or fallback

    @staticmethod
    def truncate_text(value: Any, limit: int = 260) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3].rstrip()}..."

    @staticmethod
    def format_score(score: Any) -> str:
        try:
            return f"{float(score):.3f}"
        except (TypeError, ValueError):
            return "n/a"

    @staticmethod
    def format_authors(authors: Any) -> str:
        if isinstance(authors, (list, tuple, set)):
            values = [str(author).strip() for author in authors if str(author).strip()]
            return ", ".join(values) if values else "Authors unavailable"
        text = str(authors or "").strip()
        return text or "Authors unavailable"

    @staticmethod
    def match_key(match: Dict[str, Any]) -> str:
        entity_type = str(match.get("type", "UNKNOWN")).strip()
        name = str(match.get("name", "")).strip()
        return f"{entity_type}::{name}"

    @staticmethod
    def format_entity_match(match: Dict[str, Any]) -> str:
        paper_count = match.get("paper_count", 0)
        return f"{match.get('type', 'UNKNOWN')} | {match.get('name', 'Unknown')} ({paper_count} papers)"

    @staticmethod
    def records_to_dataframe(records: List[Dict[str, Any]], columns: List[str]) -> pd.DataFrame:
        return pd.DataFrame(records).reindex(columns=columns)


def run_web_interface() -> None:
    interface = StreamlitWebInterface()
    interface.run()


if __name__ == "__main__":
    run_web_interface()
