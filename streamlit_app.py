"""Streamlit deployment entry point for EKGA."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from src.agents import Orchestrator


os.environ.setdefault("EKGA_DEMO_MODE", "true")

st.set_page_config(
    page_title="EKGA | Enterprise Knowledge Graph Agent",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      :root {
        --bg: #080b16;
        --panel: rgba(16, 22, 40, 0.78);
        --line: rgba(148, 163, 184, 0.18);
        --text: #eef4ff;
        --muted: #9fb0ca;
        --cyan: #22d3ee;
        --blue: #60a5fa;
        --violet: #a78bfa;
        --green: #34d399;
      }

      .stApp {
        color: var(--text);
        background:
          radial-gradient(circle at 12% 10%, rgba(96, 165, 250, 0.22), transparent 30rem),
          radial-gradient(circle at 88% 8%, rgba(167, 139, 250, 0.20), transparent 26rem),
          linear-gradient(135deg, #070a13 0%, #0a1022 48%, #0c1224 100%);
      }

      [data-testid="stHeader"] {
        background: transparent;
      }

      section[data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.68);
        border-right: 1px solid var(--line);
      }

      .hero-card, .glass-card, .metric-card {
        border: 1px solid var(--line);
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.80), rgba(15, 23, 42, 0.42));
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(22px);
        padding: 1.4rem;
      }

      .hero-title {
        margin: 0;
        font-size: clamp(2.4rem, 6vw, 4.8rem);
        line-height: 0.95;
        letter-spacing: -0.07em;
        font-weight: 850;
      }

      .eyebrow {
        color: var(--cyan);
        font-size: 0.74rem;
        font-weight: 850;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
      }

      .subcopy {
        color: var(--muted);
        max-width: 800px;
        line-height: 1.7;
        font-size: 1.05rem;
      }

      .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
      }

      .badge {
        border: 1px solid rgba(34, 211, 238, 0.24);
        border-radius: 999px;
        padding: 0.45rem 0.65rem;
        color: #cffafe;
        background: rgba(8, 145, 178, 0.12);
        font-size: 0.82rem;
        font-weight: 800;
      }

      .value-card {
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1rem;
        height: 100%;
        background: rgba(15, 23, 42, 0.58);
      }

      .value-card strong {
        display: block;
        margin-bottom: 0.45rem;
        font-size: 1rem;
      }

      .value-card p {
        color: var(--muted);
        margin: 0;
        line-height: 1.55;
      }

      .answer-box {
        white-space: pre-line;
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.1rem;
        background: rgba(2, 6, 23, 0.52);
        line-height: 1.75;
      }

      .citation {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.9rem;
        margin-bottom: 0.65rem;
        background: rgba(15, 23, 42, 0.55);
      }

      .citation small, .trace small {
        color: var(--muted);
      }

      .trace {
        border-left: 3px solid var(--green);
        padding-left: 0.85rem;
        margin-bottom: 0.85rem;
      }

      div[data-testid="stMetricValue"] {
        color: #e0f2fe;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


def run_query(query: str, top_k: int) -> dict[str, Any]:
    return get_orchestrator().query(query, top_k=top_k, session_id="streamlit-demo").to_dict()


def summarize_hop(agent_name: str, metadata: dict[str, Any]) -> str:
    if agent_name == "router":
        return f"classified as {metadata.get('query_type', 'unknown')}"
    if agent_name == "retrieval":
        return f"{metadata.get('retrieval_path', 'none')} retrieval, {len(metadata.get('chunks', []))} evidence chunks"
    if agent_name == "synthesis":
        return f"grounded answer with {len(metadata.get('citations', []))} citations"
    if agent_name == "eval":
        return f"grounded: {'yes' if metadata.get('grounded') else 'needs data'}"
    return "completed"


with st.sidebar:
    st.markdown("## EKGA")
    st.caption("Enterprise Knowledge Graph Agent")
    st.markdown("---")
    st.markdown("### Demo Settings")
    top_k = st.slider("Evidence chunks", min_value=3, max_value=10, value=5)
    st.toggle("Demo knowledge fallback", value=True, disabled=True)
    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown(
        """
        - Router Agent
        - Hybrid Retrieval
        - Graph Query Layer
        - Grounded Synthesis
        - Inline Evaluation
        - MCP/API Ready
        """
    )
    st.info("Demo mode stays presentation-ready even without live AlloyDB or Neo4j credentials.")


st.markdown(
    """
    <div class="hero-card">
      <div class="eyebrow">Google Cloud GenAI FDE Demo</div>
      <h1 class="hero-title">Enterprise memory, retrieved as a graph.</h1>
      <p class="subcopy">
        EKGA turns siloed Confluence pages, Jira tickets, Slack decisions, and PDFs into
        cited answers with relationship-aware retrieval, agent tracing, and evaluation.
      </p>
      <div class="badge-row">
        <span class="badge">AlloyDB pgvector</span>
        <span class="badge">Neo4j graph</span>
        <span class="badge">Vertex AI ready</span>
        <span class="badge">Streamlit Cloud</span>
        <span class="badge">MCP API</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

value_cols = st.columns(3)
with value_cols[0]:
    st.markdown(
        """
        <div class="value-card">
          <strong>Cross-source reasoning</strong>
          <p>Follows relationships between docs, authors, tickets, services, owners, and decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with value_cols[1]:
    st.markdown(
        """
        <div class="value-card">
          <strong>Enterprise trust</strong>
          <p>Every response shows citations, retrieval path, faithfulness score, and agent trace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with value_cols[2]:
    st.markdown(
        """
        <div class="value-card">
          <strong>Field-ready story</strong>
          <p>Designed for customer discovery: prove value first, then connect production data sources.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

query_options = [
    "Why was the auth service migrated off PostgreSQL?",
    "What does onboarding for a backend engineer look like?",
    "Which team owns the rate-limiting module?",
    "What value does this project create for a 600 engineer enterprise?",
]

left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.markdown('<div class="eyebrow">Ask the knowledge graph</div>', unsafe_allow_html=True)
    selected = st.selectbox("Sample prompts", query_options, label_visibility="collapsed")
    query = st.text_area(
        "Query",
        value=selected,
        height=160,
        placeholder="Ask about a decision, owner, blocker, runbook, or enterprise process...",
    )
    ask = st.button("Ask EKGA", type="primary", use_container_width=True)

with right:
    st.markdown('<div class="eyebrow">Agent output</div>', unsafe_allow_html=True)
    if ask and query.strip():
        with st.spinner("Running router, retrieval, synthesis, and eval agents..."):
            result = run_query(query.strip(), top_k)

        metric_cols = st.columns(4)
        metric_cols[0].metric("Route", result["query_type"])
        metric_cols[1].metric("Retrieval", result["retrieval_path"])
        metric_cols[2].metric("Latency", f'{result["latency_ms"]} ms')
        metric_cols[3].metric("Faithfulness", f'{round(result["eval"].get("faithfulness", 0) * 100)}%')

        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

        st.markdown("#### Citations")
        for citation in result["citations"]:
            st.markdown(
                f"""
                <div class="citation">
                  <strong>{citation.get("title", "Untitled source")}</strong><br />
                  <small>{citation.get("source_type", "source")} · {citation.get("chunk_id", "unknown")}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Agent Trace")
        for hop in result["hops"]:
            metadata = hop.get("metadata", {})
            st.markdown(
                f"""
                <div class="trace">
                  <strong>{hop.get("agent_name", "agent").title()}</strong><br />
                  <small>{hop.get("latency_ms", 0)} ms · {summarize_hop(hop.get("agent_name", ""), metadata)}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="answer-box">
              Ready for the demo. Pick a sample prompt or ask your own enterprise knowledge question.
              The app will return a grounded answer, citations, metrics, and agent trace.
            </div>
            """,
            unsafe_allow_html=True,
        )

