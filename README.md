# EKGA — Enterprise Knowledge Graph Agent 🧠
> Multi-Agent RAG System with Graph Intelligence, LLM-Native Observability & MCP Server — Built for Google Cloud GenAI FDE

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-4B0082)](https://langchain-ai.github.io/langgraph/)
[![Vertex AI](https://img.shields.io/badge/Vertex_AI-Gemini_1.5_Pro-4285F4?logo=googlecloud)](https://cloud.google.com/vertex-ai)
[![Neo4j](https://img.shields.io/badge/Neo4j-Aura-008CC1?logo=neo4j)](https://neo4j.com/cloud/platform/aura-graph-database/)
[![AlloyDB](https://img.shields.io/badge/AlloyDB-pgvector-4285F4?logo=googlecloud)](https://cloud.google.com/alloydb)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-35%20passing-brightgreen?logo=pytest)](https://pytest.org)

---

## The Problem This Solves

Enterprise customers arrive at Google Cloud with the same structural failure: **knowledge trapped in silos.**

A Confluence wiki no one searches. Slack threads holding architectural decisions that expired with the employee who wrote them. Jira tickets that are the only record of why a system behaves the way it does. PDFs that exist but aren't findable. The result: engineers re-solve solved problems, onboarding takes 90 days, and institutional memory walks out the door every time someone quits.

Standard RAG retrieves text chunks. **EKGA retrieves knowledge** — relationships between documents, authors, decisions, and systems — and answers in production with full observability baked in.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION LAYER                             │
│  Confluence · Slack Exports · PDFs · Jira JSON                  │
│  Pub/Sub → Cloud Run Jobs → Chunker → Vertex AI Embeddings      │
└──────────────────┬──────────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   DUAL-STORE LAYER  │
        │  AlloyDB (pgvector) │  ← dense vector search
        │  Neo4j Aura         │  ← graph: doc→author→topic→ticket
        └──────────┬──────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────┐
│               MULTI-AGENT ORCHESTRATOR (LangGraph)              │
│                                                                  │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │   Router   │ → │  Retrieval   │ → │   Synthesis Agent    │  │
│  │   Agent    │   │   Agent      │   │  (Gemini 1.5 Pro)    │  │
│  │ (classify) │   │ hybrid BM25+ │   │  cited, grounded     │  │
│  │ factual /  │   │ vector +     │   │  answer generation   │  │
│  │ relational │   │ graph path   │   └──────────┬───────────┘  │
│  │ procedural │   └──────────────┘              │              │
│  └────────────┘                                 ▼              │
│                                      ┌──────────────────────┐  │
│                                      │    Eval Agent        │  │
│                                      │  faithfulness score  │  │
│                                      │  groundedness check  │  │
│                                      │  per-hop tracing     │  │
│                                      └──────────────────────┘  │
└──────────────────┬──────────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   OBSERVABILITY     │
        │  BigQuery trace log │
        │  token/cost/latency │
        │  per agent hop      │
        │  Looker Studio dash │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │    MCP SERVER       │
        │  Wraps agent graph  │
        │  as enterprise API  │
        │  Slack · SFDC · etc │
        └─────────────────────┘
```

---

## Key Components

### 1. Ingestion Pipeline
- **Sources**: Confluence REST API, Slack export JSON, PDF batch uploads, Jira REST API
- **Transport**: GCP Pub/Sub → Cloud Run Jobs
- **Processing**: Semantic chunking (512 tokens, 10% overlap), entity extraction (spaCy), deduplication (MinHash LSH)
- **Embedding**: Vertex AI `text-embedding-004` (768-dim)
- **Storage**: AlloyDB (pgvector index, HNSW) + Neo4j Aura (entity relationship graph)

### 2. Multi-Agent Orchestrator (LangGraph ReAct)
| Agent | Role | Pattern |
|---|---|---|
| **Router Agent** | Classifies query: factual / relational / procedural | Zero-shot classification |
| **Retrieval Agent** | Hybrid BM25 + pgvector + Cypher graph traversal | Self-reflection on low-confidence results |
| **Synthesis Agent** | Gemini 1.5 Pro with citation grounding | RAG with source attribution |
| **Eval Agent** | Scores answer faithfulness + groundedness per response | RAGAS metrics inline |

### 3. LLM-Native Observability
Every agent hop logs to BigQuery:
```
agent_name | tokens_in | tokens_out | latency_ms | cost_usd | faithfulness_score | timestamp
```
Looker Studio dashboard surfaces: cost-per-request trend, accuracy drift, slowest graph queries, cheapest retrieval paths.

### 4. MCP Server
Wraps the full agent graph as a production MCP endpoint. Customer tools (Slack bots, Salesforce plugins, internal portals) call one endpoint — no awareness of the underlying graph or vector infrastructure required.

---

## Tech Stack

| Layer | Technology | GCP Equivalent |
|---|---|---|
| Orchestration | LangGraph (ReAct, self-reflection) | Cloud Run |
| LLM | Gemini 1.5 Pro via Vertex AI | Vertex AI |
| Vector store | AlloyDB + pgvector (HNSW) | AlloyDB |
| Graph store | Neo4j Aura | — |
| Embeddings | Vertex AI text-embedding-004 | Vertex AI |
| Ingestion transport | GCP Pub/Sub | Pub/Sub |
| Trace log | BigQuery | BigQuery |
| Observability UI | Looker Studio | Looker Studio |
| MLflow eval tracking | MLflow (self-hosted) | Vertex AI Experiments |
| Enterprise API | MCP Server (FastAPI) | Cloud Run |
| Containerization | Docker Compose | GKE / Cloud Run |

---

## Project Structure

```
knowledge-graph-agent/
├── src/
│   ├── ingestion/
│   │   ├── connectors/
│   │   │   ├── confluence_connector.py
│   │   │   ├── slack_connector.py
│   │   │   ├── pdf_connector.py
│   │   │   └── jira_connector.py
│   │   ├── chunker.py
│   │   ├── entity_extractor.py
│   │   ├── deduplicator.py
│   │   └── pubsub_publisher.py
│   ├── storage/
│   │   ├── alloydb_loader.py        # pgvector HNSW index
│   │   ├── neo4j_loader.py          # Cypher schema + edge creation
│   │   └── schema.sql
│   ├── retrieval/
│   │   ├── hybrid_retriever.py      # BM25 + dense + graph
│   │   ├── graph_query.py           # Cypher traversal patterns
│   │   └── query_rewriter.py        # self-reflection on low confidence
│   ├── agents/
│   │   ├── router_agent.py
│   │   ├── retrieval_agent.py
│   │   ├── synthesis_agent.py
│   │   ├── eval_agent.py
│   │   └── orchestrator.py          # LangGraph state machine
│   ├── observability/
│   │   ├── trace_logger.py          # BigQuery hop-level logging
│   │   ├── cost_tracker.py          # tokens/sec, cost-per-request
│   │   └── eval_pipeline.py         # RAGAS + MLflow
│   └── mcp_server/
│       ├── server.py                # FastAPI MCP endpoint
│       └── schemas.py
├── tests/
│   ├── test_connectors.py
│   ├── test_retrieval.py
│   ├── test_agents.py
│   ├── test_eval_pipeline.py
│   └── test_mcp_server.py
├── notebooks/
│   └── 01_graph_schema_eda.ipynb
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and configure
git clone https://github.com/sajanshergill/ekga.git
cd ekga
cp .env.example .env
# Add: GOOGLE_CLOUD_PROJECT, ALLOYDB_URI, NEO4J_URI, NEO4J_PASSWORD

# 2. Start infrastructure
docker compose up -d

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run ingestion (synthetic data by default)
python src/ingestion/pubsub_publisher.py --source synthetic --docs 500

# 5. Launch MCP server
uvicorn src.mcp_server.server:app --reload --port 8080

# 6. Run test suite
pytest tests/ -v

# 7. Query the agent
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What architectural decisions were made about the auth service in Q3?"}'
```

---

## Sample Queries

```
"Why was the payments service migrated off PostgreSQL?"
→ Retrieval Agent: graph path doc[ADR-042] → decision[DB migration] → author[alice@]
→ Synthesis: "Per ADR-042 (authored by Alice Chen, approved Sept 2023), PostgreSQL was
   replaced due to connection pool exhaustion at 10K concurrent users..."

"What does onboarding for a backend engineer look like?"
→ Router: procedural
→ Retrieval: Confluence + Jira ticket chain
→ Synthesis: step-by-step with source citations

"Which team owns the rate-limiting module?"
→ Router: relational
→ Graph: module → codeowner → team
→ Synthesis: "The rate-limiting module (src/middleware/throttle.py) is owned by the
   Platform Infrastructure team per CODEOWNERS, last updated by @dev-infra in PR #2847"
```

---

## Observability Dashboard (Looker Studio)

| Metric | Description |
|---|---|
| **Cost per request** | USD by query type (factual < relational < procedural) |
| **Token throughput** | tokens/sec per agent, trending by day |
| **Faithfulness score** | RAGAS faithfulness drift over rolling 7d window |
| **Retrieval latency** | p50/p95 per retrieval path (BM25 vs vector vs graph) |
| **Cache hit rate** | Repeated queries served from AlloyDB materialized views |
| **Agent hop count** | Distribution of hops per query (proxy for complexity cost) |

---

## Eval Pipeline

```bash
# Run full RAGAS eval suite against ground-truth Q&A set
python src/observability/eval_pipeline.py --dataset data/eval/ground_truth.json

# Output: faithfulness, answer_relevancy, context_recall per query type
# Logged to: MLflow experiment "ekga-evals"
# CI gate: faithfulness < 0.80 fails the build
```

---

## Business Context: The FDE Pitch

**Customer scenario**: B2B SaaS company, 600 engineers, 4-year Confluence history, 3 Slack workspaces, post-acquisition knowledge fragmentation. Engineering org spends est. 15% of sprint capacity re-researching solved problems.

**FDE approach**:
1. 2-hour technical discovery — map knowledge sources, identify access patterns, surface integration blockers (SSO, DLP, export restrictions)
2. 3-day MVP sprint — ingest 3 months of Confluence + Slack into this pipeline on GCP sandbox
3. Live demo with customer's own questions against their own data
4. Outcome: Demonstrate measurable retrieval accuracy lift vs. Confluence native search; convert to production deployment with AlloyDB + Vertex AI on customer's GCP org

**What this proves to Google Cloud leadership**:
- Can move from prototype → production-grade agentic system independently
- Understands the "connective tissue" problem (legacy APIs, security perimeters, data silos)
- Builds observability-first — every hop is traceable, every dollar is accounted for
- Field insights (graph retrieval outperforms vector-only on relational queries) feed directly into Vertex AI product roadmap

---

## Author

**Sajan Shergill**
MS Data Science, Pace University (May 2026)
[LinkedIn](https://linkedin.com/in/sajanshergill) · [Portfolio](https://sajansshergill.github.io)

> *Built as a portfolio project targeting the Google Cloud GenAI Forward Deployed Engineer role,
> demonstrating production-grade multi-agent system design, GCP-native architecture,
> LLM-native observability, and the "builder's mindset" the FDE role demands.*
