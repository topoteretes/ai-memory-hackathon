# cognee-qdrant-starter

Starter templates for building with [Cognee](https://cognee.ai) + [Qdrant Cloud](https://qdrant.tech). Three ready-to-run FastAPI projects that demonstrate semantic search, analytics, and anomaly detection on procurement data — all using **local embeddings** (nomic-embed-text) and **Qdrant Cloud** as the vector backend.

Built for the [Cognee x Qdrant Hackathon](https://luma.com/50si7fw4) (Feb 7, 2026).

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- A [Qdrant Cloud](https://cloud.qdrant.io) cluster
- The cognee minihack data (provided at the hackathon)
- `nomic-embed-text-v1.5.f16.gguf` model file (provided at the hackathon)

## Setup

### 1. Migrate data to Qdrant Cloud

If your data is in Cognee's LanceDB format, use the migration tool:

```bash
cd lance-to-qdrant
cp .env.example .env
# Edit .env with your Qdrant Cloud URL and API key

uv sync
uv run python migrate.py
```

This reads 6 LanceDB tables and uploads them as Qdrant collections:

| Collection | Records | Description |
|---|---|---|
| DocumentChunk_text | 2,000 | Invoice and transaction chunks with embeddings |
| Entity_name | 8,816 | Extracted entities (products, vendors, SKUs) |
| EntityType_name | 8 | Entity type definitions |
| EdgeType_relationship_name | 13 | Relationship type definitions |
| TextDocument_name | 2,000 | Document references |
| TextSummary_text | 2,000 | Document summaries |

All vectors are **768-dimensional** (nomic-embed-text).

### 2. Place the embedding model

Put `nomic-embed-text-v1.5.f16.gguf` in `models/nomic-embed-text/`:

```
models/
  nomic-embed-text/
    nomic-embed-text-v1.5.f16.gguf
```

### 3. Run a project

Each project is self-contained with its own `pyproject.toml`:

```bash
cd project1-procurement-search  # or project2 or project3
cp .env.example .env
# Edit .env with your Qdrant Cloud URL and API key

uv sync
uv run python app.py
```

## Projects

### Project 1: Procurement Semantic Search (port 7777)

Semantic search across all procurement data with interactive UI.

**Qdrant features used:**
- **Query API** — vector similarity search with local embeddings
- **Group API** (`query_points_groups`) — group results by payload field (type, vendor)
- **Point-ID queries** — "More like this" / "Less like this" discovery
- **Payload indexing** (`create_payload_index`) — keyword + full-text indexes for filtered search
- **Filtered search** — combine vector similarity with metadata filters

**Endpoints:**
- `GET /` — Interactive search UI
- `GET /search?q=...` — Semantic search
- `GET /search/grouped?q=...` — Grouped by type
- `GET /discover?q=...&point_id=...&positive=true` — Discovery/recommendation
- `GET /recommend?point_ids=...` — Find similar records
- `GET /filter?q=...&type_filter=...` — Filtered semantic search
- `GET /collections` — Collection stats

### Project 2: Spend Analytics Dashboard (port 5553)

Interactive analytics dashboard with charts and semantic search.

**Qdrant features used:**
- **Scroll API** — bulk data extraction for aggregation
- **Query API** — semantic search over invoices
- **Group API** — vendor-grouped search results
- **Payload indexing** — fast vendor/type filtering

**What it shows:**
- Total spend, invoice/transaction counts, vendor breakdown
- Monthly spend trends (line chart)
- Vendor spend distribution (doughnut chart)
- Top products by quantity and revenue (bar charts)
- Semantic search overlay

**Endpoints:**
- `GET /` — Dashboard UI
- `GET /api/analytics` — Aggregated analytics JSON
- `GET /api/search?q=...` — Semantic search
- `GET /api/search/grouped?q=...` — Grouped results

### Project 3: Anomaly Detective (port 6971)

Automated anomaly detection using vector analysis and Qdrant's batch query API.

**Qdrant features used:**
- **Batch Query API** (`query_batch_points`) — 50 recommend queries per request for near-duplicate detection
- **Point-ID queries** — find records similar to flagged anomalies
- **Scroll API** with vectors — bulk vector retrieval for centroid analysis
- **Payload indexing** — fast anomaly filtering

**Detection methods:**
- **Amount outliers** — z-score analysis on invoice totals (flags z > 2.5)
- **Embedding outliers** — vectors far from collection centroid (flags z > 2.0)
- **Near-duplicates** — similarity > 0.99 via batch recommend queries
- **Vendor variance** — coefficient of variation > 0.8 across vendor invoices

**Endpoints:**
- `GET /` — Anomaly dashboard UI
- `GET /api/anomalies` — All detected anomalies (filterable by severity/type)
- `GET /api/search?q=...` — Semantic search over procurement data
- `GET /api/investigate/{point_id}` — Find records similar to an anomaly

## Qdrant Features Summary

| Feature | P1 | P2 | P3 |
|---|:---:|:---:|:---:|
| Query API (vector search) | ✓ | ✓ | ✓ |
| Batch Query API | | | ✓ |
| Point-ID Recommend | ✓ | | ✓ |
| Group API | ✓ | ✓ | |
| Scroll API | | ✓ | ✓ |
| Payload Indexing | ✓ | ✓ | ✓ |
| Filtered Search | ✓ | ✓ | |

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  nomic-embed-text    │  ← Local GGUF model (768-dim)
│  (llama-cpp-python)  │
└────────┬────────────┘
         │ query vector
         ▼
┌─────────────────────┐
│   Qdrant Cloud       │  ← 14,837 vectors across 6 collections
│   (managed cluster)  │
└────────┬────────────┘
         │ results
         ▼
┌─────────────────────┐
│   FastAPI App        │  ← Formatted cards, charts, anomaly detection
└─────────────────────┘
```

No external API keys needed for embeddings — everything runs locally via `llama-cpp-python`.

## Data

The procurement dataset contains IT hardware purchase records:
- **Invoices** with line items (laptops, monitors, keyboards, SSDs, RAM, etc.)
- **Transactions** with vendor/amount/discount data
- **Entities** extracted by Cognee (product names, SKUs, vendor IDs)
- **10 vendors**, ~2000 documents, spanning 2025

## License

MIT
