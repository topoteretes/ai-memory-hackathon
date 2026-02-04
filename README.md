![AI-Memory Hackathon by cognee](hackathon-banner.avif)

# cognee-qdrant-starter

Starter templates for the **AI-Memory Hackathon by cognee**. Three ready-to-run FastAPI projects: semantic search, spend analytics, and anomaly detection on procurement data.

**Stack:** [cognee](https://github.com/topoteretes/cognee) (knowledge graph memory) + [Qdrant Cloud](https://cloud.qdrant.io) (vector search) + [Distil Labs](https://www.distillabs.ai/) (LLM reasoning) + [DigitalOcean](https://www.digitalocean.com/) (deployment)

## How it works

```
Raw documents
    |
    v
cognee.add() + cognee.cognify()     <-- cognee extracts entities, relationships, summaries
    |
    v
Qdrant Cloud (6 collections)        <-- vectors + knowledge graph stored here
    |
    v
FastAPI apps                         <-- search, analytics, anomaly detection
    |
    v
Distil Labs SLM                     <-- LLM reasoning (local GGUF or hosted API)
    |
    v
DigitalOcean App Platform           <-- deployed and shareable
```

## Quick Start

### 1. Set up Qdrant Cloud

Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io), then download and restore the pre-built snapshots.

> **Already have the files locally?** If you have `snapshots/` and `models/` directories, skip the download steps and go straight to restoring/running.

**Direct download:** [cognee-vectors-snapshot.tar.gz](https://cognee-data.nyc3.digitaloceanspaces.com/cognee-vectors-snapshot.tar.gz) (91 MB)

```bash
# Download snapshots (from DO Spaces)
uv run python download-from-spaces.py

# Add your credentials
cp .env.example .env
# Edit .env with your Qdrant Cloud URL and API key

# Restore all 6 collections
uv run python restore-snapshots.py
```

This uploads 6 pre-built collections (14,837 vectors, 768-dim) to your cluster:

| Collection | Records | Content |
|---|---|---|
| DocumentChunk_text | 2,000 | Invoice and transaction chunks |
| Entity_name | 8,816 | Products, vendors, SKUs |
| EntityType_name | 8 | Entity type definitions |
| EdgeType_relationship_name | 13 | Relationship types |
| TextDocument_name | 2,000 | Document references |
| TextSummary_text | 2,000 | Document summaries |

### 2. Download models

```bash
# Included in download-from-spaces.py, or manually:
curl -O https://cognee-qdrant-starter.s3.amazonaws.com/models.zip
unzip models.zip -d models/
```

Models:
- **nomic-embed-text** (768-dim embeddings, local inference)
- **Distil Labs SLM** (fine-tuned reasoning model, GGUF quantized)
- **Qwen3-4B** (fallback LLM, optional)

### 3. Run a project

Each project is self-contained:

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

**Qdrant features:** Query API, Prefetch + RRF Fusion, Group API, Discovery API, Recommend API, payload indexing, filtered search

**Endpoints:** `/search`, `/search/grouped`, `/discover`, `/recommend`, `/filter`, `/ask` (RAG Q&A), `/cognee-search`, `/add-knowledge`, `/collections`

### Project 2: Spend Analytics Dashboard (port 5553)

Interactive analytics dashboard with Chart.js visualizations and semantic search.

**Qdrant features:** Scroll API (bulk extraction), Query API, Group API, payload indexing

**Endpoints:** `/api/analytics`, `/api/search`, `/api/search/grouped`, `/api/insights` (LLM analysis)

### Project 3: Anomaly Detective (port 6971)

Automated anomaly detection using vector analysis and Qdrant's batch API.

**Qdrant features:** Batch Query API (50 recommend queries/request), Recommend API, Scroll API with vectors, payload indexing

**Detection methods:** amount outliers (z-score), embedding outliers (centroid distance), near-duplicates (similarity > 0.99), vendor variance

**Endpoints:** `/api/anomalies`, `/api/search`, `/api/investigate/{point_id}`, `/api/explain/{point_id}` (LLM explanation)

## cognee Pipeline

The starter data was built using cognee's ECL (Extract, Cognify, Load) pipeline. You can add your own data:

### Quick start

```bash
cd cognee-pipeline
cp .env.example .env
# Edit .env: add Qdrant credentials + LLM provider
uv sync
uv run python ingest.py
```

### Add your own data

```python
import cognee
from cognee.api.v1.search import SearchType

# 1. Add documents (text, files, URLs)
await cognee.add("Your document text here...")
await cognee.add("/path/to/document.pdf")
await cognee.add(["doc1.txt", "doc2.csv", "doc3.pdf"])

# 2. Build knowledge graph (extracts entities, relationships, summaries)
await cognee.cognify()

# 3. Search with graph context
results = await cognee.search(
    query_text="What vendors supply IT equipment?",
    query_type=SearchType.CHUNKS,  # or SUMMARIES, GRAPH_COMPLETION, RAG_COMPLETION
)
```

### Supported input types

- Plain text strings
- PDF, DOCX, TXT, CSV files
- URLs (web pages)
- Directories of files

### Reset and re-ingest

```python
# Clear all data and start fresh
await cognee.prune.prune_data()
await cognee.prune.prune_system(metadata=True)
```

See [cognee docs](https://docs.cognee.ai) for full pipeline options.

## Deployment

Two modes: **local** (dev with GGUF models) and **remote** (deployed with API-based inference).

### Local dev (Distil Labs GGUF)

```bash
# .env: LLM_MODE=local, EMBED_MODE=local (defaults)
uv run python app.py
```

Runs the Distil Labs SLM locally via llama-cpp-python. Requires 4-8GB RAM.

### Deploy to DigitalOcean App Platform

```bash
# 1. Upload data to DO Spaces
uv run python upload-to-spaces.py

# 2. Set remote mode in .env
#    LLM_MODE=remote
#    LLM_API_URL=<distil-labs-hosted-endpoint>
#    EMBED_MODE=remote
#    EMBED_API_URL=<embedding-api-endpoint>

# 3. Deploy
doctl apps create --spec .do/app.yaml
```

Or use Docker locally:

```bash
docker compose up
```

The deployed version calls the Distil Labs hosted API (or any OpenAI-compatible endpoint) instead of loading GGUF files. This keeps the container small and runs on a $6/mo App Platform instance.

**Free credits:** New DigitalOcean accounts get [$200 in free credits](https://www.digitalocean.com/try/free-trial) for 60 days.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | - | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | - | Qdrant Cloud API key |
| `LLM_MODE` | `local` | `local` (GGUF) or `remote` (API) |
| `LLM_API_URL` | - | OpenAI-compatible chat completions endpoint |
| `LLM_API_KEY` | - | API key for remote LLM |
| `LLM_MODEL_NAME` | `distil-labs-slm` | Model name for remote LLM |
| `EMBED_MODE` | `local` | `local` (GGUF) or `remote` (API) |
| `EMBED_API_URL` | - | OpenAI-compatible embeddings endpoint |
| `EMBED_API_KEY` | - | API key for remote embeddings |
| `SPACES_ENDPOINT` | - | DO Spaces endpoint (e.g. `https://nyc3.digitaloceanspaces.com`) |
| `SPACES_BUCKET` | - | DO Spaces bucket name |

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [cognee](https://github.com/topoteretes/cognee) (knowledge graph memory)
- [Qdrant Cloud](https://cloud.qdrant.io) cluster (free tier available)
- [DigitalOcean](https://www.digitalocean.com/) account ($200 free credits available)
