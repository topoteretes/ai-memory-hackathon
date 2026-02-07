# Setting Up Question Answering with Qdrant and a Local Model

## Quick Start

**We will set up**:
- Ollama with two local models (embedding and LLM)
- A Python virtual environment with pinned dependencies
- A Cognee knowledge graph imported from prebuilt data
- A local Qdrant vector store loaded with snapshot data
- The question answering script (`solution_q_and_a.py`)

**This will allow you to**:
- Access ingested data from invoice and transaction documents
- Retrieve structured context from a knowledge graph for LLM queries
- Ask natural-language questions about the data using a local language model
- Build tools, agents, or workflows on top of the Q&A pipeline

**Before installation**:
- copy `models/` from the USB to your working directory (or download via `uv run python download-from-spaces.py`)
- verify the three subdirectories contain Modelfile and a *.gguf each

**Project installation**:
```bash
# Ollama installation
brew install ollama
ollama serve

# Ollama model registration
cd models
ollama create nomic-embed-text -f nomic-embed-text/Modelfile
ollama create cognee-distillabs-model-gguf-quantized -f cognee-distillabs-model-gguf-quantized/Modelfile
cd ..

# Initialize python environment, install dependencies
uv venv
source .venv/bin/activate
uv sync

# Graph setup
python setup.py

# Qdrant (local Docker)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Configure for use locally, retrieve data, restore to database
cp .env.example.local .env
uv run python download-from-spaces.py
uv run python restore-snapshots.py

# Run Q and A example
python solution_q_and_a.py
```

**Pitfalls to avoid**:
- building the venv in `models/` instead of the project root
- having a stale venv activated

**Next steps**:
- look around the code
- play with the queries
- check out the databases
- build something


## Qdrant Cloud (alternative to local Docker)

If you prefer hosted Qdrant over local Docker, set up a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io) and use `.env.example` instead of `.env.example.local`:
```bash
cp .env.example .env
# Edit .env -- fill in QDRANT_URL and QDRANT_API_KEY with your Cloud values
uv run python download-from-spaces.py
uv run python restore-snapshots.py
```

After restore, your cluster contains 14,837 vectors across 6 collections.


## Example results

Example results comparing LLM and SLM outputs can be found in `responses.txt`.


## Example projects using Qdrant

**Project 1: Procurement Semantic Search** (port 7777) -- semantic search across all procurement data with interactive UI. Qdrant features include Query API, Prefetch + RRF Fusion, Group API, Discovery API, Recommend API, payload indexing, and filtered search.

**Project 2: Spend Analytics Dashboard** (port 5553) -- interactive analytics dashboard with Chart.js visualizations and semantic search. Uses Scroll API for bulk extraction, Query API, Group API, and payload indexing.

**Project 3: Anomaly Detective** (port 6971) -- automated anomaly detection using vector analysis and Qdrant's Batch Query API. Detection methods include amount outliers (z-score), embedding outliers (centroid distance), near-duplicates (similarity > 0.99), and vendor variance.

Each project is self-contained and can be run with:
```bash
cd project1-procurement-search  # or project2 or project3
uv run python app.py
```


## Adding your own data

The starter data was built using cognee's ECL (Extract, Cognify, Load) pipeline:
```bash
cd cognee-pipeline
cp .env.example .env
# Edit .env: add Qdrant credentials + LLM provider
uv sync
uv run python ingest.py
```

Programmatic usage:
```python
import cognee
from cognee.api.v1.search import SearchType

await cognee.add("Your document text here...")
await cognee.cognify()
results = await cognee.search(
    query_text="What vendors supply IT equipment?",
    query_type=SearchType.CHUNKS,
)
```

Supported input types: plain text strings, PDF, DOCX, TXT, CSV files, URLs, and directories of files. See [cognee docs](https://docs.cognee.ai) for full pipeline options.


## Using Qwen3 as an alternative model

Register the Qwen3 model with Ollama:
```bash
cd models
ollama create Qwen3-4B-Q4_K_M -f Qwen3-4B-Q4_K_M/Modelfile
cd ..
```

Access it via the standard OpenAI-compatible interface at `http://localhost:11434/v1` with model name `Qwen3-4B-Q4_K_M`.


## DigitalOcean deployment

Two modes are available: **local** (GGUF models, default) and **remote** (API-based inference).

**Local dev** runs the Distil Labs SLM via llama-cpp-python (requires 4-8GB RAM):
```bash
# .env: LLM_MODE=local, EMBED_MODE=local (defaults)
uv run python app.py
```

**Remote deployment** to DigitalOcean App Platform:
```bash
uv run python upload-to-spaces.py
# Set LLM_MODE=remote and EMBED_MODE=remote in .env
doctl apps create --spec .do/app.yaml
```

Or run remotely via Docker:
```bash
docker compose up
```

**Environment variables**:

| Variable          | Default           | Description                                 |
| ----------------- | ----------------- | ------------------------------------------- |
| `QDRANT_URL`      | -                 | Qdrant Cloud cluster URL                    |
| `QDRANT_API_KEY`  | -                 | Qdrant Cloud API key                        |
| `LLM_MODE`        | `local`           | `local` (GGUF) or `remote` (API)            |
| `LLM_API_URL`     | -                 | OpenAI-compatible chat completions endpoint |
| `LLM_API_KEY`     | -                 | API key for remote LLM                      |
| `LLM_MODEL_NAME`  | `distil-labs-slm` | Model name for remote LLM                   |
| `EMBED_MODE`      | `local`           | `local` (GGUF) or `remote` (API)            |
| `EMBED_API_URL`   | -                 | OpenAI-compatible embeddings endpoint       |
| `EMBED_API_KEY`   | -                 | API key for remote embeddings               |
| `SPACES_ENDPOINT` | -                 | DO Spaces endpoint                          |
| `SPACES_BUCKET`   | -                 | DO Spaces bucket name                       |


## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- Docker (for local Qdrant)
- [cognee](https://github.com/topoteretes/cognee)


## Useful commands

Turn off and remove Qdrant if necessary for recreating:
```bash
docker stop qdrant && docker rm qdrant
docker volume rm qdrant_storage
```