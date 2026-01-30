# cognee-qdrant-starter

Hackathon starter repo for "AI-Memory Hackathon by cognee" (Feb 7, 2026).
GitHub: https://github.com/thierrypdamiba/cognee-qdrant-starter

## Architecture
- 6 Qdrant Cloud collections, 14,837 vectors, 768-dim, cosine distance
- Local embeddings: nomic-embed-text-v1.5.f16.gguf via llama-cpp-python
- Qdrant Cloud: https://41933e38-7320-4675-b795-ffa6a7ed86d3.us-west-2-0.aws.cloud.qdrant.io
- Credentials in each project's .env (and ~/twelve/avenue/.env has originals)

## Projects
- P1: Procurement Search (port 7777) — Prefetch+RRF Fusion, Discovery API, Recommend API, Group API, Filtered Search
- P2: Spend Analytics (port 5553) — Prefetch+RRF Fusion, Scroll API, Group API, Chart.js dashboard
- P3: Anomaly Detective (port 6971) — Batch Query API, Recommend API (BEST_SCORE), Scroll with vectors, centroid outlier detection

## Qdrant Advanced Features Used
- Prefetch + Fusion (RRF) — multi-stage retrieval pipeline (all projects)
- Discovery API (DiscoverQuery + ContextPair) — P1
- Recommend API (RecommendQuery + RecommendInput + AVERAGE_VECTOR/BEST_SCORE) — P1, P3
- Batch Query API (query_batch_points) — P3 duplicate detection
- Group API (query_points_groups) — P1, P2
- Scroll API — P2, P3
- Payload Indexing (keyword + full-text) — all projects

## Collections
| Collection | Records | Content |
|---|---|---|
| DocumentChunk_text | 2,000 | Invoice and transaction chunks |
| Entity_name | 8,816 | Products, vendors, SKUs |
| EntityType_name | 8 | Entity type definitions |
| EdgeType_relationship_name | 13 | Relationship types |
| TextDocument_name | 2,000 | Document references |
| TextSummary_text | 2,000 | Document summaries |

## Qdrant Cloud Inference (potential upgrade)
- Could replace local GGUF with server-side embeddings via `cloud_inference=True`
- Would need to re-embed all data with the new model since 768-dim nomic-embed-text vectors are model-specific
- Free tier available with `Document(text="...", model="Qdrant/Qwen/Qwen3-Embedding-0.6B")`
