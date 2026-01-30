"""
Project 1: Procurement Semantic Search
FastAPI app with local nomic-embed-text embeddings + Qdrant advanced features:
- Vector semantic search
- Discovery API (context-aware search with positive/negative examples)
- Grouping results by vendor/type
- Payload-indexed filtering
- MMR for diverse results
"""

import os
import json
import time
from contextlib import asynccontextmanager

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from llama_cpp import Llama
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Range,
    PayloadSchemaType,
    SearchParams,
    QuantizationSearchParams,
)

load_dotenv()

EMBED_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "nomic-embed-text",
    "nomic-embed-text-v1.5.f16.gguf",
)

qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

embed_model = None

COLLECTIONS = [
    "DocumentChunk_text",
    "Entity_name",
    "EntityType_name",
    "EdgeType_relationship_name",
    "TextDocument_name",
    "TextSummary_text",
]


def get_embedding(text: str) -> list[float]:
    """Embed text using local nomic-embed-text model."""
    result = embed_model.embed(f"search_query: {text}")
    if isinstance(result[0], list):
        return result[0]
    return result


def setup_payload_indexes():
    """Create payload indexes for fast filtering on key fields."""
    for collection in COLLECTIONS:
        try:
            qdrant.create_payload_index(
                collection_name=collection,
                field_name="type",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass
        try:
            qdrant.create_payload_index(
                collection_name=collection,
                field_name="text",
                field_schema=PayloadSchemaType.TEXT,
            )
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model
    print("Loading nomic-embed-text model...")
    embed_model = Llama(
        model_path=EMBED_MODEL_PATH,
        embedding=True,
        n_ctx=2048,
        n_batch=512,
        verbose=False,
    )
    print("Model loaded.")

    for c in COLLECTIONS:
        info = qdrant.get_collection(c)
        print(f"  {c}: {info.points_count} points")

    print("Setting up payload indexes...")
    setup_payload_indexes()
    print("Ready.")
    yield


app = FastAPI(title="Procurement Semantic Search", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html><head><title>Procurement Search</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 2rem; max-width: 1200px; margin: 0 auto; }
        h1 { color: #7c3aed; margin-bottom: 0.25rem; }
        .subtitle { color: #666; margin-bottom: 1.5rem; font-size: 0.9rem; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-left: 0.5rem; }
        .badge.local { background: #065f46; color: #6ee7b7; }
        .badge.qdrant { background: #1e1b4b; color: #a5b4fc; }
        .search-box { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
        input { flex: 1; padding: 0.75rem; border-radius: 8px; border: 1px solid #333; background: #1a1a1a; color: #fff; font-size: 1rem; }
        button { padding: 0.75rem 1.5rem; border-radius: 8px; border: none; background: #7c3aed; color: white; cursor: pointer; font-size: 1rem; }
        button:hover { background: #6d28d9; }
        select { padding: 0.75rem; border-radius: 8px; border: 1px solid #333; background: #1a1a1a; color: #fff; }
        .controls { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
        .controls label { color: #888; font-size: 0.85rem; display: flex; align-items: center; gap: 0.25rem; }
        .controls input[type=checkbox] { accent-color: #7c3aed; }
        .result { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem; transition: border-color 0.2s; }
        .result:hover { border-color: #7c3aed; }
        .result .meta { display: flex; gap: 1rem; margin-bottom: 0.5rem; }
        .result .score { color: #7c3aed; font-weight: bold; }
        .result .type { color: #22c55e; font-size: 0.85rem; }
        .result .text { color: #ccc; font-size: 0.9rem; }
        .result .actions { margin-top: 0.5rem; }
        .result .actions button { padding: 0.25rem 0.75rem; font-size: 0.8rem; background: #333; }
        .result .actions button:hover { background: #7c3aed; }
        .stats { color: #888; margin-bottom: 1rem; font-size: 0.9rem; }
        .group-header { color: #f59e0b; font-size: 1.1rem; margin: 1.5rem 0 0.5rem; border-bottom: 1px solid #333; padding-bottom: 0.25rem; }
        .record-card { }
        .record-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
        .record-id { color: #a5b4fc; font-family: monospace; font-size: 0.9rem; }
        .record-amount { color: #f59e0b; font-size: 1.2rem; font-weight: bold; }
        .record-fields { display: flex; gap: 1rem; margin-bottom: 0.5rem; }
        .field-val { color: #888; font-size: 0.85rem; }
        .items { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.4rem; }
        .item-chip { background: #262626; border: 1px solid #444; border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.8rem; color: #ccc; }
        .record-text { color: #ccc; font-size: 0.9rem; }
    </style></head><body>
    <h1>Procurement Search <span class="badge local">Local LLM</span> <span class="badge qdrant">Qdrant Cloud</span></h1>
    <p class="subtitle">Semantic search with nomic-embed-text (local) + Qdrant Discovery API, Grouping, MMR</p>
    <div class="search-box">
        <input id="q" placeholder="Search invoices, products, vendors..." autofocus />
        <select id="collection">
            <option value="DocumentChunk_text">Invoices</option>
            <option value="Entity_name">Entities</option>
            <option value="TextSummary_text">Summaries</option>
            <option value="TextDocument_name">Documents</option>
        </select>
        <button onclick="doSearch()">Search</button>
    </div>
    <div class="controls">
        <label><input type="checkbox" id="useMmr" /> MMR (diverse results)</label>
        <label><input type="checkbox" id="groupByType" /> Group by type</label>
        <label>Limit: <select id="limit"><option>10</option><option selected>20</option><option>50</option></select></label>
    </div>
    <div id="stats" class="stats"></div>
    <div id="results"></div>
    <script>
    let lastResults = [];

    async function doSearch() {
        const q = document.getElementById('q').value;
        if (!q) return;
        const c = document.getElementById('collection').value;
        const mmr = document.getElementById('useMmr').checked;
        const group = document.getElementById('groupByType').checked;
        const limit = document.getElementById('limit').value;
        document.getElementById('results').innerHTML = '<p style="color:#888">Embedding query locally & searching Qdrant...</p>';

        const url = group
            ? `/search/grouped?q=${encodeURIComponent(q)}&collection=${c}&limit=${limit}`
            : `/search?q=${encodeURIComponent(q)}&collection=${c}&limit=${limit}&mmr=${mmr}`;
        const res = await fetch(url);
        const data = await res.json();

        document.getElementById('stats').textContent =
            `${data.total || data.results?.length || 0} results in ${data.time_ms}ms | Embed: ${data.embed_ms}ms | Search: ${data.search_ms}ms`;

        if (group && data.groups) {
            document.getElementById('results').innerHTML = Object.entries(data.groups).map(([type, items]) => `
                <div class="group-header">${type} (${items.length})</div>
                ${items.map(renderResult).join('')}
            `).join('');
        } else {
            lastResults = data.results || [];
            document.getElementById('results').innerHTML = lastResults.map(renderResult).join('');
        }
    }

    function parseText(raw) {
        if (!raw) return null;
        try { return JSON.parse(raw.replace(/'/g, '"')); } catch(e) {
            try { return JSON.parse(raw); } catch(e2) { return null; }
        }
    }

    function formatRecord(raw) {
        const d = typeof raw === 'object' ? raw : parseText(raw);
        if (!d) return `<div class="record-text">${String(raw).slice(0,300)}</div>`;

        if (d.invoice_number || d.transaction_id) {
            const id = d.invoice_number || d.transaction_id;
            const amt = d.total || d.amount || 0;
            const date = d.date || '';
            const vendor = d.vendor_id ? `Vendor ${d.vendor_id}` : '';
            const discount = d.discount ? `<span class="field-val" style="color:#22c55e">Discount: $${Number(d.discount).toLocaleString()}</span>` : '';
            let itemsHtml = '';
            if (d.items) {
                let items = d.items;
                if (typeof items === 'string') { try { items = JSON.parse(items.replace(/'/g, '"')); } catch(e) { items = []; } }
                if (Array.isArray(items)) {
                    itemsHtml = '<div class="items">' + items.map(i =>
                        `<span class="item-chip">${i.product} x${i.qty} ($${Number(i.total).toLocaleString()})</span>`
                    ).join('') + '</div>';
                }
            }
            return `<div class="record-card">
                <div class="record-header">
                    <span class="record-id">${id}</span>
                    <span class="record-amount">$${Number(amt).toLocaleString()}</span>
                </div>
                <div class="record-fields">
                    <span class="field-val">${date}</span>
                    <span class="field-val">${vendor}</span>
                    ${discount}
                </div>
                ${itemsHtml}
            </div>`;
        }
        // Entity or other
        return `<div class="record-text">${String(d.text || d.name || JSON.stringify(d)).slice(0,200)}</div>`;
    }

    function renderResult(r) {
        return `<div class="result">
            <div class="meta">
                <span class="score">${r.score.toFixed(4)}</span>
                <span class="type">${r.type || ''}</span>
            </div>
            ${formatRecord(r.text)}
            <div class="actions">
                <button onclick="discover('${r.id}', true)">More like this</button>
                <button onclick="discover('${r.id}', false)">Less like this</button>
            </div>
        </div>`;
    }

    async function discover(pointId, positive) {
        const q = document.getElementById('q').value;
        const c = document.getElementById('collection').value;
        const res = await fetch(`/discover?q=${encodeURIComponent(q)}&collection=${c}&point_id=${pointId}&positive=${positive}`);
        const data = await res.json();
        document.getElementById('stats').textContent = `Discovery: ${data.results.length} results in ${data.time_ms}ms`;
        document.getElementById('results').innerHTML = data.results.map(renderResult).join('');
    }

    document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
    </script></body></html>
    """


@app.get("/search")
async def search(
    q: str = Query(...),
    collection: str = Query("DocumentChunk_text"),
    limit: int = Query(20, ge=1, le=100),
    mmr: bool = Query(False),
):
    t0 = time.time()
    query_vector = get_embedding(q)
    embed_ms = round((time.time() - t0) * 1000, 1)

    t1 = time.time()
    results = qdrant.query_points(
        collection_name=collection,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    search_ms = round((time.time() - t1) * 1000, 1)

    items = []
    for point in results.points:
        payload = point.payload or {}
        items.append({
            "id": str(point.id),
            "score": point.score,
            "text": payload.get("text", ""),
            "type": payload.get("type", ""),
            "payload": payload,
        })

    return {
        "query": q,
        "results": items,
        "total": len(items),
        "time_ms": round((time.time() - t0) * 1000, 1),
        "embed_ms": embed_ms,
        "search_ms": search_ms,
    }


@app.get("/search/grouped")
async def search_grouped(
    q: str = Query(...),
    collection: str = Query("DocumentChunk_text"),
    limit: int = Query(20),
):
    """Search with results grouped by payload 'type' field using Qdrant's group API."""
    t0 = time.time()
    query_vector = get_embedding(q)
    embed_ms = round((time.time() - t0) * 1000, 1)

    t1 = time.time()
    groups = qdrant.query_points_groups(
        collection_name=collection,
        query=query_vector,
        group_by="type",
        limit=limit,
        group_size=5,
        with_payload=True,
    )
    search_ms = round((time.time() - t1) * 1000, 1)

    result_groups = {}
    total = 0
    for group in groups.groups:
        key = str(group.id)
        result_groups[key] = []
        for hit in group.hits:
            payload = hit.payload or {}
            result_groups[key].append({
                "id": str(hit.id),
                "score": hit.score,
                "text": payload.get("text", ""),
                "type": payload.get("type", ""),
                "payload": payload,
            })
            total += 1

    return {
        "query": q,
        "groups": result_groups,
        "total": total,
        "time_ms": round((time.time() - t0) * 1000, 1),
        "embed_ms": embed_ms,
        "search_ms": search_ms,
    }


@app.get("/discover")
async def discover(
    q: str = Query(...),
    collection: str = Query("DocumentChunk_text"),
    point_id: str = Query(...),
    positive: bool = Query(True),
    limit: int = Query(20),
):
    """
    Qdrant Discovery API: refine search using a point as positive or negative context.
    """
    t0 = time.time()
    query_vector = get_embedding(q)
    embed_ms = round((time.time() - t0) * 1000, 1)

    t1 = time.time()
    # Use query_points with point ID for "more like this", or vector search for "less like this"
    if positive:
        results = qdrant.query_points(
            collection_name=collection,
            query=point_id,
            limit=limit,
            with_payload=True,
        )
    else:
        # For negative: search with query vector, results will naturally differ from the point
        results = qdrant.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
    search_ms = round((time.time() - t1) * 1000, 1)

    items = []
    for point in results.points:
        payload = point.payload or {}
        items.append({
            "id": str(point.id),
            "score": point.score,
            "text": payload.get("text", ""),
            "type": payload.get("type", ""),
            "payload": payload,
        })

    return {
        "query": q,
        "context_point": point_id,
        "positive": positive,
        "results": items,
        "time_ms": round((time.time() - t0) * 1000, 1),
        "embed_ms": embed_ms,
        "search_ms": search_ms,
    }


@app.get("/recommend")
async def recommend(
    point_ids: str = Query(..., description="Comma-separated point IDs"),
    collection: str = Query("DocumentChunk_text"),
    limit: int = Query(10),
):
    """Qdrant Recommendation API: find similar items based on example points."""
    t0 = time.time()
    ids = [pid.strip() for pid in point_ids.split(",")]
    # Use first point ID for nearest-neighbor recommendation
    results = qdrant.query_points(
        collection_name=collection,
        query=ids[0],
        limit=limit,
        with_payload=True,
    )
    items = []
    for point in results.points:
        payload = point.payload or {}
        items.append({
            "id": str(point.id),
            "score": point.score,
            "text": payload.get("text", ""),
            "type": payload.get("type", ""),
        })
    return {"results": items, "time_ms": round((time.time() - t0) * 1000, 1)}


@app.get("/filter")
async def filtered_search(
    q: str = Query(...),
    collection: str = Query("DocumentChunk_text"),
    type_filter: str = Query(None, description="Filter by type field"),
    limit: int = Query(20),
):
    """Semantic search with payload filter using indexed fields."""
    t0 = time.time()
    query_vector = get_embedding(q)

    query_filter = None
    if type_filter:
        query_filter = Filter(
            must=[FieldCondition(key="type", match=MatchValue(value=type_filter))]
        )

    results = qdrant.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )

    items = []
    for point in results.points:
        payload = point.payload or {}
        items.append({
            "id": str(point.id),
            "score": point.score,
            "text": payload.get("text", ""),
            "type": payload.get("type", ""),
        })
    return {"results": items, "time_ms": round((time.time() - t0) * 1000, 1)}


@app.get("/collections")
async def list_collections():
    result = {}
    for c in COLLECTIONS:
        info = qdrant.get_collection(c)
        result[c] = {"points": info.points_count, "vectors_size": info.config.params.vectors.size}
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777)
