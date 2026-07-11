# Recommendation Engine

A personalised cart recommendation system that uses **order history** and **product visit behaviour** to suggest products via semantic similarity search.

## Architecture

```
Client Request
    │
    ├─ POST /visitProduct ── saves user visit to MySQL
    │
    └─ POST /recommendations
           │
           ├─ extract_purchased_products()  ← from request body
           ├─ extract_visited_products()    ← from MySQL visits table
           │
            ├─ build_user_profile()
            │     ├─ load embeddings from in-memory dict (fastest)
            │     ├─ fallback: load from MySQL product_embeddings
            │     └─ weighted average of purchased + visited embeddings
            │
            ├─ search_faiss()
            │     └─ FAISS Index (FlatIP by default) → top 30 candidates (exclude purchased)
           │
           └─ apply_business_rules()
                 ├─ filter inactive products
                 ├─ max 2 products per category
                 └─ return top 10
```

### Embedding Lookup Chain

| Priority | Source | Speed |
|----------|--------|-------|
| 1st | In-memory `engine._backup_embeddings` dict | ~0.001 ms |
| 2nd | MySQL `product_embeddings` table | ~2 ms |

## Files

### Core

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app with 4 endpoints |
| `recommender.py` | Recommendation pipeline (profile → search → rules) |
| `initialize.py` | FAISS index load/save/backup + FastAPI lifespan |
| `models.py` | Pydantic request/response models + SQLAlchemy ORM models |
| `db.py` | MySQL connection + CRUD helpers |

### Scripts & Tests

| File | Purpose |
|------|---------|
| `seed_from_csv.py` | Bulk-embed products from `products.csv` into DB + FAISS + backup |
| `test_validation.py` | End-to-end test against real MySQL + FAISS (uses `TestClient`) |
| `tests/test.py` | **Self-contained** test — uses SQLite + temp FAISS; no production resources touched |

## How the Recommendation Works

1. **User profile** — embeddings of purchased/visited products are averaged with recency and frequency weights:
   - Purchase weight = `1.0 × quantity × exp(-days / 30)`
   - Visit weight = `0.3 × exp(-days / 7)`

2. **FAISS search** — the profile vector is queried against a FAISS index (default: `IndexFlatL2`; configurable to `IndexIVFPQ` via `RECOMMENDER_INDEX_TYPE=ivfpq`) to find the nearest neighbours.

3. **Business rules** — candidates are filtered:
   - Inactive products removed
   - Max **2 products per category**
   - Final list capped at **10**

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/visitProduct` | POST | Record a product visit (`{"userid": int, "productid": int}`) |
| `/recommendations` | POST | Get recommendations (`{"user_id": int, "orders": [...]}`) |
| `/product-embedding` | POST | Add embeddings for new products (admin) |

## Resource Consumption (per request at 10K req/min)

### `/recommendations` (~100 req/s, ~60% of traffic)

| Function | Time | DB Queries | Lock Held | Memory | Complexity |
|----------|------|-----------|-----------|--------|------------|
| `extract_purchased_products` | ~10-50μs | 0 | No | ~1KB | O(n·m), n=orders, m=items |
| `extract_visited_products` | ~1-3ms | 1 SELECT (index seek) | No | ~5KB | O(50) + DB |
| `build_user_profile` (cache hit) | ~5-50μs | 0 | Yes (~1μs) | ~50KB | O(k·d), k=PIDs, d=384 |
| `build_user_profile` (cache miss) | +~2ms | +1 SELECT (IN clause) | Yes (+~1μs) | +~50KB | Same + DB fetch |
| `search_faiss` (FlatL2 5K) | ~0.5-2ms | 0 | Yes (~0.5-2ms) | ~40KB copy | O(d·n) BLAS matmul |
| `search_faiss` (IVFPQ 5K) | ~0.1ms | 0 | Yes (~0.1ms) | ~40KB copy | O(d·log n) PQ decode |
| `apply_business_rules` | ~1-3ms | 1 SELECT (IN clause) | No | ~5KB | O(k), k=candidates |
| **Total (cache hit)** | **~4-8ms** | **2** | **~1ms** | **~100KB** | |
| **Total (cache miss)** | **~6-10ms** | **3** | **~1ms** | **~150KB** | |

### `/visitProduct` (~50 req/s, ~30% of traffic)

| Function | Time | DB Queries | Memory |
|----------|------|-----------|--------|
| `save_visit` | ~2-5ms | 1 INSERT + 1 COMMIT | ~1KB |

### `/product-embedding` (~0.2 req/s, <0.1% of traffic)

| Function | Time | DB Queries | Memory | Notes |
|----------|------|-----------|--------|-------|
| `model.encode()` (per product) | ~50ms | 0 | ~80MB (model) | SentenceTransformer inference |
| `store_embedding()` (per product) | ~2ms | 1 INSERT | ~2KB | Binary BLOB write |
| `add_to_index()` (per batch 32) | ~10-50ms | 0 | ~500KB | FAISS add + file I/O (lock released) |

### `/health` (~17 req/s, ~10% of traffic)

| Function | Time | DB Queries |
|----------|------|-----------|
| `health()` | ~1ms | 1 SELECT 1 (ping) |

### Server-Wide Resource Budget at 10K req/min

| Resource | Per Request | At 167 req/s | Budget (4 workers) | Headroom |
|----------|------------|-------------|-------------------|----------|
| CPU (search) | ~1ms | ~167ms/s | 4000ms/s (4 cores) | **96%** |
| CPU (total) | ~6ms | ~1000ms/s | 4000ms/s | **75%** |
| DB connections | 2-3 concurrent | ~100 active | 75 pool | Tight at peak |
| DB queries | 2-3 | ~350/s | MySQL handles ~5000/s | **93%** |
| Memory (FAISS index) | — | 8MB | 32MB (4 workers) | **Negligible** |
| Memory (embeddings dict) | — | 9MB | 36MB (4 workers) | **Negligible** |
| Memory (model) | — | 80MB | 320MB (4 workers) | **Noticeable** |
| Network (MySQL) | ~5KB | ~850KB/s | GB/s link | **Negligible** |

### Bottlenecks at Scale

| Bottleneck | Threshold | Mitigation |
|-----------|-----------|-----------|
| DB connection pool | ~75 concurrent | Increase `pool_size`; add PgBouncer/ProxySQL |
| FAISS lock contention | ~500 req/s per worker | Use `--workers 4` (no cross-worker contention); switch to IVFPQ |
| Model memory per worker | ~80MB each | Use `--workers 1` + async; or pre-warm model at startup |
| Visit commit fsync | ~2000 writes/s | Set `innodb_flush_log_at_trx_commit=2`; or batch visits

## Test Results

### Self-Contained Test (`tests/test.py`)

Uses a **temporary directory** with SQLite + a fresh FAISS index seeded from 300 products in `products.csv`.  
**Nothing touches production resources.**

**13/13 checks passed:**

| # | Check |
|---|-------|
| 1 | GET /health returns ok |
| 2 | Saved product visits |
| 3 | Recommendations endpoint responds (~8 ms) |
| 4 | Got non-empty recommendations (6 products) |
| 5 | At most 10 recommendations |
| 6 | All IDs unique |
| 7 | No purchased products in recommendations |
| 8 | Category diversity enforced (max 2/category) |
| 9 | All recommended products are active |
| 10 | FAISS index loaded with vectors |
| 11 | Local backup `.npz` exists |
| 12 | Embedding for visited product loaded (384 dims) |
| 13 | `stored_index` fully populated |

### Integration Test (`test_validation.py`)

Runs against the actual MySQL database and production FAISS index.  
**Use only after seeding the DB.** Same 13 checks as above.

## Deployment

### Prerequisites

- Python 3.10+
- MySQL 8.0+ with a database (e.g. `nexKirana`)
- `products.csv` in the parent directory (id, name, price, category, description, image, barcode, created_at)

### Setup

```bash
cd recommender2

# Install dependencies
pip install -r requirements.txt

# Configure .env
# DB_URL=mysql+pymysql://user:password@host:port/database

# Seed 5,000 products from CSV into DB + FAISS + backup
python seed_from_csv.py

# To rebuild from scratch (clears DB and re-embeds everything)
python seed_from_csv.py --rebuild
```

### Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
# Fast self-contained test (no production resources)
python tests/test.py

# Integration test (requires seeded DB + FAISS index)
python test_validation.py
```

### Production Considerations

- Run behind **uvicorn with workers** (`--workers 4`) or a process manager (supervisor, systemd).
- The SentenceTransformer model is cached in memory after the first request to `/product-embedding`.
- FAISS index is loaded at startup and persists in memory — restart the server to pick up re-seeded data.
- Embeddings are kept in an in-memory dict (`engine._backup_embeddings`) populated at startup from local `.npz` backup and updated by API requests.
- Local `.npz` backup is created during seeding; use `sync_local_backup()` to persist runtime updates to disk.
- Use MySQL connection pooling (default via `pool_pre_ping=True`).
