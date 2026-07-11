"""Self-contained test for the recommendation engine.

Uses a temporary directory, SQLite, and a small FAISS index seeded from
products.csv.  Nothing touches the real PostgreSQL DB or production files.
"""

import os
import sys
import csv
import time
import shutil
import tempfile
from pathlib import Path

import numpy as np

RECOMMENDER = Path(__file__).resolve().parent.parent
ROOT = RECOMMENDER.parent
sys.path.insert(0, str(RECOMMENDER))

_tmp = tempfile.mkdtemp(prefix="rec_test_")
_index_path = os.path.join(_tmp, "products.index")
_backup_path = _index_path + ".npz"


def main() -> bool:
    all_checks: list[tuple[str, bool]] = []

    # ── 1. Setup ────────────────────────────────────────────────────────
    os.environ["DB_URL"] = f"sqlite:///{_tmp}/test.db"

    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session, sessionmaker
    from sentence_transformers import SentenceTransformer

    import db
    import initialize
    import faiss

    db.engine = create_engine(os.environ["DB_URL"], echo=False)
    db.SessionLocal = sessionmaker(bind=db.engine, expire_on_commit=False, class_=Session)
    db.Base.metadata.create_all(bind=db.engine)

    initialize.INDEX_PATH = _index_path
    initialize.BACKUP_PATH = _backup_path
    DIMENSION = 384

    with db.engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL, price REAL DEFAULT 0,
                category TEXT DEFAULT '', description TEXT DEFAULT '',
                image TEXT DEFAULT '', barcode TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                is_active INTEGER DEFAULT 1
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS product_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER UNIQUE NOT NULL,
                embedding TEXT NOT NULL
            )
        """))
        conn.commit()

    CSV_PATH = os.path.join(ROOT, "products.csv")
    SAMPLE_SIZE = 300

    products_raw = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= SAMPLE_SIZE:
                break
            products_raw.append({
                "product_id": int(row["id"]),
                "product_name": row["name"],
                "product_category": row["category"],
                "product_description": row["description"],
                "product_price": float(row.get("price", 0)),
                "product_image": row.get("image", ""),
                "product_barcode": row.get("barcode", ""),
            })

    # Insert products into SQLite
    with db.engine.connect() as conn:
        for p in products_raw:
            conn.execute(
                text("""INSERT INTO products (id, name, price, category, description, image, barcode)
                         VALUES (:id, :name, :price, :cat, :desc, :img, :bc)
                         ON CONFLICT (id) DO NOTHING"""),
                {"id": p["product_id"], "name": p["product_name"],
                 "price": p["product_price"], "cat": p["product_category"],
                 "desc": p["product_description"], "img": p["product_image"],
                 "bc": p["product_barcode"]},
            )
        conn.commit()
    print(f"Seeded {len(products_raw)} products into SQLite")

    # ── 2. Generate embeddings & populate FAISS + DB ────────────────────
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=_tmp)

    quantizer = faiss.IndexFlatL2(DIMENSION)
    idx = faiss.IndexIVFPQ(quantizer, DIMENSION, 100, 8, 8)
    idx.nprobe = 10
    initialize.engine.index = idx
    initialize.engine.stored_index = []

    all_embeddings = []
    all_ids = []
    embedding_strs = []

    for p in products_raw:
        text_repr = (
            f"Name = {p['product_name']}\n"
            f"Category = {p['product_category']}\n"
            f"Description = {p['product_description']}"
        )
        emb = model.encode(text_repr)
        emb_np = np.array(emb, dtype=np.float32)
        all_embeddings.append(emb_np)
        all_ids.append(p["product_id"])
        embedding_strs.append("[" + ",".join(f"{v}" for v in emb) + "]")

    with db.engine.connect() as conn:
        for pid, estr in zip(all_ids, embedding_strs):
            conn.execute(
                text("INSERT INTO product_embeddings (product_id, embedding) VALUES (:pid, :emb)"),
                {"pid": pid, "emb": estr},
            )
        conn.commit()

    batch = np.stack(all_embeddings).astype(np.float32)
    print(f"Training IVFPQ with {len(batch)} vectors...")
    idx.train(batch)
    idx.add(batch)
    initialize.engine.stored_index.extend(all_ids)
    for pid, emb in zip(all_ids, batch):
        initialize.engine._backup_embeddings[pid] = emb

    faiss.write_index(idx, _index_path)
    with open(_index_path + ".ids", "w") as f:
        for pid in initialize.engine.stored_index:
            f.write(f"{pid}\n")
    np.savez_compressed(_backup_path,
        embeddings=batch,
        product_ids=np.array(all_ids, dtype=np.int32))
    print(f"FAISS index ready: {idx.ntotal} vectors, backup saved\n")

    # ── 3. Tests ────────────────────────────────────────────────────────
    from fastapi.testclient import TestClient
    from main import app

    print(f"{'='*60}")
    print("RECOMMENDATION ENGINE — SELF-CONTAINED TEST")
    print(f"{'='*60}\n")

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}
        all_checks.append(("GET /health returns ok", True))
        print("  [✓] GET /health returns ok")

        TEST_USER = 88888
        visited_pids = [products_raw[i % len(products_raw)]["product_id"]
                        for i in range(6)]
        for pid in visited_pids:
            r = client.post("/visitProduct", json={"userid": TEST_USER, "productid": pid})
            assert r.status_code == 200
            assert r.json() == {"success": True}
        all_checks.append((f"Saved {len(visited_pids)} product visits", True))
        print(f"  [✓] Saved {len(visited_pids)} product visits")

        orders = [
            {"created_at": "2026-06-25T10:00:00.000Z",
             "items": [{"product_id": products_raw[0]["product_id"], "quantity": 2}]},
            {"created_at": "2026-06-20T14:30:00.000Z",
             "items": [{"product_id": products_raw[3]["product_id"], "quantity": 1}]},
        ]
        payload = {"user_id": TEST_USER, "orders": orders}
        t0 = time.time()
        r = client.post("/recommendations", json=payload)
        elapsed = time.time() - t0
        assert r.status_code == 200
        data = r.json()
        recs = data.get("recommended_product_ids", [])

        all_checks.append(("Recommendations endpoint responds", True))
        print(f"  [✓] Recommendations endpoint responds ({elapsed*1000:.0f}ms)")

        if recs:
            all_checks.append(("Got non-empty recommendations", True))
            print(f"  [✓] Got {len(recs)} recommendations: {recs}")

            assert len(recs) <= 10
            all_checks.append(("At most 10 recommendations", True))

            assert len(recs) == len(set(recs))
            all_checks.append(("All IDs unique", True))

            purchased = {items["product_id"] for order in orders for items in order["items"]}
            assert not (set(recs) & purchased)
            all_checks.append(("No purchased products in recommendations", True))

            sess = db.SessionLocal()
            try:
                placeholders = ",".join(str(pid) for pid in recs)
                rows = sess.execute(
                    text(f"SELECT category, COUNT(*) FROM products WHERE id IN ({placeholders}) GROUP BY category")
                ).fetchall()
                for cat, cnt in rows:
                    assert cnt <= 2, f"Category '{cat}' has {cnt} products (max 2)"
                all_checks.append(("Category diversity enforced (max 2/category)", True))
            finally:
                sess.close()

            sess = db.SessionLocal()
            try:
                placeholders = ",".join(str(pid) for pid in recs)
                rows = sess.execute(
                    text(f"SELECT is_active FROM products WHERE id IN ({placeholders})")
                ).fetchall()
                assert all(r[0] == 1 for r in rows)
                all_checks.append(("All recommended products are active", True))
            finally:
                sess.close()
        else:
            all_checks.append(("Recommendations returned (may be empty)", False))
            print("  [!] Recommendations returned empty")

        assert initialize.engine.index is not None
        assert initialize.engine.index.ntotal > 0
        all_checks.append(("FAISS index loaded with vectors", True))

        assert os.path.exists(_backup_path)
        all_checks.append(("Local backup .npz exists", True))

        test_pid = products_raw[0]["product_id"]
        emb = initialize.load_embedding_from_backup(test_pid)
        assert emb is not None
        assert len(emb) == DIMENSION
        all_checks.append((f"Embedding for product {test_pid} loaded (384 dims)", True))

        assert len(initialize.engine.stored_index) == len(products_raw)
        all_checks.append(("stored_index fully populated", True))

    total = len(all_checks)
    passed = sum(1 for _, p in all_checks if p)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} checks passed")
    print(f"{'='*60}")

    return passed == total


if __name__ == "__main__":
    success = False
    try:
        success = main()
    finally:
        shutil.rmtree(_tmp, ignore_errors=True)
        print(f"Cleaned up temp directory: {_tmp}")

    sys.exit(0 if success else 1)
