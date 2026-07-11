import csv
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(__file__))

from db import engine, init_db, store_embedding
from initialize import add_to_index, _ensure_index_and_mapping, sync_local_backup, engine as rec_engine

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "products.csv")
BATCH_SIZE = 32

model = SentenceTransformer("all-MiniLM-L6-v2")


def read_products_from_csv():
    products = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            products.append({
                "product_id": int(row["id"]),
                "product_name": row["name"],
                "product_category": row["category"],
                "product_description": row["description"],
                "product_price": float(row.get("price", 0)),
                "product_image": row.get("image", ""),
                "product_barcode": row.get("barcode", ""),
            })
    return products


def seed_embeddings(rebuild=False):
    print(f"Reading products from {CSV_PATH}")
    products = read_products_from_csv()
    print(f"Found {len(products)} products")

    init_db()
    _ensure_index_and_mapping()

    with engine.connect() as conn:
        conn.execute(
            text("""
                CREATE TABLE IF NOT EXISTS product_embeddings (
                    id INTEGER AUTO_INCREMENT PRIMARY KEY,
                    product_id INTEGER UNIQUE NOT NULL,
                    embedding LONGBLOB,
                    embedding_binary LONGBLOB
                )
            """)
        )
        conn.commit()

    if rebuild:
        print("Rebuild mode: clearing existing embeddings from DB and FAISS...")
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM product_embeddings"))
            conn.execute(text("DROP TABLE IF EXISTS products"))
            conn.commit()
        import faiss
        from initialize import DIMENSION, INDEX_TYPE
        if INDEX_TYPE == "ivfpq":
            from initialize import NLIST, M, NBITS
            quantizer = faiss.IndexFlatL2(DIMENSION)
            new_index = faiss.IndexIVFPQ(quantizer, DIMENSION, NLIST, M, NBITS)
            new_index.nprobe = 10
        else:
            new_index = faiss.IndexFlatL2(DIMENSION)
        rec_engine.index = new_index
        rec_engine.stored_index = []
        existing_ids = set()
    else:
        with engine.connect() as conn:
            existing_ids = {
                row[0]
                for row in conn.execute(
                    text("SELECT product_id FROM product_embeddings")
                ).fetchall()
            }
        print(f"{len(existing_ids)} embeddings already exist in DB")

    new_batch_embeddings = []
    new_batch_ids = []
    total_skipped = 0
    total_new = 0

    for i, product in enumerate(products):
        pid = product["product_id"]
        if pid in existing_ids:
            total_skipped += 1
            continue

        text_repr = (
            f"Name = {product['product_name']}\n"
            f"Category = {product['product_category']}\n"
            f"Description = {product['product_description']}"
        )
        embedding = model.encode(text_repr)
        embedding_np = np.array(embedding, dtype=np.float32)

        new_batch_embeddings.append(embedding_np)
        new_batch_ids.append(pid)

        with engine.connect() as conn:
            store_embedding(pid, embedding_np, conn)
            conn.commit()

        total_new += 1

        if len(new_batch_ids) >= BATCH_SIZE:
            if rec_engine.index is not None and hasattr(rec_engine.index, 'is_trained') and not rec_engine.index.is_trained:
                continue
            batch = np.stack(new_batch_embeddings).astype(np.float32)
            add_to_index(batch, new_batch_ids)
            print(f"  Added batch of {len(new_batch_ids)} to FAISS index")
            new_batch_embeddings = []
            new_batch_ids = []

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(products)} products")

    if new_batch_ids:
        batch = np.stack(new_batch_embeddings).astype(np.float32)
        add_to_index(batch, new_batch_ids)
        print(f"  Added final batch of {len(new_batch_ids)} to FAISS index")

    print(f"\nDone! {total_new} new embeddings created, {total_skipped} already existed")
    sync_local_backup()

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                price REAL NOT NULL DEFAULT 0,
                category TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                image TEXT NOT NULL DEFAULT '',
                barcode TEXT NOT NULL DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active TINYINT(1) NOT NULL DEFAULT 1
            )
        """))
        conn.commit()

    with engine.connect() as conn:
        for product in products:
            conn.execute(
                text("""
                    INSERT INTO products (id, name, price, category, description, image, barcode)
                    VALUES (:id, :name, :price, :category, :description, :image, :barcode)
                    ON DUPLICATE KEY UPDATE id=id
                """),
                {
                    "id": product["product_id"],
                    "name": product["product_name"],
                    "price": product["product_price"],
                    "category": product["product_category"],
                    "description": product["product_description"],
                    "image": product["product_image"],
                    "barcode": product["product_barcode"],
                },
            )
        conn.commit()
    print(f"Products table populated with {len(products)} products")
    print("\nSeeding complete!")


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    seed_embeddings(rebuild=rebuild)
