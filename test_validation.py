"""
End-to-end validation script for the recommendation engine.

Tests:
1. /visitProduct — save 10 dummy visits for 2–4 products
2. /recommendations — get recommendations using order history
3. Validates response structure, business rules, category diversity
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from main import app
from db import engine as db_engine
from sqlalchemy import text

TEST_USER_ID = 99999
VISIT_PRODUCTS = [10, 25, 50, 100]


def clean_test_data():
    """Remove any previous test data for our test user."""
    with db_engine.connect() as conn:
        conn.execute(text("DELETE FROM product_visit WHERE userid = :uid"), {"uid": TEST_USER_ID})
        conn.commit()
    print("  Cleaned previous test data")


def send_visits(client):
    """Send 10 dummy product visits for 2-4 products."""
    print("\n[STEP 1] Sending 10 dummy product visits...")
    visit_log = []
    for i in range(10):
        pid = VISIT_PRODUCTS[i % len(VISIT_PRODUCTS)]
        resp = client.post("/visitProduct", json={"userid": TEST_USER_ID, "productid": pid})
        assert resp.status_code == 200, f"Visit failed: {resp.text}"
        visit_log.append({"productid": pid, "response": resp.json()})
        print(f"  Visit {i+1}: user={TEST_USER_ID}, product={pid} → {resp.json()}")

    # verify via DB
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT productid, COUNT(*) as cnt FROM product_visit WHERE userid = :uid GROUP BY productid"),
            {"uid": TEST_USER_ID}
        ).fetchall()
        print(f"\n  DB has {sum(r[1] for r in rows)} total visits across {len(rows)} products:")
        for r in rows:
            print(f"    product={r[0]}: {r[1]} visits")

    return visit_log


def send_recommendations(client):
    """Send recommendation request with sample order history."""
    print("\n[STEP 2] Requesting recommendations...")

    recent_orders = [
        {
            "created_at": "2026-06-25T10:00:00.000Z",
            "items": [
                {"product_id": 5, "quantity": 1},
                {"product_id": 15, "quantity": 2},
            ]
        },
        {
            "created_at": "2026-06-20T14:30:00.000Z",
            "items": [
                {"product_id": 5, "quantity": 3},
                {"product_id": 200, "quantity": 1},
            ]
        },
        {
            "created_at": "2026-06-15T09:00:00.000Z",
            "items": [
                {"product_id": 42, "quantity": 1},
            ]
        },
    ]

    request_body = {
        "user_id": TEST_USER_ID,
        "orders": recent_orders,
    }

    start = time.time()
    resp = client.post("/recommendations", json=request_body)
    elapsed = time.time() - start

    print(f"  Response time: {elapsed*1000:.1f}ms")
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {json.dumps(resp.json(), indent=4)}")

    return resp.json(), request_body


def validate_recommendations(response_data, request_body):
    """Thoroughly validate the recommendation response."""
    print("\n[STEP 3] Validating recommendations...")
    checks = []

    # 3a. Response structure
    assert "recommended_product_ids" in response_data, "Missing recommended_product_ids"
    recs = response_data["recommended_product_ids"]
    checks.append(("Response has recommended_product_ids key", True))

    assert isinstance(recs, list), "recommended_product_ids not a list"
    checks.append(("recommended_product_ids is a list", True))

    # 3b. Max 10 recommendations
    assert len(recs) <= 10, f"Got {len(recs)} recs, expected ≤ 10"
    checks.append((f"At most 10 recommendations (got {len(recs)})", True))

    # 3c. No purchased products in recommendations
    purchased_pids = set()
    for order in request_body["orders"]:
        for item in order["items"]:
            purchased_pids.add(item["product_id"])
    overlap = set(recs) & purchased_pids
    assert not overlap, f"Recommended purchased products: {overlap}"
    checks.append((f"No purchased products in recommendations", True))

    # 3d. Unique product IDs
    assert len(recs) == len(set(recs)), "Duplicate products in recommendations"
    checks.append(("All recommended IDs are unique", True))

    # 3e. All IDs are positive ints
    assert all(isinstance(pid, int) and pid > 0 for pid in recs), "Invalid product IDs"
    checks.append(("All IDs are positive integers", True))

    # 3f. Category diversity (max 2 per category)
    from db import fetch_product_details
    from db import SessionLocal
    db = SessionLocal()
    try:
        if recs:
            details = fetch_product_details(recs, db)
            cat_counts = {}
            for d in details:
                cat = d.get("category", "Unknown")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            for cat, cnt in cat_counts.items():
                assert cnt <= 2, f"Category '{cat}' has {cnt} products (max 2)"
            checks.append((f"Category diversity enforced (counts: {cat_counts})", True))
    finally:
        db.close()

    # 3g. No inactive products
    db = SessionLocal()
    try:
        if recs:
            details = fetch_product_details(recs, db)
            for d in details:
                assert d.get("is_active") != 0, f"Product {d['id']} is inactive"
            checks.append(("No inactive products in recommendations", True))
    finally:
        db.close()

    print()
    for label, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  [{status}] {label}")

    all_passed = all(p for _, p in checks)
    print(f"\n  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    return checks


def verify_fallback_chain():
    """Verify the fallback chain: FAISS → local backup → pgvector."""
    print("\n[STEP 4] Verifying fallback chain...")
    checks = []

    # 4a. FAISS index is loaded
    from initialize import engine
    assert engine.index is not None, "FAISS index not loaded"
    assert engine.index.ntotal > 0, "FAISS index is empty"
    checks.append((f"FAISS index loaded with {engine.index.ntotal} vectors", True))

    # 4b. Local backup file exists
    from initialize import BACKUP_PATH
    exists = os.path.exists(BACKUP_PATH)
    assert exists, "Local backup .npz file not found"
    checks.append((f"Local backup exists at {BACKUP_PATH}", True))

    # 4c. Can load an embedding from local backup
    from initialize import load_embedding_from_backup
    test_pid = VISIT_PRODUCTS[0]
    emb = load_embedding_from_backup(test_pid)
    assert emb is not None, f"Could not load embedding for product {test_pid}"
    assert len(emb) == 384, f"Embedding has {len(emb)} dimensions (expected 384)"
    checks.append((f"Embedding for product {test_pid} loaded from backup (384 dims)", True))

    # 4d. stored_index is populated
    assert len(engine.stored_index) > 0, "stored_index is empty"
    checks.append((f"stored_index has {len(engine.stored_index)} entries", True))

    print()
    for label, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  [{status}] {label}")

    return checks


def run_all():
    print("=" * 60)
    print("RECOMMENDATION ENGINE — END-TO-END VALIDATION")
    print("=" * 60)

    from fastapi.testclient import TestClient
    from main import app

    clean_test_data()

    all_checks = []

    with TestClient(app) as client:
        visit_log = send_visits(client)
        all_checks.append(("Visits sent successfully", True))

        response_data, request_body = send_recommendations(client)
    if response_data.get("recommended_product_ids"):
        checks = validate_recommendations(response_data, request_body)
        all_checks.extend(checks)
    else:
        print("\n[STEP 3] No recommendations returned — checking if fallback works...")
        all_checks.append(("Recommendations returned (may be empty)", False))

    fallback_checks = verify_fallback_chain()
    all_checks.extend(fallback_checks)

    print("\n" + "=" * 60)
    total = len(all_checks)
    passed = sum(1 for _, p in all_checks if p)
    print(f"SUMMARY: {passed}/{total} checks passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
