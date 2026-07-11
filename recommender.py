import logging
import numpy as np
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from models import RecommendRequest
from db import fetch_visits, fetch_embeddings, fetch_product_details
from initialize import engine

logger = logging.getLogger(__name__)


PURCHASE_FREQ_WEIGHT = 1.0
PURCHASE_RECENCY_HALFLIFE = 30
VISIT_WEIGHT = 0.3
VISIT_RECENCY_HALFLIFE = 7


def _parse_timestamp(ts_str: str) -> datetime:
    try:
        s = ts_str.replace("Z", "+00:00")
        if "+" not in s:
            s += "+00:00"
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        logger.warning("Failed to parse timestamp: %r, falling back to now", ts_str)
        return datetime.now(timezone.utc)


def _days_since(dt: datetime, now: datetime | None = None) -> float:
    if now is None:
        now = datetime.now(timezone.utc)
    return (now - dt).total_seconds() / 86400.0


def _recency_weight(days: float, halflife: float) -> float:
    return np.exp(-days / halflife)


def extract_purchased_products(orders: list, now: datetime) -> dict[int, dict]:
    purchased = {}
    for order in orders:
        ts = _parse_timestamp(order.created_at)
        days = _days_since(ts, now)
        recency = _recency_weight(days, PURCHASE_RECENCY_HALFLIFE)
        for item in order.items:
            pid = item.product_id
            if pid not in purchased:
                purchased[pid] = {"frequency": 0, "recency": 0.0, "count": 0}
            purchased[pid]["frequency"] += item.quantity
            purchased[pid]["recency"] = max(purchased[pid]["recency"], recency)
            purchased[pid]["count"] += 1
    return purchased


def extract_visited_products(user_id: int, db: Session, now: datetime) -> dict[int, dict]:
    visits = fetch_visits(user_id, db)
    visited = {}
    for v in visits:
        pid = v["productid"]
        ts = _parse_timestamp(v["timestamp"])
        days = _days_since(ts, now)
        recency = _recency_weight(days, VISIT_RECENCY_HALFLIFE)
        if pid not in visited:
            visited[pid] = {"recency": 0.0}
        visited[pid]["recency"] = max(visited[pid]["recency"], recency)
    return visited


def _get_embedding(pid: int) -> np.ndarray | None:
    emb = engine._backup_embeddings.get(pid)
    if emb is not None:
        return emb
    return None


def build_user_profile(
    purchased: dict[int, dict],
    visited: dict[int, dict],
    db: Session
) -> np.ndarray | None:
    all_pids = list(purchased.keys()) + [pid for pid in visited if pid not in purchased]
    if not all_pids:
        return None

    emb_map: dict[int, np.ndarray] = {}
    misses = []
    with engine._lock:
        for pid in all_pids:
            emb = engine._backup_embeddings.get(pid)
            if emb is not None:
                emb_map[pid] = emb
            else:
                misses.append(pid)

    if misses:
        for pid, vec in fetch_embeddings(misses, db).items():
            arr = np.array(vec, dtype=np.float32)
            emb_map[pid] = arr
            with engine._lock:
                if len(engine._backup_embeddings) < engine._max_backup_entries:
                    engine._backup_embeddings[pid] = arr

    if not emb_map:
        return None

    weighted_sum = None
    total_weight = 0.0

    for pid, info in purchased.items():
        emb = emb_map.get(pid)
        if emb is None:
            continue
        w = PURCHASE_FREQ_WEIGHT * info["frequency"] * info["recency"]
        if weighted_sum is None:
            weighted_sum = emb * w
        else:
            weighted_sum += emb * w
        total_weight += w

    for pid, info in visited.items():
        if pid in purchased:
            continue
        emb = emb_map.get(pid)
        if emb is None:
            continue
        w = VISIT_WEIGHT * info["recency"]
        if weighted_sum is None:
            weighted_sum = emb * w
        else:
            weighted_sum += emb * w
        total_weight += w

    if weighted_sum is None or total_weight == 0:
        return None

    return (weighted_sum / total_weight).astype(np.float32)


def search_faiss(
    query_emb: np.ndarray,
    exclude_pids: set[int],
    top_k: int = 30
) -> list[int]:
    with engine._lock:
        if engine.index is None or engine.index.ntotal == 0:
            return []

        query_emb = query_emb.reshape(1, -1).astype(np.float32)
        k = min(top_k, engine.index.ntotal)
        distances, indices = engine.index.search(query_emb, k)
        stored_index = list(engine.stored_index)

    candidates = []
    for i in indices[0]:
        if i < 0 or i >= len(stored_index):
            continue
        pid = stored_index[i]
        if pid not in exclude_pids:
            candidates.append(pid)
    return candidates


def apply_business_rules(candidates: list[int], db: Session, max_results: int = 10) -> list[int]:
    if not candidates:
        return []

    details = fetch_product_details(candidates, db)
    detail_map = {d["id"]: d for d in details}

    filtered = []
    for pid in candidates:
        info = detail_map.get(pid)
        if info is None:
            continue
        if info.get("is_active") == 0:
            continue
        filtered.append(pid)

    category_count: dict[str, int] = {}
    MAX_PER_CATEGORY = 2
    result = []
    for pid in filtered:
        info = detail_map.get(pid, {})
        cat = info.get("category", "Unknown")
        if category_count.get(cat, 0) >= MAX_PER_CATEGORY:
            continue
        result.append(pid)
        category_count[cat] = category_count.get(cat, 0) + 1
        if len(result) >= max_results:
            break

    return result


def recommend(request: RecommendRequest, db: Session) -> list[int]:
    now = datetime.now(timezone.utc)

    purchased = extract_purchased_products(request.orders, now)
    visited = extract_visited_products(request.user_id, db, now)

    user_emb = build_user_profile(purchased, visited, db)
    if user_emb is None:
        return []

    exclude_pids = set(purchased.keys())
    candidates = search_faiss(user_emb, exclude_pids, top_k=30)
    if not candidates:
        return []

    return apply_business_rules(candidates, db, max_results=10)
