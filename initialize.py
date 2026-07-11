import os
import faiss
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI

from models import RecommendationEngine
from db import init_db

_basedir = os.path.dirname(__file__)
INDEX_PATH = os.path.join(_basedir, "..", "products.index")
BACKUP_PATH = INDEX_PATH + ".npz"
DIMENSION = 384
NLIST = 100
M = 8
NBITS = 8

INDEX_TYPE = os.getenv("RECOMMENDER_INDEX_TYPE", "flat").lower()

engine = RecommendationEngine()


def _load_existing_index():
    if not os.path.exists(INDEX_PATH):
        return None
    idx = faiss.read_index(INDEX_PATH)
    if isinstance(idx, faiss.IndexIVFPQ):
        idx.nprobe = 10
    return idx


def _create_new_index():
    if INDEX_TYPE == "ivfpq":
        quantizer = faiss.IndexFlatL2(DIMENSION)
        idx = faiss.IndexIVFPQ(quantizer, DIMENSION, NLIST, M, NBITS)
        idx.nprobe = 10
    else:
        idx = faiss.IndexFlatL2(DIMENSION)
    return idx


def _load_stored_ids() -> list[int]:
    ids_path = INDEX_PATH + ".ids"
    if not os.path.exists(ids_path):
        return []
    with open(ids_path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def _save_stored_ids(ids: list[int]):
    _save_stored_ids_from_list(ids)


def _save_stored_ids_from_list(ids: list[int]):
    ids_path = INDEX_PATH + ".ids"
    os.makedirs(os.path.dirname(ids_path) or ".", exist_ok=True)
    with open(ids_path, "w") as f:
        for pid in ids:
            f.write(f"{pid}\n")


def sync_local_backup():
    if not engine._backup_embeddings or not engine.stored_index:
        return
    all_embs = []
    all_ids = []
    for pid in engine.stored_index:
        emb = engine._backup_embeddings.get(pid)
        if emb is not None:
            all_embs.append(emb)
            all_ids.append(pid)
    if not all_embs:
        return
    stacked = np.stack(all_embs).astype(np.float32)
    np.savez_compressed(BACKUP_PATH, embeddings=stacked, product_ids=np.array(all_ids, dtype=np.int32))
    print(f"Local backup saved: {len(all_ids)} embeddings -> {BACKUP_PATH}")


def load_embedding_from_backup(product_id: int) -> np.ndarray | None:
    if not os.path.exists(BACKUP_PATH):
        return None
    try:
        data = np.load(BACKUP_PATH)
        pids = data["product_ids"]
        idx = int(np.where(pids == product_id)[0][0])
        return data["embeddings"][idx].astype(np.float32)
    except (IndexError, KeyError, ValueError):
        return None


def load_embeddings_batch_from_backup(product_ids: list[int]) -> dict[int, np.ndarray]:
    if not os.path.exists(BACKUP_PATH):
        return {}
    try:
        data = np.load(BACKUP_PATH)
        all_pids = data["product_ids"]
        all_embs = data["embeddings"]
        pid_to_pos = {int(pid): i for i, pid in enumerate(all_pids)}
        result = {}
        for pid in product_ids:
            pos = pid_to_pos.get(pid)
            if pos is not None:
                result[pid] = all_embs[pos].astype(np.float32)
        return result
    except (IOError, ValueError, KeyError) as e:
        print(f"Error loading backup file {BACKUP_PATH}: {e}")
        return {}


def _ensure_index_and_mapping():
    idx = _load_existing_index()
    if idx is None:
        idx = _create_new_index()

    stored_ids = _load_stored_ids()
    if stored_ids and idx.ntotal != len(stored_ids):
        stored_ids = stored_ids[: idx.ntotal]

    with engine._lock:
        engine.index = idx
        engine.stored_index = stored_ids

    if stored_ids and os.path.exists(BACKUP_PATH):
        engine._backup_embeddings = load_embeddings_batch_from_backup(stored_ids)


def add_to_index(embeddings: np.ndarray, product_ids: list[int]):
    with engine._lock:
        idx = engine.index
        if idx is None:
            raise RuntimeError("FAISS index not initialized")

        if isinstance(idx, faiss.IndexIVFPQ) and not idx.is_trained:
            print(f"Training IndexIVFPQ with {len(embeddings)} vectors...")
            idx.train(embeddings)

        idx.add(embeddings)
        engine.stored_index.extend(product_ids)
        for pid, emb in zip(product_ids, embeddings):
            if len(engine._backup_embeddings) < engine._max_backup_entries:
                engine._backup_embeddings[pid] = emb
        stored_copy = list(engine.stored_index)

    faiss.write_index(idx, INDEX_PATH)
    _save_stored_ids_from_list(stored_copy)
    sync_local_backup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _ensure_index_and_mapping()
    yield
