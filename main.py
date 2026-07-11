from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
import numpy as np

from db import get_db, save_visit, store_embedding
from models import (
    ProductVisited,
    RecommendRequest,
    RecommendResponse,
    ProductEmbeddingRequest,
    ProductEmbeddingResponse,
)
from initialize import lifespan, add_to_index, engine
from recommender import recommend

app = FastAPI(lifespan=lifespan)


@app.post("/visitProduct")
def product_visited(visit: ProductVisited, db: Session = Depends(get_db)):
    success = save_visit(visit.userid, visit.productid, db)
    if not success:
        raise HTTPException(500, "Failed to record product visit")
    return {"success": True}


@app.post("/recommendations", response_model=RecommendResponse)
def get_recommendations(body: RecommendRequest, db: Session = Depends(get_db)):
    products = recommend(body, db)
    return RecommendResponse(recommended_product_ids=products)


_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


@app.post("/product-embedding", response_model=ProductEmbeddingResponse)
def add_product_embedding(
    request: list[ProductEmbeddingRequest], db: Session = Depends(get_db)
):
    model = _get_model()

    new_embeddings = []
    new_product_ids = []
    existing_products = []

    for product in request:
        existing = db.execute(
            text("SELECT id FROM product_embeddings WHERE product_id = :pid"),
            {"pid": product.product_id},
        ).fetchone()
        if existing:
            existing_products.append(product)
            continue

        text_repr = (
            f"Name = {product.product_name}\n"
            f"Category = {product.product_category}\n"
            f"Description = {product.product_description}"
        )
        embedding = model.encode(text_repr)
        embedding_np = np.array(embedding, dtype=np.float32)

        new_embeddings.append(embedding_np)
        new_product_ids.append(product.product_id)

        store_embedding(product.product_id, embedding_np, db)

    if new_embeddings:
        batch = np.stack(new_embeddings).astype(np.float32)
        add_to_index(batch, new_product_ids)
        db.commit()

    if not existing_products:
        return ProductEmbeddingResponse(
            success=True,
            message="All product embeddings created successfully",
            existing_products=[],
        )
    return ProductEmbeddingResponse(
        success=True,
        message="Could not embed the following products because they already exist",
        existing_products=existing_products,
    )


@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception:
        raise HTTPException(503, "Database unavailable")
