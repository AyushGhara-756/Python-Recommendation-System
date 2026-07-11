from __future__ import annotations

import threading
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import DateTime, func, Index
from sqlalchemy.orm import Mapped, mapped_column
import numpy as np

from db import Base


class ProductVisit(Base):
    __tablename__ = "product_visit"
    id: Mapped[int] = mapped_column(primary_key=True)
    userid: Mapped[int] = mapped_column()
    productid: Mapped[int] = mapped_column(index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_product_visit_userid_timestamp", "userid", "timestamp"),
    )


class ProductVisited(BaseModel):
    userid: int
    productid: int


class OrderItem(BaseModel):
    product_id: int
    quantity: int


class Order(BaseModel):
    created_at: str
    items: list[OrderItem]


class RecommendRequest(BaseModel):
    user_id: int
    orders: list[Order]


class RecommendResponse(BaseModel):
    recommended_product_ids: list[int]


class ProductEmbeddingRequest(BaseModel):
    product_id: int
    product_name: str
    product_category: str
    product_description: str


class ProductEmbeddingResponse(BaseModel):
    success: bool
    message: str
    existing_products: list[ProductEmbeddingRequest]


class RecommendationEngine:
    def __init__(self) -> None:
        self.index = None
        self.stored_index: list[int] = []
        self._backup_embeddings: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self._max_backup_entries = 20000
