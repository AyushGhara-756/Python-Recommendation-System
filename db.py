from sqlalchemy.orm import Session, sessionmaker, DeclarativeBase
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DatabaseError, OperationalError, IntegrityError
from dotenv import load_dotenv
import os
import numpy as np

from models import ProductVisit

load_dotenv()

DBURL = os.getenv("DB_URL", "mysql+pymysql://root:root@localhost:3306/nexKirana")

engine = create_engine(DBURL, pool_pre_ping=True, echo=False, pool_size=50, max_overflow=25)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


class Base(DeclarativeBase):
    pass


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_visit(userid: int, productid: int, db: Session) -> bool:
    try:
        db.add(ProductVisit(userid=userid, productid=productid))
        db.commit()
        return True
    except (IntegrityError, OperationalError) as e:
        db.rollback()
        print(f"DB error at save_visit: {e}")
        return False


def fetch_visits(userid: int, db: Session, limit: int = 50) -> list[dict]:
    try:
        rows = (
            db.query(ProductVisit)
            .filter(ProductVisit.userid == userid)
            .order_by(ProductVisit.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {"productid": r.productid, "timestamp": r.timestamp.isoformat()}
            for r in rows
        ]
    except DatabaseError as e:
        print(f"DB error at fetch_visits: {e}")
        return []


def _parse_embedding(raw) -> list[float]:
    if isinstance(raw, bytes):
        return np.frombuffer(raw, dtype=np.float32).tolist()
    vec_str = raw.strip("[]")
    return [float(x) for x in vec_str.split(",")]


def store_embedding(product_id: int, embedding: np.ndarray, db: Session):
    binary = embedding.astype(np.float32).tobytes()
    db.execute(
        text("INSERT INTO product_embeddings (product_id, embedding, embedding_binary) "
             "VALUES (:pid, :emb, :binary)"),
        {"pid": product_id, "emb": binary, "binary": binary},
    )


def fetch_embeddings(product_ids: list[int], db: Session) -> dict[int, list[float]]:
    if not product_ids:
        return {}
    assert all(isinstance(pid, int) for pid in product_ids), "product_ids must be ints"
    try:
        placeholders = ",".join(str(pid) for pid in product_ids)
        rows = db.execute(
            text(f"SELECT product_id, COALESCE(embedding_binary, embedding) "
                 f"FROM product_embeddings WHERE product_id IN ({placeholders})")
        ).fetchall()
        return {row[0]: _parse_embedding(row[1]) for row in rows}
    except DatabaseError as e:
        print(f"DB error at fetch_embeddings: {e}")
        return {}


def fetch_product_details(product_ids: list[int], db: Session) -> list[dict]:
    if not product_ids:
        return []
    assert all(isinstance(pid, int) for pid in product_ids), "product_ids must be ints"
    try:
        placeholders = ",".join(str(pid) for pid in product_ids)
        rows = db.execute(
            text(f"SELECT id, name, category, is_active FROM products "
                 f"WHERE id IN ({placeholders})")
        ).fetchall()
        return [
            {"id": r[0], "name": r[1], "category": r[2], "is_active": r[3]}
            for r in rows
        ]
    except DatabaseError as e:
        print(f"DB error at fetch_product_details: {e}")
        return []
