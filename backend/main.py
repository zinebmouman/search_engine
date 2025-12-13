# backend/main.py
import os
from typing import List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from search_core import (
    ensure_index_built_enhanced,
    search_db_enhanced,
    DATA_DIR,
    get_bm25_model
)

EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npz")

app = FastAPI(
    title="Moteur de Recherche (MLOps incremental)",
    description="Indexation incrémentale + tracking + embeddings",
    version="3.0.0",
)

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchResult(BaseModel):
    doc_id: str
    filename: str
    score: float
    snippet: str
    match_terms: List[str]
    pdf_url: Optional[str] = None

@app.on_event("startup")
def on_startup():
    # 1) Index incremental (nouveaux fichiers seulement)
    ensure_index_built_enhanced(embed_path=EMBEDDINGS_PATH, compute_embeddings=True)

    # 2) Warm BM25 cache
    get_bm25_model()

@app.get("/search", response_model=List[SearchResult])
def search_endpoint(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    use_semantic: bool = Query(True),
):
    results = search_db_enhanced(
        query=query,
        top_k=top_k,
        embed_path=EMBEDDINGS_PATH,
        use_semantic=use_semantic
    )
    return [SearchResult(**r) for r in results]

@app.get("/")
def root():
    return {"message": "OK. Utilise /search?query=..."}
