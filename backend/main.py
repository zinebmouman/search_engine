# backend/main.py

from typing import List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from search_core import ensure_index_built, search_db

# S'assure que la DB est prête au démarrage
ensure_index_built()

app = FastAPI(
    title="Moteur de Recherche Fuzzy N-Gram",
    description="Backend FastAPI pour ton moteur de recherche sur PDF (fuzzy n-grams + SQLite).",
    version="1.0.0",
)

# CORS : autoriser le front React en local
origins = [
    "http://localhost:5173",  # Vite
    "http://localhost:3000",  # CRA
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # tu peux mettre ["*"] en dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchResult(BaseModel):
    doc_id: str
    filename: str
    score: float
    snippet: str


@app.get("/search", response_model=List[SearchResult])
def search_endpoint(
    query: str = Query(..., min_length=1, description="Requête texte libre"),
    top_k: int = Query(5, ge=1, le=20),
):
    """
    Endpoint principal : /search?query=...&top_k=5
    """
    results = search_db(query, top_k=top_k)
    return [SearchResult(**r) for r in results]


@app.get("/")
def root():
    return {"message": "Moteur de recherche fuzzy n-gram opérationnel. Utilise /search?query=..."}
