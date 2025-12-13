# backend/main.py

from typing import List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from search_core import ensure_index_built_enhanced, search_db_enhanced, PDF_DIR

# -----------------
# Indexation au démarrage
# -----------------
# force_reindex=True pour reconstruire l'index à chaque lancement (utile pour dev / nouveaux PDFs)
ensure_index_built_enhanced(force_reindex=True, compute_embeddings=True)

app = FastAPI(
    title="Moteur de Recherche Fuzzy N-Gram + Lemmatisation + BM25 + Embeddings",
    description="Backend FastAPI pour moteur de recherche sur PDF avec fuzzy, n-grams, lemmatisation, BM25 et embeddings.",
    version="2.0.0",
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
    allow_origins=origins,  # mettre ["*"] en dev si nécessaire
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir les fichiers PDF statiques
@app.get("/pdfs/{filename}")
async def get_pdf(filename: str):
    """
    Endpoint pour servir les fichiers PDF.
    """
    pdf_path = os.path.join(PDF_DIR, filename)
    if os.path.exists(pdf_path):
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=filename
        )
    return {"error": "PDF not found"}


class SearchResult(BaseModel):
    doc_id: str
    filename: str
    score: float
    snippet: str
    match_terms: Optional[List[str]] = []  # ajouté pour voir quels termes ont matché
    pdf_url: Optional[str] = None  # URL pour accéder au PDF


@app.get("/search", response_model=List[SearchResult])
def search_endpoint(
    query: str = Query(..., min_length=1, description="Requête texte libre"),
    top_k: int = Query(5, ge=1, le=20),
):
    """
    Endpoint principal : /search?query=...&top_k=5
    Utilise la recherche améliorée : lemmatisation, fuzzy, BM25, embeddings.
    """
    results = search_db_enhanced(query, top_k=top_k)
    return [SearchResult(**r) for r in results]


@app.get("/")
def root():
    return {
        "message": "Moteur de recherche fuzzy n-gram opérationnel. Utilise /search?query=...&top_k=5"
    }


# Lancer le serveur si exécuté directement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
