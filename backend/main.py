# backend/main.py
import os
from typing import List, Optional

from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from search_core import (
    DATA_DIR,
    ensure_index_built_enhanced,
    search_db_enhanced,
    get_bm25_model,
    connect_db,
    presigned_get_url,
    upsert_document_index_from_minio,
    put_file_bytes_to_minio,
    delete_document_everything,
    delete_object_from_minio,
)

from metrics import (
    REQ_COUNT, REQ_LATENCY, set_process_mem,
    INDEX_PASS_MS, INDEX_INSERTED, INDEX_UPDATED, INDEX_SKIPPED, timer
)

EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npz")

ALLOWED_EXT = {".pdf": "pdf", ".txt": "txt", ".html": "html", ".htm": "html"}

app = FastAPI(
    title="Moteur de Recherche Fuzzy N-Gram (Enhanced)",
    description="Backend FastAPI + Stockage MinIO (PDF/TXT/HTML)",
    version="3.1.0",
)

origins = [
    "http://localhost:5173", "http://localhost:3000",
    "http://127.0.0.1:5173", "http://127.0.0.1:3000",
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
    pdf_url: Optional[str] = None   # on garde le même champ côté front


@app.on_event("startup")
def on_startup():
    with timer() as t:
        stats = ensure_index_built_enhanced(embed_path=EMBEDDINGS_PATH, compute_embeddings=True)

    INDEX_PASS_MS.set((t.dt * 1000))
    if isinstance(stats, dict):
        INDEX_INSERTED.set(stats.get("inserted", 0))
        INDEX_UPDATED.set(stats.get("updated", 0))
        INDEX_SKIPPED.set(stats.get("skipped", 0))

    get_bm25_model()
    set_process_mem()


@app.get("/metrics")
def metrics():
    set_process_mem()
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/search", response_model=List[SearchResult])
def search_endpoint(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    use_semantic: bool = Query(True),
):
    REQ_COUNT.inc()
    with REQ_LATENCY.time():
        results = search_db_enhanced(
            query,
            top_k=top_k,
            embed_path=EMBEDDINGS_PATH,
            use_semantic=use_semantic
        )
    set_process_mem()
    return [SearchResult(**r) for r in results]


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload (PDF/TXT/HTML) -> MinIO -> indexation depuis MinIO -> SQLite.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    filename_lower = file.filename.lower().strip()
    ext = os.path.splitext(filename_lower)[1]

    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXT.keys()))}"
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # 1) Upload MinIO
    doc_id, object_key, doc_type = put_file_bytes_to_minio(file_bytes, file.filename)

    # 2) Indexation depuis MinIO (doc_type IMPORTANT)
    res = upsert_document_index_from_minio(
        doc_id=doc_id,
        filename=file.filename,
        doc_type=doc_type,
        object_key=object_key,
        embed_path=EMBEDDINGS_PATH,
        compute_embeddings=True,
    )
    return res


@app.get("/files/{doc_id}")
def get_file(doc_id: str):
    """
    Redirige vers MinIO presigned URL (PDF/TXT/HTML).
    """
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT object_key, doc_type FROM documents WHERE doc_id=?", (doc_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    object_key, doc_type = row
    if not object_key:
        raise HTTPException(status_code=400, detail="Missing object_key")

    url = presigned_get_url(object_key, expires_sec=900)
    return RedirectResponse(url=url)


@app.delete("/documents/{doc_id}")
def delete_doc(doc_id: str):
    """
    Supprime côté MinIO + DB (tous types).
    """
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT object_key FROM documents WHERE doc_id=?", (doc_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return {"status": "not_found", "doc_id": doc_id}

    object_key = row[0]
    if object_key:
        delete_object_from_minio(object_key)

    delete_document_everything(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.get("/")
def root():
    return {
        "message": "OK. Use /upload (POST), /search?query=..., /files/{doc_id}, /documents/{doc_id} (DELETE), /metrics"
    }
