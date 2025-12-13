# backend/search_core.py
import os
import re
import time
import json
import uuid
import hashlib
import string
import sqlite3
import logging
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import nltk
from pypdf import PdfReader
from bs4 import BeautifulSoup

try:
    import numpy as np
except Exception:
    np = None

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:
    SentenceTransformer = None
    st_util = None

from nltk.util import ngrams
from nltk.tokenize import word_tokenize


# ==========================
# 0) Paths & constants
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
TEXT_DIR = os.path.join(DATA_DIR, "texts")
HTML_DIR = os.path.join(DATA_DIR, "htmls")
LOG_DIR = os.path.join(DATA_DIR, "logs")

DB_PATH = os.path.join(DATA_DIR, "index_moteur_recherche.db")
DEFAULT_EMBED_PATH = os.path.join(DATA_DIR, "embeddings.npz")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "indexer.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("search_core")


# ==========================
# 1) NLTK init
# ==========================

def init_nltk():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]
    for r in resources:
        try:
            nltk.data.find(
                r if "/" in r else (f"tokenizers/{r}" if r == "punkt" else f"corpora/{r}")
            )
        except Exception:
            try:
                nltk.download(r, quiet=True)
            except Exception:
                pass

init_nltk()

try:
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    stop_words_english = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
except Exception:
    stop_words_english = set()

    class DummyLemmatizer:
        def lemmatize(self, word, pos="n"):
            return word

    lemmatizer = DummyLemmatizer()


# ==========================
# 2) Embedding model (lazy)
# ==========================

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None and SentenceTransformer is not None:
        _embedding_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embedding_model


# ==========================
# 3) SQLite helpers (WAL + busy_timeout)
# ==========================

def connect_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


# ==========================
# 4) Schema (MLOps tracking)
# ==========================

def init_db_enhanced(db_path: str = DB_PATH):
    conn = connect_db(db_path)
    cur = conn.cursor()

    # documents table: store metadata
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL UNIQUE,
            doc_type TEXT NOT NULL,
            clean_text TEXT,
            file_hash TEXT,
            file_size INTEGER,
            added_at TEXT,
            updated_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS postings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            term TEXT NOT NULL,
            n INTEGER NOT NULL,
            algo TEXT NOT NULL,
            score REAL NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_meta (
            doc_id TEXT PRIMARY KEY,
            has_embedding INTEGER DEFAULT 0
        )
    """)

    # logs of updates
    cur.execute("""
        CREATE TABLE IF NOT EXISTS updates_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            action TEXT NOT NULL,          -- INSERT / UPDATE / DELETE / ERROR
            filename TEXT,
            doc_id TEXT,
            status TEXT NOT NULL,          -- OK / FAIL
            duration_ms REAL,
            file_size INTEGER,
            file_hash TEXT,
            details TEXT
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_postings_doc_id ON postings(doc_id)")

    conn.commit()
    conn.close()
    
    ensure_columns_exist(db_path)


def ensure_columns_exist(db_path: str = DB_PATH):
    """
    Migration légère : ajoute les colonnes manquantes dans documents (sans supprimer la DB).
    """
    conn = connect_db(db_path)
    cur = conn.cursor()

    # Récupérer les colonnes existantes
    cur.execute("PRAGMA table_info(documents)")
    existing_cols = {row[1] for row in cur.fetchall()}

    # Colonnes attendues
    needed = {
        "doc_type": "TEXT",
        "file_hash": "TEXT",
        "file_size": "INTEGER",
        "added_at": "TEXT",
        "updated_at": "TEXT",
    }

    for col, col_type in needed.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE documents ADD COLUMN {col} {col_type}")
            conn.commit()

    conn.close()


def log_update(action: str, filename: str, doc_id: Optional[str], status: str,
               duration_ms: float = 0.0, file_size: Optional[int] = None,
               file_hash: Optional[str] = None, details: Optional[dict] = None,
               db_path: str = DB_PATH):
    conn = connect_db(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO updates_log(ts, action, filename, doc_id, status, duration_ms, file_size, file_hash, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        action,
        filename,
        doc_id,
        status,
        float(duration_ms),
        int(file_size) if file_size is not None else None,
        file_hash,
        json.dumps(details or {}, ensure_ascii=False)
    ))
    conn.commit()
    conn.close()


# ==========================
# 5) File hashing + scanning
# ==========================

def compute_file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def list_data_files() -> List[Tuple[str, str]]:
    """
    Return list of (absolute_path, doc_type) for all supported files inside data folders.
    doc_type in {"pdf","txt","html"}
    """
    files: List[Tuple[str, str]] = []

    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs(HTML_DIR, exist_ok=True)

    for name in os.listdir(PDF_DIR):
        if name.lower().endswith(".pdf"):
            files.append((os.path.join(PDF_DIR, name), "pdf"))

    for name in os.listdir(TEXT_DIR):
        if name.lower().endswith(".txt"):
            files.append((os.path.join(TEXT_DIR, name), "txt"))

    for name in os.listdir(HTML_DIR):
        if name.lower().endswith(".html") or name.lower().endswith(".htm"):
            files.append((os.path.join(HTML_DIR, name), "html"))

    return files


# ==========================
# 6) Extractors
# ==========================

def extract_text_from_pdf(abs_path: str, max_pages: int = 999) -> str:
    texte = ""
    with open(abs_path, "rb") as f:
        reader = PdfReader(f)
        n_pages = min(len(reader.pages), max_pages)
        for i in range(n_pages):
            texte += "\n" + (reader.pages[i].extract_text() or "")
    return texte.strip()

def extract_text_from_txt(abs_path: str) -> str:
    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def extract_text_from_html(abs_path: str) -> str:
    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    for element in soup(["script", "style", "header", "footer", "nav", "form"]):
        element.decompose()
    texte = soup.get_text()
    texte = "\n".join([line.strip() for line in texte.splitlines() if line.strip()])
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()

def enlever_references(texte: str) -> str:
    if not texte:
        return texte
    txt_lower = texte.lower()
    patterns = [r"\nreferences\b", r"\nreference\b", r"\nbibliography\b", r"\nbibliographie\b"]
    last_pos = -1
    for pat in patterns:
        m = re.search(pat, txt_lower)
        if m:
            last_pos = max(last_pos, m.start())
    return texte[:last_pos] if last_pos != -1 else texte

def nettoyer_texte(texte: str) -> str:
    if not texte:
        return ""
    ptd = texte.lower()
    ptd = ptd.translate(str.maketrans("", "", string.punctuation))
    ptd = re.sub(r"\d+", "", ptd)
    ptd = re.sub(r"\s+", " ", ptd)
    return ptd.strip()


# ==========================
# 7) Tokenization / Lemmatization
# ==========================

def pos_tag_to_wordnet(tag: str):
    try:
        tag = (tag or "").upper()
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN
    except Exception:
        return "n"

def preprocess_and_lemmatize(text: str, language: str = "english",
                             remove_stopwords: bool = True, min_len: int = 3) -> List[str]:
    if not text:
        return []

    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if any(c.isalpha() for c in t) and len(t) >= min_len]

    stop_words = set()
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words(language))
        except Exception:
            stop_words = stop_words_english

    try:
        from nltk import pos_tag
        pos_tags = pos_tag(tokens)
    except Exception:
        pos_tags = [(t, "") for t in tokens]

    out = []
    for tok, pos in pos_tags:
        if remove_stopwords and tok in stop_words:
            continue
        wn_pos = pos_tag_to_wordnet(pos)
        try:
            lemma = lemmatizer.lemmatize(tok, pos=wn_pos)
        except Exception:
            lemma = lemmatizer.lemmatize(tok)
        if len(lemma) >= min_len:
            out.append(lemma)
    return out

def tokenize_to_ngrams_lemm(text: str, n: int = 1, min_len: int = 3) -> Dict[str, int]:
    tokens = preprocess_and_lemmatize(text, min_len=min_len)
    counts = defaultdict(int)
    if len(tokens) < n:
        return {}
    for ng in ngrams(tokens, n):
        term = " ".join(ng)
        counts[term] += 1
    return dict(counts)


# ==========================
# 8) BM25
# ==========================

class SimpleBM25:
    def __init__(self, corpus_term_counts: List[Dict[str, int]], k1=1.5, b=0.75):
        self.corpus = corpus_term_counts
        self.N = len(corpus_term_counts)
        self.avgdl = sum(sum(d.values()) for d in corpus_term_counts) / max(1, self.N)
        self.k1 = k1
        self.b = b
        self.df = defaultdict(int)
        for d in corpus_term_counts:
            for t in d.keys():
                self.df[t] += 1

    def score(self, query_terms: List[str], doc_index: int) -> float:
        from math import log
        if not self.corpus or doc_index >= len(self.corpus):
            return 0.0

        doc = self.corpus[doc_index]
        dl = sum(doc.values()) if doc else 0
        score = 0.0

        for q in query_terms:
            f = doc.get(q, 0)
            if f == 0:
                continue
            df_q = self.df.get(q, 0)
            idf = log((self.N - df_q + 0.5) / (df_q + 0.5) + 1.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
            score += idf * (f * (self.k1 + 1)) / denom

        return float(score)

_bm25_model: Optional[SimpleBM25] = None

def reset_bm25_cache():
    global _bm25_model
    _bm25_model = None

def get_bm25_model(db_path: str = DB_PATH) -> SimpleBM25:
    global _bm25_model
    if _bm25_model is not None:
        return _bm25_model

    conn = connect_db(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT doc_id, clean_text FROM documents ORDER BY doc_id")
        docs = cur.fetchall()
    except sqlite3.OperationalError:
        docs = []
    conn.close()

    corpus_counts = []
    for _, clean_text in docs:
        corpus_counts.append(tokenize_to_ngrams_lemm(clean_text, n=1))

    _bm25_model = SimpleBM25(corpus_counts) if corpus_counts else SimpleBM25([])
    return _bm25_model


# ==========================
# 9) Embeddings store (.npz) incremental
# ==========================

def load_embeddings_map(path: str) -> Dict[str, "np.ndarray"]:
    if np is None or not os.path.exists(path):
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}

def save_embeddings_map(emb_map: Dict[str, List[float]], path: str):
    if np is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{k: np.array(v, dtype=np.float32) for k, v in emb_map.items()})


# ==========================
# 10) Incremental indexing (core)
# ==========================

def extract_by_type(abs_path: str, doc_type: str) -> str:
    if doc_type == "pdf":
        return extract_text_from_pdf(abs_path)
    if doc_type == "txt":
        return extract_text_from_txt(abs_path)
    if doc_type == "html":
        return extract_text_from_html(abs_path)
    raise ValueError(f"Unsupported doc_type: {doc_type}")

def delete_document_everything(doc_id: str, db_path: str = DB_PATH):
    conn = connect_db(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM postings WHERE doc_id = ?", (doc_id,))
    cur.execute("DELETE FROM doc_meta WHERE doc_id = ?", (doc_id,))
    cur.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()

def upsert_document_index(
    abs_path: str,
    doc_type: str,
    embed_path: str = DEFAULT_EMBED_PATH,
    compute_embeddings: bool = True,
    db_path: str = DB_PATH
):
    filename = os.path.basename(abs_path)
    start = time.perf_counter()

    try:
        file_size = os.path.getsize(abs_path)
        file_hash = compute_file_hash(abs_path)

        init_db_enhanced(db_path)

        # check existing
        conn = connect_db(db_path)
        cur = conn.cursor()
        cur.execute("SELECT doc_id, file_hash FROM documents WHERE filename = ?", (filename,))
        row = cur.fetchone()
        conn.close()

        if row:
            existing_doc_id, existing_hash = row
            if existing_hash == file_hash:
                # unchanged -> nothing
                return {"action": "SKIP", "filename": filename, "doc_id": existing_doc_id}

            # modified -> delete old then reinsert
            delete_document_everything(existing_doc_id, db_path=db_path)
            action = "UPDATE"
            doc_id = existing_doc_id
        else:
            action = "INSERT"
            doc_id = str(uuid.uuid4())

        raw_text = extract_by_type(abs_path, doc_type)
        if not raw_text:
            raise ValueError("Empty extracted text")

        clean_text = nettoyer_texte(enlever_references(raw_text))

        # write documents + postings + meta
        conn = connect_db(db_path)
        cur = conn.cursor()

        now = datetime.utcnow().isoformat()

        cur.execute("""
            INSERT OR REPLACE INTO documents(doc_id, filename, doc_type, clean_text, file_hash, file_size, added_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT added_at FROM documents WHERE doc_id = ?), ?), ?)
        """, (doc_id, filename, doc_type, clean_text, file_hash, file_size, doc_id, now, now))

        # postings insert
        # bigrams
        bigram_counts = tokenize_to_ngrams_lemm(clean_text, n=2)
        m = max(bigram_counts.values()) if bigram_counts else 1
        for term, count in bigram_counts.items():
            cur.execute(
                "INSERT INTO postings(doc_id, term, n, algo, score) VALUES (?, ?, ?, ?, ?)",
                (doc_id, term, 2, "n_gram_lemm", float(count / m))
            )

        # unigrams
        unigram_counts = tokenize_to_ngrams_lemm(clean_text, n=1)
        mu = max(unigram_counts.values()) if unigram_counts else 1
        for term, count in unigram_counts.items():
            cur.execute(
                "INSERT INTO postings(doc_id, term, n, algo, score) VALUES (?, ?, ?, ?, ?)",
                (doc_id, term, 1, "mots_lemm", float(count / mu))
            )

        # embeddings incremental
        has_emb = 0
        if compute_embeddings and SentenceTransformer is not None and np is not None:
            try:
                model = get_embedding_model()
                snippet = clean_text[:2048]
                vec = model.encode(snippet, show_progress_bar=False)
                has_emb = 1

                existing = load_embeddings_map(embed_path)
                existing[doc_id] = vec.astype(np.float32)
                # rewrite whole npz (simple & safe)
                np.savez_compressed(embed_path, **existing)
            except Exception as e:
                logger.warning("Embedding failed for %s: %s", filename, e)
                has_emb = 0

        cur.execute("INSERT OR REPLACE INTO doc_meta(doc_id, has_embedding) VALUES (?, ?)", (doc_id, has_emb))
        conn.commit()
        conn.close()

        # invalidate BM25 cache
        reset_bm25_cache()

        dur = (time.perf_counter() - start) * 1000.0
        log_update(
            action=action,
            filename=filename,
            doc_id=doc_id,
            status="OK",
            duration_ms=dur,
            file_size=file_size,
            file_hash=file_hash,
            details={"doc_type": doc_type, "unigrams": len(unigram_counts), "bigrams": len(bigram_counts)}
        )

        return {"action": action, "filename": filename, "doc_id": doc_id, "duration_ms": dur}

    except Exception as e:
        dur = (time.perf_counter() - start) * 1000.0
        log_update(
            action="ERROR",
            filename=os.path.basename(abs_path),
            doc_id=None,
            status="FAIL",
            duration_ms=dur,
            details={"error": str(e), "path": abs_path, "doc_type": doc_type}
        )
        raise


def index_new_or_modified_documents(
    embed_path: str = DEFAULT_EMBED_PATH,
    compute_embeddings: bool = True,
    db_path: str = DB_PATH
) -> Dict[str, int]:
    """
    Scan data folders and index only new/changed files.
    Return counts.
    """
    init_db_enhanced(db_path)
    files = list_data_files()

    inserted = 0
    updated = 0
    skipped = 0

    for abs_path, doc_type in files:
        res = upsert_document_index(
            abs_path=abs_path,
            doc_type=doc_type,
            embed_path=embed_path,
            compute_embeddings=compute_embeddings,
            db_path=db_path
        )
        if res["action"] == "INSERT":
            inserted += 1
        elif res["action"] == "UPDATE":
            updated += 1
        elif res["action"] == "SKIP":
            skipped += 1

    return {"inserted": inserted, "updated": updated, "skipped": skipped}


def ensure_index_built_enhanced(embed_path: str = DEFAULT_EMBED_PATH, compute_embeddings: bool = True):
    """
    Ensure schema exists and do an incremental indexing pass.
    """
    init_db_enhanced(DB_PATH)
    return index_new_or_modified_documents(embed_path=embed_path, compute_embeddings=compute_embeddings, db_path=DB_PATH)


# ==========================
# 11) Search (using cached BM25)
# ==========================

def fuzzy_score(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if fuzz is not None:
        try:
            return fuzz.partial_ratio(a, b) / 100.0
        except Exception:
            pass
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()

def search_db_enhanced(
    query: str,
    db_path: str = DB_PATH,
    top_k: int = 5,
    w_unigram: float = 1.0,
    w_bigram: float = 1.5,
    w_bm25: float = 1.5,
    w_fuzzy: float = 0.7,
    w_semantic: float = 1.0,
    embed_path: str = DEFAULT_EMBED_PATH,
    use_semantic: bool = True,
    fuzzy_threshold: float = 0.6
):
    unigrams = preprocess_and_lemmatize(query)
    bigrams = [" ".join(bg) for bg in ngrams(unigrams, 2)] if len(unigrams) >= 2 else []

    if not unigrams and not bigrams:
        return []

    conn = connect_db(db_path)
    cur = conn.cursor()

    scores_docs = defaultdict(float)
    doc_hits = defaultdict(lambda: {"match_terms": set()})

    # 1) exact matches
    if unigrams:
        placeholders = ",".join(["?"] * len(unigrams))
        sql = f"""
            SELECT doc_id, term, score
            FROM postings
            WHERE algo='mots_lemm' AND n=1 AND term IN ({placeholders})
        """
        for doc_id, term, score in cur.execute(sql, unigrams):
            scores_docs[doc_id] += float(score) * w_unigram
            doc_hits[doc_id]["match_terms"].add(term)

    if bigrams:
        placeholders = ",".join(["?"] * len(bigrams))
        sql = f"""
            SELECT doc_id, term, score
            FROM postings
            WHERE algo='n_gram_lemm' AND n=2 AND term IN ({placeholders})
        """
        for doc_id, term, score in cur.execute(sql, bigrams):
            scores_docs[doc_id] += float(score) * w_bigram
            doc_hits[doc_id]["match_terms"].add(term)

    # 2) fuzzy
    if fuzz is not None:
        cur.execute("SELECT DISTINCT term FROM postings")
        all_terms = [r[0] for r in cur.fetchall()]

        for q in unigrams + bigrams:
            if q in all_terms:
                continue
            best_term, best_score = None, 0.0
            for t in all_terms:
                s = fuzzy_score(q, t)
                if s > best_score:
                    best_score, best_term = s, t

            if best_term and best_score >= fuzzy_threshold:
                cur.execute("SELECT doc_id, score FROM postings WHERE term=?", (best_term,))
                for doc_id, score in cur.fetchall():
                    scores_docs[doc_id] += float(score) * w_fuzzy * best_score
                    doc_hits[doc_id]["match_terms"].add(f"{best_term} (fuzzy:{best_score:.2f})")

    # 3) docs order stable
    cur.execute("SELECT doc_id, clean_text FROM documents ORDER BY doc_id")
    docs = cur.fetchall()
    docid_to_index = {doc_id: i for i, (doc_id, _) in enumerate(docs)}

    bm25 = get_bm25_model(db_path)
    for doc_id, idx in docid_to_index.items():
        s = bm25.score(unigrams, idx)
        if s:
            scores_docs[doc_id] += float(s) * w_bm25

    # 4) semantic (optional)
    if use_semantic and SentenceTransformer is not None and np is not None and st_util is not None:
        emb_map = load_embeddings_map(embed_path)
        model = get_embedding_model()
        if model:
            try:
                q_emb = model.encode(query, convert_to_tensor=True)
                q_vec = q_emb.cpu().numpy()
                q_norm = np.linalg.norm(q_vec) + 1e-12
            except Exception:
                q_vec, q_norm = None, None

            if q_vec is not None:
                for doc_id, clean_text in docs:
                    sim_norm = 0.0
                    if doc_id in emb_map:
                        d_vec = np.array(emb_map[doc_id])
                        d_norm = np.linalg.norm(d_vec) + 1e-12
                        sim = float(np.dot(q_vec, d_vec) / (q_norm * d_norm))
                        sim_norm = (sim + 1.0) / 2.0
                    scores_docs[doc_id] += sim_norm * w_semantic

    if not scores_docs:
        conn.close()
        return []

    ranked = sorted(scores_docs.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc_id, score in ranked:
        cur.execute("SELECT filename, doc_type, clean_text FROM documents WHERE doc_id=?", (doc_id,))
        row = cur.fetchone()
        if not row:
            continue
        filename, doc_type, clean_text = row
        snippet = (clean_text[:300] + "...") if clean_text else ""

        if doc_type == "pdf":
            url = f"/pdfs/{filename}"
        elif doc_type == "txt":
            url = f"/texts/{filename}"
        elif doc_type == "html":
            url = f"/htmls/{filename}"
        else:
            url = None

        results.append({
            "doc_id": doc_id,
            "filename": filename,
            "score": float(score),
            "snippet": snippet,
            "match_terms": sorted(list(doc_hits[doc_id]["match_terms"])) if doc_id in doc_hits else [],
            "pdf_url": url
        })

    conn.close()
    return results
