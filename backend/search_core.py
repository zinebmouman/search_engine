# backend/search_core.py
# --- Ajouts / remplacements pour support lemmatisation, fuzzy, BM25, embeddings ---

import os
import re
import string
import sqlite3
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import nltk
from pypdf import PdfReader

# Optional libs (import safe)
try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet, stopwords
except Exception:
    # fallback: ensure nltk will be downloaded later in init
    from nltk.corpus import stopwords  # may still be needed for default

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
# 0. Configuration / constantes
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
DB_PATH = os.path.join(DATA_DIR, "index_moteur_recherche.db")

# garde ta liste PDF existante
PDF_FILES = [
    "Integrating_artificial_intelligence_and_quantum_computing.pdf",
    # ...
]

# ==========================
# 1. Initialisation NLTK (avec WordNet)
# ==========================

def init_nltk():
    # Téléchargements nécessaires (idempotent)
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]
    for r in resources:
        try:
            nltk.data.find(r if "/" in r else f"tokenizers/{r}" if r == "punkt" else f"corpora/{r}")
        except Exception:
            try:
                nltk.download(r, quiet=True)
            except Exception:
                pass

init_nltk()

# objets globaux
try:
    stop_words_english = set(stopwords.words("english"))
except Exception:
    stop_words_english = set()
lemmatizer = WordNetLemmatizer()

# Optional: model sentence-transformers (lazy loaded)
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # rapide & compact
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None and SentenceTransformer is not None:
        _embedding_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embedding_model

# ==========================
# 2. Extract & cleanup (inchangés mais conservés)
# ==========================

def extract_text_from_pdf(path: str, max_pages: int = 999) -> str:
    full_path = os.path.join(PDF_DIR, path) if not os.path.isabs(path) else path
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"PDF not found: {full_path}")

    texte = ""
    with open(full_path, "rb") as f:
        reader = PdfReader(f)
        n_pages = min(len(reader.pages), max_pages)
        for i in range(n_pages):
            page = reader.pages[i]
            page_text = page.extract_text() or ""
            texte += "\n" + page_text

    return texte.strip()

def enlever_references(texte: str) -> str:
    if not texte:
        return texte
    txt_lower = texte.lower()
    patterns = [r"\nreferences\b", r"\nreference\b", r"\nbibliography\b", r"\nbibliographie\b"]
    last_pos = -1
    for pat in patterns:
        idx = re.search(pat, txt_lower)
        if idx:
            pos = idx.start()
            if pos > last_pos:
                last_pos = pos
    if last_pos != -1:
        return texte[:last_pos]
    return texte

def nettoyer_texte(texte: str) -> str:
    if not texte:
        return ""
    ptd = texte.lower()
    ptd = ptd.translate(str.maketrans("", "", string.punctuation))
    ptd = re.sub(r"\d+", "", ptd)
    ptd = re.sub(r"\s+", " ", ptd)
    return ptd.strip()

# ==========================
# 3. Tokenisation améliorée : Lemmatisation + longueur >= 3
# ==========================

def pos_tag_to_wordnet(tag):
    """
    Map POS tag to WordNet POS tag for better lemmatization (optional).
    """
    from nltk import pos_tag
    # This function used only if we call pos_tag externally; here simple fallback is 'n'
    tag = (tag or "").upper()
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN

def preprocess_and_lemmatize(text: str, language: str = "english", remove_stopwords: bool = True, min_len: int = 3) -> List[str]:
    """
    Tokenize -> lowercase -> remove tokens shorter than min_len -> remove stopwords (if asked)
    -> lemmatize with WordNet.
    Retourne la liste de tokens lemmatisés.
    """
    if not text:
        return []

    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    # keep tokens containing a letter and length >= min_len
    tokens = [t for t in tokens if any(c.isalpha() for c in t) and len(t) >= min_len]

    stop_words = set()
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words(language))
        except Exception:
            stop_words = stop_words_english

    result = []
    # Optionnel : POS tagging pour meilleur lemmatization (coûteux)
    try:
        from nltk import pos_tag
        pos_tags = pos_tag(tokens)
    except Exception:
        pos_tags = [(t, "") for t in tokens]

    for tok, pos in pos_tags:
        if remove_stopwords and tok in stop_words:
            continue
        wn_pos = pos_tag_to_wordnet(pos)
        try:
            lemma = lemmatizer.lemmatize(tok, pos=wn_pos)
        except Exception:
            lemma = lemmatizer.lemmatize(tok)
        # filter again for length after lemmatize
        if len(lemma) >= min_len:
            result.append(lemma)
    return result

# n-gram generator using lemmatized tokens
def tokenize_to_ngrams_lemm(text: str, n: int = 1, language: str = "english", remove_stopwords: bool = True, min_len: int = 3) -> Dict[str, int]:
    tokens = preprocess_and_lemmatize(text, language=language, remove_stopwords=remove_stopwords, min_len=min_len)
    counts = defaultdict(int)
    if len(tokens) < n:
        return {}
    for ng in ngrams(tokens, n):
        if any(not t for t in ng):
            continue
        ngram_str = " ".join(ng)
        counts[ngram_str] += 1
    return dict(counts)

# ==========================
# 4. Scoring : TF, TF-IDF, BM25 simple
# ==========================

def compute_idf(all_docs_counts: List[Dict[str,int]]) -> Dict[str, float]:
    """
    all_docs_counts: list where each item is a dict(term->count) for a doc.
    Retourne IDF(term) = log((N - df + 0.5) / (df + 0.5))  (BM25-style)
    """
    import math
    N = len(all_docs_counts)
    df = defaultdict(int)
    for td in all_docs_counts:
        for t in td.keys():
            df[t] += 1
    idf = {}
    for t, df_t in df.items():
        # avoid negative/zero
        idf[t] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0)
    return idf

def compute_tfidf_for_doc(term_counts: Dict[str,int], idf: Dict[str,float]) -> Dict[str,float]:
    tfidf = {}
    max_tf = max(term_counts.values()) if term_counts else 1
    for t, tf in term_counts.items():
        tf_norm = tf / max_tf
        tfidf[t] = tf_norm * idf.get(t, 0.0)
    return tfidf

class SimpleBM25:
    """
    Very small BM25-ish implementation.
    """
    def __init__(self, corpus_term_counts: List[Dict[str,int]], k1=1.5, b=0.75):
        self.corpus = corpus_term_counts
        self.N = len(corpus_term_counts)
        self.avgdl = sum(sum(d.values()) for d in corpus_term_counts) / max(1, self.N)
        self.k1 = k1
        self.b = b
        # compute df
        self.df = defaultdict(int)
        for d in corpus_term_counts:
            for t in d.keys():
                self.df[t] += 1

    def score(self, query_terms: List[str], doc_index: int) -> float:
        from math import log
        score = 0.0
        doc = self.corpus[doc_index]
        doc_len = sum(doc.values()) if doc else 0
        for q in query_terms:
            f = doc.get(q, 0)
            if f == 0:
                continue
            idf = log((self.N - self.df.get(q,0) + 0.5) / (self.df.get(q,0) + 0.5) + 1.0)
            denom = f + self.k1 * (1 - self.b + self.b * doc_len / max(1, self.avgdl))
            score += idf * (f * (self.k1 + 1)) / denom
        return score

# ==========================
# 5. Fuzzy & semantic helpers
# ==========================

def fuzzy_score(a: str, b: str) -> float:
    """
    Retourne une score [0,1] de similarité via rapidfuzz (partial_ratio),
    fallback to simple ratio by common substring.
    """
    if a == b:
        return 1.0
    if fuzz is not None:
        try:
            return fuzz.partial_ratio(a, b) / 100.0
        except Exception:
            pass
    # fallback cheap heuristic: longest common subsequence ratio
    import difflib
    sm = difflib.SequenceMatcher(None, a, b)
    return sm.ratio()

def compute_semantic_similarity(query: str, doc_snippet: str) -> float:
    """
    Si sentence-transformers disponible, calcule cosine similarity entre embeddings.
    Sinon : fallback (overlap token / WordNet path).
    """
    # try sentence-transformers
    model = get_embedding_model()
    if model is not None and np is not None:
        try:
            q_emb = model.encode(query, convert_to_tensor=True)
            d_emb = model.encode(doc_snippet, convert_to_tensor=True)
            # util.pytorch_cos_sim/from sentence_transformers
            cosine_sim = float(st_util.cos_sim(q_emb, d_emb).cpu().numpy()[0][0])
            # normalize from [-1,1] to [0,1]
            return (cosine_sim + 1.0) / 2.0
        except Exception:
            pass

    # fallback: token-overlap normalized
    q_tokens = set(preprocess_and_lemmatize(query))
    d_tokens = set(preprocess_and_lemmatize(doc_snippet))
    if not q_tokens or not d_tokens:
        return 0.0
    overlap = len(q_tokens.intersection(d_tokens)) / max(1, len(q_tokens.union(d_tokens)))
    return float(overlap)

# ==========================
# 6. Indexation améliorée (enrichir la DB et fichiers d'embeddings)
# ==========================

def init_db_enhanced(db_path: str = DB_PATH):
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id    TEXT PRIMARY KEY,
            filename  TEXT NOT NULL,
            clean_text TEXT
        )
    """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS postings (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id  TEXT NOT NULL,
            term    TEXT NOT NULL,
            n       INTEGER NOT NULL,
            algo    TEXT NOT NULL,
            score   REAL NOT NULL
        )
    """
    )

    # optional table to store doc-level vectors existence (we store vectors in files, but keep flag)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_meta (
            doc_id TEXT PRIMARY KEY,
            has_embedding INTEGER DEFAULT 0
        )
    """
    )

    # create index on term for quicker lookup
    cur.execute("CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_postings_doc_id ON postings(doc_id)")

    conn.commit()
    conn.close()

def save_embeddings_map(emb_map: Dict[str, List[float]], path: str):
    """
    Save embeddings dict as numpy .npz if numpy available.
    emb_map: dict doc_id -> vector (list or np array)
    """
    if np is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{k: np.array(v, dtype=np.float32) for k, v in emb_map.items()})

def load_embeddings_map(path: str) -> Dict[str, np.ndarray]:
    if np is None:
        return {}
    if not os.path.exists(path):
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}

def indexer_corpus_en_bdd_enhanced(pdf_files: List[str], db_path: str = DB_PATH, embed_path: Optional[str] = None, compute_embeddings: bool = True):
    """
    Pipeline d’indexation amélioré :
    - lemmatisation
    - n-grams (1 & 2)
    - TF/BM25 prep (not stored globally in DB, calcul à la volée)
    - embeddings (optionnel) sauvegardés dans un .npz (map doc_id -> vector)
    """
    init_db_enhanced(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    embeddings_map = {}
    all_unigram_counts = []

    for i, fichier in enumerate(pdf_files, start=1):
        doc_id = f"doc{i}"
        pdf_path = os.path.join(PDF_DIR, fichier)
        print(f"[INDEX] {doc_id} <- {pdf_path}")

        try:
            texte_brut = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"Erreur extraction PDF pour {fichier} : {e}")
            continue

        if not texte_brut:
            print("Aucun texte extrait, on saute ce document.")
            continue

        texte_sans_refs = enlever_references(texte_brut)
        texte_clean = nettoyer_texte(texte_sans_refs)

        cur.execute(
            "INSERT OR REPLACE INTO documents(doc_id, filename, clean_text) VALUES (?, ?, ?)",
            (doc_id, fichier, texte_clean),
        )

        # --- Bigrammes (lemmatisés) ---
        index_counts_bigram = tokenize_to_ngrams_lemm(texte_clean, n=2, language="english", remove_stopwords=True, min_len=3)
        # Score flou / normalisation
        if index_counts_bigram:
            m = max(index_counts_bigram.values())
        else:
            m = 1
        for term, count in index_counts_bigram.items():
            score = count / m
            cur.execute(
                "INSERT INTO postings(doc_id, term, n, algo, score) VALUES (?, ?, ?, ?, ?)",
                (doc_id, term, 2, "n_gram_lemm", float(score)),
            )

        # --- Unigrammes (lemmatisés) ---
        s_counts_unigram = tokenize_to_ngrams_lemm(texte_clean, n=1, language="english", remove_stopwords=True, min_len=3)
        all_unigram_counts.append(s_counts_unigram)
        for term, count in s_counts_unigram.items():
            score = count / max(1, max(s_counts_unigram.values()))
            cur.execute(
                "INSERT INTO postings(doc_id, term, n, algo, score) VALUES (?, ?, ?, ?, ?)",
                (doc_id, term, 1, "mots_lemm", float(score)),
            )

        # optionally compute embedding for doc (use first 512 chars as snippet)
        if compute_embeddings and SentenceTransformer is not None and np is not None:
            try:
                model = get_embedding_model()
                # encode snippet or full text depending on memory
                snippet = texte_clean[:2048]
                vec = model.encode(snippet, show_progress_bar=False)
                embeddings_map[doc_id] = vec.astype(np.float32).tolist()
                cur.execute("INSERT OR REPLACE INTO doc_meta(doc_id, has_embedding) VALUES (?, ?)", (doc_id, 1))
            except Exception as e:
                print("Embedding failed for", doc_id, e)
                cur.execute("INSERT OR REPLACE INTO doc_meta(doc_id, has_embedding) VALUES (?, ?)", (doc_id, 0))
        else:
            cur.execute("INSERT OR REPLACE INTO doc_meta(doc_id, has_embedding) VALUES (?, ?)", (doc_id, 0))

        conn.commit()

    # save embeddings map to file if requested
    if embed_path and embeddings_map and np is not None:
        save_embeddings_map(embeddings_map, embed_path)

    conn.close()
    print(f"[OK] Indexation terminée -> {db_path}")

# ==========================
# 7. Recherche : combinaison lexique + fuzzy + sémantique + BM25
# ==========================

def termes_requete_lemm(query: str, min_len: int = 3) -> Tuple[List[str], List[str]]:
    tokens = preprocess_and_lemmatize(query, language="english", remove_stopwords=True, min_len=min_len)
    unigrams = [" ".join([t]) for t in tokens]
    bigrams = []
    if len(tokens) >= 2:
        for bg in ngrams(tokens, 2):
            bigrams.append(" ".join(bg))
    return unigrams, bigrams

def search_db_enhanced(
    query: str,
    db_path: str = DB_PATH,
    top_k: int = 5,
    w_unigram: float = 1.0,
    w_bigram: float = 1.5,
    w_bm25: float = 1.5,
    w_fuzzy: float = 0.7,
    w_semantic: float = 1.0,
    embed_path: Optional[str] = None,
    use_semantic: bool = True,
    fuzzy_threshold: float = 0.6 # seuil de similarité pour le fuzzy matching
):
    unigrams, bigrams = termes_requete_lemm(query)
    if not unigrams and not bigrams:
        return []

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    scores_docs = defaultdict(float)
    doc_hits = defaultdict(lambda: {"match_terms": set(), "snippet": ""})

    # Unigram exact matches
    if unigrams:
        placeholders = ",".join(["?"] * len(unigrams))
        sql_uni = f"""
            SELECT doc_id, term, score
            FROM postings
            WHERE algo = 'mots_lemm' AND n = 1 AND term IN ({placeholders})
        """
        for doc_id, term, score in cur.execute(sql_uni, unigrams):
            scores_docs[doc_id] += float(score) * w_unigram
            doc_hits[doc_id]["match_terms"].add(term)

    # Bigram exact matches
    if bigrams:
        placeholders = ",".join(["?"] * len(bigrams))
        sql_bi = f"""
            SELECT doc_id, term, score
            FROM postings
            WHERE algo = 'n_gram_lemm' AND n = 2 AND term IN ({placeholders})
        """
        for doc_id, term, score in cur.execute(sql_bi, bigrams):
            scores_docs[doc_id] += float(score) * w_bigram
            doc_hits[doc_id]["match_terms"].add(term)

    # Fuzzy fallback: if no exact match for some query terms, try fuzzy match against term dictionary
    # Get distinct known terms in postings (optionally could be indexed)
    if fuzz is not None:
        # fetch candidate terms from postings (limit to top N terms to reduce cost)
        cur.execute("SELECT DISTINCT term FROM postings")
        all_terms = [r[0] for r in cur.fetchall()]
        for q in unigrams + bigrams:
            # check if exact present; skip if present
            if any(q == t for t in all_terms):
                continue
            # find best fuzzy candidate among all_terms (could be optimized with ngram index)
            best = None
            best_score = 0.0
            for t in all_terms:
                s = fuzzy_score(q, t)
                if s > best_score:
                    best_score = s
                    best = t
            if best and best_score >= fuzzy_threshold:
                # retrieve postings for this best term
                cur.execute("SELECT doc_id, score FROM postings WHERE term = ?", (best,))
                for doc_id, score in cur.fetchall():
                    scores_docs[doc_id] += float(score) * w_fuzzy * best_score
                    doc_hits[doc_id]["match_terms"].add(best + f" (fuzzy:{best_score:.2f})")

    # BM25 over documents (requires building doc term counts)
    # We'll lazily build corpus of unigram counts for all documents
    cur.execute("SELECT doc_id, clean_text FROM documents")
    docs = cur.fetchall()
    docid_to_index = {}
    corpus_counts = []
    for idx, (doc_id, clean_text) in enumerate(docs):
        docid_to_index[doc_id] = idx
        counts = tokenize_to_ngrams_lemm(clean_text, n=1, language="english", remove_stopwords=True, min_len=3)
        corpus_counts.append(counts)
    bm25 = SimpleBM25(corpus_counts)
    # query tokens (lemmatized)
    q_tokens = [t for t in preprocess_and_lemmatize(query)]
    for doc_id, idx in docid_to_index.items():
        bm25_score = bm25.score(q_tokens, idx)
        if bm25_score:
            scores_docs[doc_id] += bm25_score * w_bm25

    # Semantic similarity (embeddings) if available
    if use_semantic and SentenceTransformer is not None and np is not None:
        # load embeddings map from embed_path if provided
        emb_map = {}
        if embed_path and os.path.exists(embed_path):
            emb_map = load_embeddings_map(embed_path)
        else:
            # try to read doc_meta to find which docs have embeddings (but embeddings must be saved by indexer)
            cur.execute("SELECT doc_id FROM doc_meta WHERE has_embedding = 1")
            good = [r[0] for r in cur.fetchall()]
            # we don't have vectors unless saved externally; fallback to computing snippet embedding on the fly (costly)
            model = get_embedding_model()
            if model:
                q_emb = model.encode(query, convert_to_tensor=True)
                for doc_id, clean_text in docs:
                    snippet = clean_text[:2048]
                    try:
                        d_emb = model.encode(snippet, convert_to_tensor=True)
                        sim = float(st_util.cos_sim(q_emb, d_emb).cpu().numpy()[0][0])
                        sim_norm = (sim + 1.0) / 2.0
                        scores_docs[doc_id] += sim_norm * w_semantic
                    except Exception:
                        pass
    else:
        # fallback cheap semantic (overlap)
        for doc_id, clean_text in docs:
            sim = compute_semantic_similarity(query, clean_text[:512])
            scores_docs[doc_id] += sim * (w_semantic * 0.3)  # downweight fallback

    # Rank results
    if not scores_docs:
        conn.close()
        return []

    ranked = sorted(scores_docs.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in ranked[:top_k]:
        cur.execute("SELECT filename, clean_text FROM documents WHERE doc_id = ?", (doc_id,))
        row = cur.fetchone()
        if row:
            filename, clean_text = row
        else:
            filename, clean_text = "Unknown", ""
        snippet = clean_text[:300] + "..." if clean_text else ""
        match_terms = sorted(list(doc_hits[doc_id]["match_terms"])) if doc_id in doc_hits else []
        # Générer l'URL du PDF pour le frontend
        pdf_url = f"/pdfs/{filename}" if filename != "Unknown" else None
        results.append({
            "doc_id": doc_id,
            "filename": filename,
            "score": float(score),
            "snippet": snippet,
            "match_terms": match_terms,
            "pdf_url": pdf_url
        })

    conn.close()
    return results
def ensure_index_built_enhanced(embed_path: Optional[str] = None, compute_embeddings: bool = True, force_reindex: bool = False):
    """
    Si force_reindex=True, réindexe toujours le corpus même si DB existante.
    """
    if force_reindex or not os.path.exists(DB_PATH):
        print("[INIT] DB inexistante ou force_reindex=True, création + indexation…")
        indexer_corpus_en_bdd_enhanced(PDF_FILES, DB_PATH, embed_path=embed_path, compute_embeddings=compute_embeddings)
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM postings")
        n_postings = cur.fetchone()[0]
    except sqlite3.OperationalError:
        print("[INIT] DB invalide, recréation…")
        conn.close()
        os.remove(DB_PATH)
        indexer_corpus_en_bdd_enhanced(PDF_FILES, DB_PATH, embed_path=embed_path, compute_embeddings=compute_embeddings)
        return

    conn.close()
    if n_postings == 0:
        print("[INIT] DB vide, indexation…")
        indexer_corpus_en_bdd_enhanced(PDF_FILES, DB_PATH, embed_path=embed_path, compute_embeddings=compute_embeddings)
    else:
        print(f"[INIT] Index déjà présent ({n_postings} postings).")
