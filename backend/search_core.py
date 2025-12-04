# backend/search_core.py

import os
import re
import string
import sqlite3
from collections import defaultdict
from typing import List, Dict, Tuple

import nltk
from pypdf import PdfReader
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# ==========================
# 1. Initialisation NLTK
# ==========================

def init_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


init_nltk()
stop_words_english = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ==========================
# 2. Config fichiers & DB
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
DB_PATH = os.path.join(DATA_DIR, "index_moteur_recherche.db")

PDF_FILES = [
    "Integrating_artificial_intelligence_and_quantum_computing.pdf",
    "Advanced.pdf",
    "Bibliometric-analysis-of-author.pdf",
    "Effect of forest management on the ecosystem.pdf",
    "Experiences within pharmacies  reflections of persons with visual impairment in South Africa.pdf",
    "Gender at the crossroads of mental health and climate change.pdf",
    "Geospatial_analysis.pdf",
    "Science Education - 2025 - Alzen - Developing Science Classroom Expectations That Encourage Risk‐Taking for Learning.pdf",
    "Science Education - 2025 - Dabran‐Zivan - The Importance of Science Education  Scientific Knowledge  and Evaluation.pdf",
    "Balancing continuity of care and home care schedule.pdf",
    "Ecosystem Services.pdf",
]


# ==========================
# 3. Extraction et nettoyage texte
# ==========================

def extract_text_from_pdf(path: str, max_pages: int = 999) -> str:
    """
    Extraction simple via pypdf.
    """
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
    """
    Coupe la partie 'References' ou 'Bibliography' à la fin du texte.
    """
    if not texte:
        return texte

    txt_lower = texte.lower()
    patterns = ["\nreferences\n", "\nreference\n", "\nbibliography\n"]

    last_pos = -1
    for pat in patterns:
        idx = txt_lower.rfind(pat)
        if idx != -1 and idx > last_pos:
            last_pos = idx

    if last_pos != -1:
        return texte[:last_pos]

    return texte


def nettoyer_texte(texte: str) -> str:
    """
    lower + suppression ponctuation + chiffres + espaces multiples.
    """
    if not texte:
        return ""

    ptd = texte.lower()
    ptd = ptd.translate(str.maketrans("", "", string.punctuation))
    ptd = re.sub(r"\d+", "", ptd)
    ptd = re.sub(r"\s+", " ", ptd)
    return ptd.strip()


# ==========================
# 4. N-grams & scores flous (ton Algorithm 1)
# ==========================

def generer_ngram_graph(ptd: str, n: int, stop_words) -> Dict[str, int]:
    """
    Implémente la logique de ton Algorithm 1 :
    n-grams stemmés, sans stopwords, counts.
    """
    ngram_counts = defaultdict(int)
    tokens = nltk.word_tokenize(ptd)
    document_ngrams = ngrams(tokens, n)

    for ngram in document_ngrams:
        stemmed_ngram = tuple(PorterStemmer().stem(token) for token in ngram)
        if not any(token in stop_words for token in stemmed_ngram):
            ngram_str = " ".join(stemmed_ngram)
            ngram_counts[ngram_str] += 1

    return ngram_counts


def calculer_scores_flous(ngram_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Degré(x) = count / m (ton équation 2).
    """
    scores_flous = {}
    if not ngram_counts:
        return scores_flous

    m = max(ngram_counts.values())
    if m <= 0:
        return {}

    for ngram, count in ngram_counts.items():
        score = count / m
        scores_flous[ngram] = score

    return scores_flous


# --- Version mots simples (n = 1) ---

def tokenize_to_ngrams(
    text: str,
    n: int = 1,
    language: str = "english",
    remove_stopwords: bool = True,
    do_stem: bool = True,
) -> Dict[str, int]:
    stop_words = set(stopwords.words(language)) if remove_stopwords else set()
    stemmer_local = PorterStemmer() if do_stem else None

    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if any(c.isalpha() for c in t)]

    processed_tokens = []
    for tok in tokens:
        tok_proc = tok
        if do_stem and stemmer_local is not None:
            tok_proc = stemmer_local.stem(tok_proc)
        if remove_stopwords and tok_proc in stop_words:
            continue
        processed_tokens.append(tok_proc)

    s_counts = defaultdict(int)
    if len(processed_tokens) < n:
        return dict(s_counts)

    for ng in ngrams(processed_tokens, n):
        if any(not t for t in ng):
            continue
        ngram_str = " ".join(ng)
        s_counts[ngram_str] += 1

    return dict(s_counts)


def degree_xi_td(term: str, td_counts: Dict[str, int]) -> float:
    """
    Degree(x_i, TD) = Occurrence(x_i) / m.
    """
    if not td_counts:
        return 0.0
    occ = td_counts.get(term, 0)
    m = max(td_counts.values()) if td_counts else 1
    if m <= 0:
        return 0.0
    return occ / m


# ==========================
# 5. SQLite : création et indexation
# ==========================

def init_db(db_path: str = DB_PATH):
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

    conn.commit()
    conn.close()


def indexer_corpus_en_bdd(pdf_files: List[str], db_path: str = DB_PATH):
    """
    Ton pipeline d’indexation adapté côté backend.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

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

        # --- Bigrammes (mots composés) ---
        index_counts_bigram = generer_ngram_graph(texte_clean, n=2, stop_words=stop_words_english)
        index_flou_composes = calculer_scores_flous(index_counts_bigram)

        for term, score in index_flou_composes.items():
            cur.execute(
                """
                INSERT INTO postings(doc_id, term, n, algo, score)
                VALUES (?, ?, ?, ?, ?)
            """,
                (doc_id, term, 2, "n_gram_graph", float(score)),
            )

        # --- Unigrammes (mots simples) ---
        s_counts_unigram = tokenize_to_ngrams(
            texte_clean,
            n=1,
            language="english",
            remove_stopwords=True,
            do_stem=True,
        )

        for term in s_counts_unigram.keys():
            score = degree_xi_td(term, s_counts_unigram)
            cur.execute(
                """
                INSERT INTO postings(doc_id, term, n, algo, score)
                VALUES (?, ?, ?, ?, ?)
            """,
                (doc_id, term, 1, "mots_simples", float(score)),
            )

        conn.commit()

    conn.close()
    print(f"[OK] Indexation terminée -> {db_path}")


def ensure_index_built():
    """
    Appelée au démarrage du backend : si la DB est vide, on indexe.
    """
    if not os.path.exists(DB_PATH):
        print("[INIT] DB inexistante, création + indexation…")
        indexer_corpus_en_bdd(PDF_FILES, DB_PATH)
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM postings")
        n_postings = cur.fetchone()[0]
    except sqlite3.OperationalError:
        # Table n'existe pas -> re-init
        print("[INIT] DB invalide, recréation…")
        conn.close()
        os.remove(DB_PATH)
        indexer_corpus_en_bdd(PDF_FILES, DB_PATH)
        return

    conn.close()
    if n_postings == 0:
        print("[INIT] DB vide, indexation…")
        indexer_corpus_en_bdd(PDF_FILES, DB_PATH)
    else:
        print(f"[INIT] Index déjà présent ({n_postings} postings).")


# ==========================
# 6. Prétraitement de la requête & recherche
# ==========================

def preprocess_query(query: str) -> List[str]:
    tokens = word_tokenize(query)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if any(c.isalpha() for c in t)]

    processed = []
    for tok in tokens:
        tok_proc = stemmer.stem(tok)
        if tok_proc in stop_words_english:
            continue
        processed.append(tok_proc)

    return processed


def termes_requete(query: str) -> Tuple[List[str], List[str]]:
    tokens = preprocess_query(query)

    unigrams = [" ".join([t]) for t in tokens]

    bigrams = []
    if len(tokens) >= 2:
        for bg in ngrams(tokens, 2):
            bigrams.append(" ".join(bg))

    return unigrams, bigrams


def search_db(
    query: str,
    db_path: str = DB_PATH,
    top_k: int = 5,
    w_unigram: float = 1.0,
    w_bigram: float = 1.5,
):
    unigrams, bigrams = termes_requete(query)

    if not unigrams and not bigrams:
        return []

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    scores_docs = defaultdict(float)

    # Unigrams
    if unigrams:
        placeholders = ",".join(["?"] * len(unigrams))
        sql_uni = f"""
            SELECT doc_id, term, score
            FROM postings
            WHERE algo = 'mots_simples' AND n = 1 AND term IN ({placeholders})
        """
        for doc_id, term, score in cur.execute(sql_uni, unigrams):
            scores_docs[doc_id] += float(score) * w_unigram

    # Bigrams
    if bigrams:
        placeholders = ",".join(["?"] * len(bigrams))
        sql_bi = f"""
            SELECT doc_id, term, score
            FROM postings
            WHERE algo = 'n_gram_graph' AND n = 2 AND term IN ({placeholders})
        """
        for doc_id, term, score in cur.execute(sql_bi, bigrams):
            scores_docs[doc_id] += float(score) * w_bigram

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

        # petit snippet
        snippet = clean_text[:300] + "..." if clean_text else ""

        results.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "score": float(score),
                "snippet": snippet,
            }
        )

    conn.close()
    return results
