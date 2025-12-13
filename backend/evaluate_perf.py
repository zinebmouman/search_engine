# backend/evaluate_perf.py
import os
import csv
import json
import time
import tracemalloc

try:
    import psutil
except Exception:
    psutil = None

from search_core import (
    ensure_index_built_enhanced,
    search_db_enhanced,
    DATA_DIR,
    DEFAULT_EMBED_PATH,
    DB_PATH
)

LOG_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

CSV_PATH = os.path.join(LOG_DIR, "metrics.csv")
JSONL_PATH = os.path.join(LOG_DIR, "metrics.jsonl")

QUERIES = [
    "artificial intelligence",
    "quantum computing",
    "ecosystem services",
    "science education risk taking",
    "mental health climate change",
]

def get_process_mem_mb():
    if psutil is None:
        return None
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)

def file_size_mb(path: str):
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)

def main():
    # 1) ensure index up-to-date
    t0 = time.perf_counter()
    res = ensure_index_built_enhanced(embed_path=DEFAULT_EMBED_PATH, compute_embeddings=True)
    t_index = (time.perf_counter() - t0) * 1000.0

    # 2) evaluate queries
    rows = []
    for q in QUERIES:
        tracemalloc.start()
        mem_before = get_process_mem_mb()
        start = time.perf_counter()

        out = search_db_enhanced(query=q, top_k=5, embed_path=DEFAULT_EMBED_PATH, use_semantic=True)

        elapsed = (time.perf_counter() - start) * 1000.0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_after = get_process_mem_mb()

        row = {
            "ts": time.time(),
            "query": q,
            "top_k": 5,
            "elapsed_ms": elapsed,
            "results_count": len(out),
            "tracemalloc_peak_kb": peak / 1024.0,
            "proc_mem_before_mb": mem_before,
            "proc_mem_after_mb": mem_after,
            "db_size_mb": file_size_mb(DB_PATH),
            "embeddings_size_mb": file_size_mb(DEFAULT_EMBED_PATH),
            "index_pass_ms": t_index,
            "index_inserted": res.get("inserted", 0),
            "index_updated": res.get("updated", 0),
            "index_skipped": res.get("skipped", 0),
        }
        rows.append(row)

        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 3) write CSV (append with header if needed)
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Done. Logs written to:", CSV_PATH, JSONL_PATH)

if __name__ == "__main__":
    main()
