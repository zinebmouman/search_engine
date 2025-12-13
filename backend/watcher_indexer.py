# backend/watcher_indexer.py
import os
import time
import threading
import portalocker
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from search_core import upsert_document_index, DATA_DIR, DEFAULT_EMBED_PATH

LOCK_FILE = os.path.join(DATA_DIR, "indexer.lock")

SUPPORTED = (".pdf", ".txt", ".html", ".htm")

def guess_doc_type(path: str) -> str:
    p = path.lower()
    if p.endswith(".pdf"):
        return "pdf"
    if p.endswith(".txt"):
        return "txt"
    if p.endswith(".html") or p.endswith(".htm"):
        return "html"
    return "unknown"

def wait_file_ready(path: str, tries: int = 10, sleep_s: float = 0.5):
    # Attendre que la copie soit terminée (taille stable)
    last = -1
    for _ in range(tries):
        try:
            size = os.path.getsize(path)
            if size == last and size > 0:
                return True
            last = size
        except Exception:
            pass
        time.sleep(sleep_s)
    return True

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        self._handle(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        self._handle(event.src_path)

    def _handle(self, path: str):
        if not path.lower().endswith(SUPPORTED):
            return

        # debounce in a thread
        threading.Thread(target=self._process, args=(path,), daemon=True).start()

    def _process(self, path: str):
        wait_file_ready(path)

        doc_type = guess_doc_type(path)
        if doc_type == "unknown":
            return

        # lock global (avoid concurrent indexing)
        with portalocker.Lock(LOCK_FILE, timeout=60):
            print(f"[WATCHER] Indexing: {path}")
            try:
                res = upsert_document_index(
                    abs_path=path,
                    doc_type=doc_type,
                    embed_path=DEFAULT_EMBED_PATH,
                    compute_embeddings=True
                )
                print("[WATCHER] Done:", res)
            except Exception as e:
                print("[WATCHER] Failed:", e)

def main():
    paths = [
        os.path.join(DATA_DIR, "pdfs"),
        os.path.join(DATA_DIR, "texts"),
        os.path.join(DATA_DIR, "htmls"),
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)

    event_handler = Handler()
    observer = Observer()
    for p in paths:
        observer.schedule(event_handler, p, recursive=False)

    observer.start()
    print("[WATCHER] Running. Watching data folders...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

if __name__ == "__main__":
    main()
