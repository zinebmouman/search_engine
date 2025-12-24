import time
import psutil
from prometheus_client import Counter, Histogram, Gauge
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
REQ_COUNT = Counter("search_requests_total", "Total search requests")
REQ_LATENCY = Histogram("search_request_latency_seconds", "Latency of /search endpoint")

INDEX_PASS_MS = Gauge("index_pass_ms", "Index incremental pass duration in ms")
INDEX_INSERTED = Gauge("index_inserted", "Docs inserted in last index pass")
INDEX_UPDATED = Gauge("index_updated", "Docs updated in last index pass")
INDEX_SKIPPED = Gauge("index_skipped", "Docs skipped in last index pass")

PROC_MEM_MB = Gauge("process_memory_mb", "Process RSS memory in MB")

def set_process_mem():
    rss = psutil.Process().memory_info().rss / (1024 * 1024)
    PROC_MEM_MB.set(rss)

class timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.t0
