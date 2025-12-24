# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/app/nltk_data

# Déps système utiles (numpy / torch / rapidfuzz)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    curl \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Télécharger les ressources NLTK DANS l'image
RUN python - <<'PY'
import os, nltk
path = os.environ.get("NLTK_DATA", "/app/nltk_data")
os.makedirs(path, exist_ok=True)
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, download_dir=path, quiet=True)
print("NLTK_DATA:", path)
PY

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
