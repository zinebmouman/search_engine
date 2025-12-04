// frontend/src/api.js

const API_BASE_URL = "http://localhost:8000";

export async function searchDocs(query, topK = 5) {
  const params = new URLSearchParams({
    query,
    top_k: topK.toString(),
  });

  const res = await fetch(`${API_BASE_URL}/search?` + params.toString());
  if (!res.ok) {
    throw new Error(`Erreur API: ${res.status}`);
  }
  return await res.json();
}
