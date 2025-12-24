export const API_BASE_URL = "/api";

export async function searchDocs(query, topK = 5, useSemantic = true) {
  const params = new URLSearchParams({
    query,
    top_k: String(topK),
    use_semantic: useSemantic ? "true" : "false",
  });

  const res = await fetch(`${API_BASE_URL}/search?${params.toString()}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
