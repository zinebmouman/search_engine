// frontend/src/api.js

const API_BASE_URL = "http://localhost:8000";

export async function searchDocs(query, topK = 5) {
  const params = new URLSearchParams({
    query,
    top_k: topK.toString(),
  });

  try {
    const res = await fetch(`${API_BASE_URL}/search?` + params.toString());
    if (!res.ok) {
      throw new Error(`Erreur API: ${res.status} - ${res.statusText}`);
    }
    return await res.json();
  } catch (error) {
    if (
      error.message.includes("Failed to fetch") ||
      error.message.includes("ERR_CONNECTION_REFUSED")
    ) {
      throw new Error(
        `Impossible de se connecter au serveur backend sur ${API_BASE_URL}. Vérifiez que le serveur est lancé avec: python main.py ou uvicorn main:app --reload`
      );
    }
    throw error;
  }
}
