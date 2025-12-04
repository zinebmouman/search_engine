// frontend/src/App.jsx

import { useState } from "react";
import { searchDocs } from "./api";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async (e) => {
    e.preventDefault();
    setError("");
    setResults([]);

    if (!query.trim()) return;

    try {
      setLoading(true);
      const data = await searchDocs(query.trim(), topK);
      setResults(data);
    } catch (err) {
      console.error(err);
      setError("Erreur lors de la recherche (voir console).");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-root">

      {/* ----------- TOP SECTION ----------- */}
      <div className="hero-section">
        <h1 className="app-title">SciFindr</h1>
        <p className="subtitle">
          AI-powered scientific PDF search engine
        </p>

        {/* SEARCH BAR CENTRÉE */}
        <form className="search-box" onSubmit={handleSearch}>
          <input
            type="text"
            placeholder="Rechercher des articles scientifiques..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />

          <button type="submit" disabled={loading}>
            {loading ? "Recherche..." : "🔍"}
          </button>
        </form>

        {/* OPTIONS (Top-K) */}
        <div className="search-options">
          <label>
            Top K :
            <input
              type="number"
              min="1"
              max="20"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </label>
        </div>
      </div>

      {/* ----------- RESULTS SECTION ----------- */}
      <main className="app-main">
        {error && <div className="error-box">{error}</div>}

        {results.length > 0 && (
          <h2 className="results-title">
            Résultats ({results.length}) pour <span>"{query}"</span>
          </h2>
        )}

        <section className="results-list">
          {results.map((r) => (
            <article key={r.doc_id} className="result-card">
              <div className="result-header">
                <span className="doc-id">{r.doc_id}</span>
                <span className="score">score = {r.score.toFixed(4)}</span>
              </div>

              <h3 className="filename">{r.filename}</h3>
              <p className="snippet">{r.snippet}</p>
            </article>
          ))}

          {!loading && !error && results.length === 0 && query && (
            <p className="no-results">Aucun document trouvé.</p>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
