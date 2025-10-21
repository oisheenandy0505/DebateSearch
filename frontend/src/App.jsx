import { useState } from "react";
import axios from "axios";

const API_URL = "http://localhost:8000"; // FastAPI backend

function App() {
  const [query, setQuery] = useState("cryptocurrency regulation");
  const [results, setResults] = useState([]);

  const handleSearch = async () => {
    try {
      const res = await axios.get(`${API_URL}/search`, {
        params: { q: query, k: 10 },
      });
      setResults(res.data);
    } catch (err) {
      console.error(err);
      alert("Search failed. Is the backend running?");
    }
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "system-ui" }}>
      <h1 style={{ fontSize: "1.5rem" }}>🔥 Debate Search (BM25)</h1>

      <div style={{ marginTop: "1rem" }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search a topic..."
          style={{
            width: "400px",
            padding: "8px",
            borderRadius: "6px",
            border: "1px solid #ccc",
          }}
        />
        <button
          onClick={handleSearch}
          style={{
            marginLeft: "10px",
            padding: "8px 14px",
            borderRadius: "6px",
            border: "1px solid #aaa",
            cursor: "pointer",
            background: "#f4f4f4",
          }}
        >
          Search
        </button>
      </div>

      <div style={{ marginTop: "2rem" }}>
        {results.map((hit) => (
          <div
            key={hit.id}
            style={{
              marginBottom: "1rem",
              padding: "1rem",
              border: "1px solid #eee",
              borderRadius: "8px",
            }}
          >
            <a href={hit.url} target="_blank" rel="noopener noreferrer">
              <h3 style={{ margin: 0 }}>{hit.title}</h3>
            </a>
            <p style={{ margin: "6px 0", color: "#555" }}>{hit.body}</p>
            <small style={{ color: "#888" }}>
              {hit.source} | Score: {hit.score.toFixed(2)}
            </small>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;