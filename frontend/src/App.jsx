import { useState } from "react";
import axios from "axios";

const API_URL = import.meta.env?.VITE_API_URL || "http://127.0.0.1:8000";

export default function App() {
  const [query, setQuery] = useState("cryptocurrency regulation");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // flexible state: either clusters or flat hits
  const [clusters, setClusters] = useState([]); // for clustered shape
  const [hits, setHits] = useState([]);         // for array-of-hits shape

  async function handleSearch(e) {
    e?.preventDefault?.();
    setError("");
    setLoading(true);
    setClusters([]);
    setHits([]);

    try {
      const res = await axios.post(`${API_URL}/search_clustered`, { query, k: 30 }, {
        headers: { "Content-Type": "application/json" }
      });
      const data = res.data;
      console.debug("Search response:", data);

      if (Array.isArray(data)) {
        // Array of hits -> show one column
        setHits(data);
      } else if (data && Array.isArray(data.clusters)) {
        // Clustered shape
        setClusters(data.clusters);
      } else {
        setError("Unexpected response format from server.");
      }
    } catch (err) {
      console.error(err);
      setError(err?.response?.data?.detail || err.message || "Search failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: "2rem", fontFamily: "system-ui, -apple-system, Segoe UI, Roboto" }}>
      <h1 style={{ fontSize: "1.6rem", marginBottom: "1rem" }}>Debate Search</h1>

      <form onSubmit={handleSearch} style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Try: cryptocurrency regulation"
          style={{
            flex: 1, minWidth: 300, padding: "10px 12px",
            borderRadius: 8, border: "1px solid #ccc"
          }}
        />
        <button
          type="submit"
          disabled={loading || !query}
          style={{
            padding: "10px 14px", borderRadius: 8, border: "1px solid #111",
            background: "#111", color: "#fff", cursor: "pointer"
          }}
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      {error && (
        <div style={{ background: "#fee", border: "1px solid #f99", padding: 12, borderRadius: 8, marginBottom: 16 }}>
          {error}
        </div>
      )}

      {loading && <SkeletonGrid />}

      {!loading && clusters.length === 0 && hits.length === 0 && !error && (
        <div style={{ color: "#666" }}>No results yet. Enter a query to begin.</div>
      )}

      {/* Render clustered shape if present */}
      {!loading && clusters.length > 0 && (
        <ClusterGrid clusters={clusters} />
      )}

      {/* Render flat hits if present */}
      {!loading && hits.length > 0 && (
        <HitsGrid hits={hits} />
      )}
    </div>
  );
}

function SkeletonGrid() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
      {[0, 1, 2].map(i => (
        <div key={i} style={{ border: "1px solid #eee", borderRadius: 12, padding: 16 }}>
          <div style={{ height: 18, background: "#eee", borderRadius: 6, marginBottom: 12 }} />
          {[...Array(6)].map((_, j) => (
            <div key={j} style={{ height: 12, background: "#f2f2f2", borderRadius: 6, marginBottom: 8 }} />
          ))}
        </div>
      ))}
    </div>
  );
}

/* ===== clustered rendering ===== */
function ClusterGrid({ clusters }) {
  const stanceOrder = ["favor", "against", "none"];
  const stanceLabel = { favor: "Support", against: "Oppose", none: "Neutral" };
  const ordered = stanceOrder.map(
    s => clusters.find(c => c.stance === s) || { stance: s, items: [] }
  );
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
      {ordered.map(c => (
        <div key={c.stance} style={{ border: "1px solid #eee", borderRadius: 12, padding: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
            <h3 style={{ margin: 0, fontSize: 18 }}>{stanceLabel[c.stance] ?? c.stance}</h3>
            <span style={{ fontSize: 12, color: "#666" }}>{c.items?.length ?? 0}</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {(c.items ?? []).slice(0, 10).map(item => (
              <DocCard
                key={item.id}
                title={item.title || ""}
                text={item.text || item.body || ""}
                url={item.url}
                source={item.source}
                conf={item.conf}
                stance={item.stance}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ===== flat hits rendering ===== */
function HitsGrid({ hits }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 16 }}>
      <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
          <h3 style={{ margin: 0, fontSize: 18 }}>All results</h3>
          <span style={{ fontSize: 12, color: "#666" }}>{hits.length}</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
          {hits.slice(0, 30).map(h => (
            <DocCard
              key={h.id}
              title={h.title || ""}
              text={h.body || ""}
              url={h.url}
              source={h.source}
              score={h.score}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function DocCard({ title, text, url, source, score, conf, stance }) {
  const preview = (text || "").slice(0, 240) + ((text || "").length > 240 ? "…" : "");
  const rightSide = conf != null
    ? `${(conf * 100).toFixed(1)}%`
    : (score != null ? `${score.toFixed(2)}` : "");

  return (
    <div style={{ border: "1px solid #f0f0f0", borderRadius: 10, padding: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {source && <Badge>{source}</Badge>}
          {stance && <Badge>{stance}</Badge>}
        </div>
        {rightSide && <Badge>{rightSide}</Badge>}
      </div>
      {title && <div style={{ fontWeight: 600, marginBottom: 6 }}>{title}</div>}
      <div style={{ fontSize: 14, lineHeight: 1.4, color: "#222" }}>{preview}</div>
      {url && (
        <div style={{ marginTop: 8 }}>
          <a href={url} target="_blank" rel="noreferrer" style={{ fontSize: 13 }}>
            Open source ↗
          </a>
        </div>
      )}
    </div>
  );
}

function Badge({ children }) {
  return (
    <span style={{ border: "1px solid #ddd", borderRadius: 999, padding: "2px 8px", fontSize: 12, color: "#555" }}>
      {children}
    </span>
  );
}
