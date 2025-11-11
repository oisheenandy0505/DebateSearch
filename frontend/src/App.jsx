import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function fetchClustered({ query, k = 30, nsfw_ok = false }) {
  return fetch(`${API}/search_clustered`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, k, nsfw_ok, min_quality: 0.3, only_stanceable: true }),
  }).then((r) => r.json());
}

// --- helpers -------------------------------------------------
const stanceOrder = ["favor", "against", "none"];
const stanceLabel = { favor: "Supporting", against: "Opposing", none: "Neutral" };
const stanceColorClass = { favor: "favor", against: "against", none: "none" };

function partitionBySource(items, src) {
  if (!src || src === "all") return items;
  return items.filter((x) => (x.source || "").toLowerCase() === src);
}

function sortItems(items, sortBy) {
  if (sortBy === "recent") {
    // if timestamp is present, sort desc; else no-op
    return [...items].sort((a, b) => {
      const ta = a.timestamp ? Date.parse(a.timestamp) : 0;
      const tb = b.timestamp ? Date.parse(b.timestamp) : 0;
      return tb - ta;
    });
  }
  // default: confidence (or score fallback)
  return [...items].sort((a, b) => (b.conf ?? 0) - (a.conf ?? 0));
}

function Chip({ children }) {
  return <span className="chip">{children}</span>;
}

function Metric({ label, value, hint, dist }) {
  // dist: {favor,against,none} fractions 0..1
  const f = Math.max(0, Math.min(1, dist?.favor || 0));
  const a = Math.max(0, Math.min(1, dist?.against || 0));
  const n = Math.max(0, Math.min(1, dist?.none || 0));
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {hint && <div className="metric-hint">{hint}</div>}
      <div className="sparkbar" title={`For ${f * 100 | 0}% · Against ${a * 100 | 0}% · Neutral ${n * 100 | 0}%`}>
        <div className="seg favor" style={{ width: `${f * 100}%` }} />
        <div className="seg against" style={{ width: `${a * 100}%` }} />
        <div className="seg none" style={{ width: `${n * 100}%` }} />
      </div>
    </div>
  );
}

function ResultCard({ item, stance }) {
  const [expanded, setExpanded] = useState(false);
  const text = (item.text || "").trim();
  const isLong = text.length > 280; // tweak threshold if you want

  return (
    <div className={`card card-${stanceColorClass[stance]}`}>
      <div className="card-header">
        <div className="card-badges">
          <span className={`badge ${stanceColorClass[stance]}`}>{stanceLabel[stance]}</span>
          {item.source && <span className="badge src">{item.source}</span>}
          {typeof item.conf === "number" && <span className="badge">conf: {item.conf.toFixed(2)}</span>}
        </div>
        {item.url && (
          <a className="ext" href={item.url} target="_blank" rel="noreferrer">↗</a>
        )}
      </div>

      <div className={`card-text ${expanded ? "expanded" : "clamped"}`}>
        {text}
      </div>
      {isLong && (
        <button className="toggle-more" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "Show less" : "Show more"}
        </button>
      )}

      <div className="card-footer">
        <div className="muted">{item.timestamp ? new Date(item.timestamp).toLocaleString() : "semeval/reddit"}</div>
        <div className="actions">
          <button>Discuss</button>
          <button>Save</button>
        </div>
      </div>
    </div>
  );
}


// --- main app -----------------------------------------------
export default function App() {
  const [query, setQuery] = useState("");
  const [nsfw, setNsfw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [clusters, setClusters] = useState([]); // [{stance, items:[...]}, ...]
  const [mode, setMode] = useState("grid"); // grid | compare
  const [selectedSource, setSelectedSource] = useState("all"); // all | reddit | twitter
  const [sortBy, setSortBy] = useState("confidence"); // confidence | recent
  const qref = useRef(null);

  const counts = useMemo(() => {
    const map = { favor: 0, against: 0, none: 0 };
    (clusters || []).forEach((c) => (map[c.stance] = c.items?.length || 0));
    return map;
  }, [clusters]);

  const dist = useMemo(() => {
    const total = counts.favor + counts.against + counts.none || 1;
    return {
      favor: counts.favor / total,
      against: counts.against / total,
      none: counts.none / total,
    };
  }, [counts]);

  async function doSearch() {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const data = await fetchClustered({ query, k: 30, nsfw_ok: nsfw });
      setClusters(
        (data?.clusters || [])
          .filter((c) => stanceOrder.includes(c.stance))
          .map((c) => ({
            ...c,
            items: c.items || [],
          }))
      );
    } catch (e) {
      console.error(e);
      setClusters([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    // autofocus once
    if (qref.current) qref.current.focus();
  }, []);

  const rendered = useMemo(() => {
    const filteredSorted = {};
    for (const stance of stanceOrder) {
      const bucket = clusters.find((c) => c.stance === stance);
      const items = bucket?.items || [];
      const f = partitionBySource(items, selectedSource);
      filteredSorted[stance] = sortItems(f, sortBy);
    }
    return filteredSorted;
  }, [clusters, selectedSource, sortBy]);

  return (
    <div className="app light">
      <div className="topbar">
        <div className="brand">
          <div className="logo" aria-hidden>🔍</div>
          <div className="brand-name">Debate Search</div>
        </div>
        <div className="grow" />
      </div>

      <div className="container">
        {/* Search */}
        <div className="search-wrap">
          <div className="search-box">
            <input
              ref={qref}
              className="input"
              placeholder="Search for a topic…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && doSearch()}
            />
            <button className="btn" disabled={loading} onClick={doSearch}>
              {loading ? "Searching…" : "Search"}
            </button>
          </div>
          <label className="switch">
            <input type="checkbox" checked={nsfw} onChange={(e) => setNsfw(e.target.checked)} />
            Show NSFW/toxic
          </label>
        </div>

        {/* Filter bar */}
        <div className="filters card">
          <div className="f-left">
            <label>
              Source:
              <select value={selectedSource} onChange={(e) => setSelectedSource(e.target.value)}>
                <option value="all">All</option>
                <option value="reddit">Reddit</option>
                <option value="twitter">Twitter</option>
              </select>
            </label>
            <label>
              Sort:
              <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
                <option value="confidence">Confidence</option>
                <option value="recent">Most Recent</option>
              </select>
            </label>
          </div>
          <div className="f-right tabs">
            <button className={`tab ${mode === "grid" ? "active" : ""}`} onClick={() => setMode("grid")}>
              Grid View
            </button>
            <button className={`tab ${mode === "compare" ? "active" : ""}`} onClick={() => setMode("compare")}>
              Comparison
            </button>
          </div>
        </div>

        {/* Dashboard metrics (simple) */}
        <div className="dashboard card">
          <h3 className="dash-title">Analytics Dashboard</h3>
          <div className="metric-row">
            <Metric label="Total Posts" value={counts.favor + counts.against + counts.none} hint="" dist={dist} />
            <Metric label="For" value={counts.favor} hint="" dist={{ favor: 1, against: 0, none: 0 }} />
            <Metric label="Against" value={counts.against} hint="" dist={{ favor: 0, against: 1, none: 0 }} />
            <Metric label="Neutral" value={counts.none} hint="" dist={{ favor: 0, against: 0, none: 1 }} />
          </div>
          <div className="dist-wrap">
            <div className="dist-label">Debate Distribution</div>
            <div className="distbar">
              <div className="seg favor" style={{ width: `${dist.favor * 100}%` }} />
              <div className="seg against" style={{ width: `${dist.against * 100}%` }} />
              <div className="seg none" style={{ width: `${dist.none * 100}%` }} />
            </div>
          </div>
        </div>

        {/* Results */}
        {counts.favor + counts.against + counts.none === 0 ? (
          <div className="empty">Type a topic and hit Search to explore the debate.</div>
        ) : mode === "grid" ? (
          <>
            <div className="grid3">
              {stanceOrder.map((s) => (
                <div key={s}>
                  <div className={`panel-h ${stanceColorClass[s]}`}>
                    <strong>{stanceLabel[s]}</strong> <Chip>{(rendered[s] || []).length} posts</Chip>
                  </div>
                  {(rendered[s] || []).map((it) => (
                    <ResultCard key={it.id} item={it} stance={s} />
                  ))}
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="compare2">
            {/* left: favor */}
            <div>
              <div className={`panel-h ${stanceColorClass["favor"]}`}>
                <strong>{stanceLabel["favor"]}</strong>{" "}
                <Chip>{(rendered["favor"] || []).length} posts</Chip>
              </div>
              {(rendered["favor"] || []).map((it) => (
                <ResultCard key={it.id} item={it} stance="favor" />
              ))}
            </div>
            {/* right: against */}
            <div>
              <div className={`panel-h ${stanceColorClass["against"]}`}>
                <strong>{stanceLabel["against"]}</strong>{" "}
                <Chip>{(rendered["against"] || []).length} posts</Chip>
              </div>
              {(rendered["against"] || []).map((it) => (
                <ResultCard key={it.id} item={it} stance="against" />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
