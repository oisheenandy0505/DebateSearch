import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const DEFAULT_API = (() => {
  if (typeof window === "undefined") return "http://127.0.0.1:8000";
  const { protocol, hostname, port } = window.location;
  if (!port) return `${protocol}//${hostname}`;
  if (port === "5173") return `${protocol}//${hostname}:8000`;
  return `${protocol}//${hostname}:${port}`;
})();

const API = import.meta.env.VITE_API_URL || DEFAULT_API;

async function fetchClustered({ query, k = 30, nsfw_ok = false, only_stanceable = false }) {
  const res = await fetch(`${API}/search_clustered`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, k, nsfw_ok, min_quality: 0.3, only_stanceable }),
  });
  let payload = null;
  try {
    payload = await res.json();
  } catch {
    payload = null;
  }
  if (!res.ok) {
    const detail = payload?.detail || payload?.message;
    throw new Error(detail || `Search failed (${res.status})`);
  }
  return payload;
}

// Helpers -----------------------------------------------------------
const stanceOrder = ["favor", "against", "none"];
const stanceLabel = { favor: "Supporting", against: "Opposing", none: "Neutral" };
const stanceColorClass = { favor: "favor", against: "against", none: "none" };
function normalizeSource(value) {
  const normalized = (value || "").toLowerCase();
  if (!normalized) return "";
  if (["twitter", "semeval", "twitter/semeval"].includes(normalized)) return "semeval";
  return normalized;
}
const sourceOptions = [
  { value: "all", label: "All" },
  { value: "reddit", label: "Reddit" },
  { value: "semeval", label: "Twitter/SemEval" },
];
function partitionBySource(items, src) {
  if (!src || src === "all") return items;
  const needle = normalizeSource(src);
  return items.filter((x) => normalizeSource(x.source) === needle);
}

function sortItems(items, sortBy) {
  if (sortBy === "recent") {
    // Sort newest first when timestamps exist; fall back to stable ordering.
    return [...items].sort((a, b) => {
      const ta = a.timestamp ? Date.parse(a.timestamp) : 0;
      const tb = b.timestamp ? Date.parse(b.timestamp) : 0;
      return tb - ta;
    });
  }
  // Default: sort by stance confidence (or score fallback).
  return [...items].sort((a, b) => (b.conf ?? 0) - (a.conf ?? 0));
}

function Chip({ children }) {
  return <span className="chip">{children}</span>;
}

function Metric({ label, value, hint, dist }) {
  // dist is a {favor,against,none} fraction map.
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

function ResultCard({ item, stance, saved = false, onToggleSave }) {
  const [expanded, setExpanded] = useState(false);
  const text = (item.text || item.body || "").trim();
  const isLong = text.length > 140; // two-line teaser before expanding
  const author = item.author || item.user || item.username || (item.source ? `${item.source} contributor` : "Community member");
  const avatarLabel = author ? author.toString().trim().slice(0, 1).toUpperCase() : "D";
  const confidence = typeof item.conf === "number" ? item.conf.toFixed(2) : null;
  const stanceSource = item.stance_source || null;
  const sourceLabel = String(item.source || "dataset").replace(/^[a-z]/, (c) => c.toUpperCase());

  return (
    <div className={`card card-${stanceColorClass[stance]}`}>
      <div className="card-header">
        <div className="card-header-left">
          <div className="avatar" aria-hidden>{avatarLabel}</div>
          <div>
            <div className="author">{author}</div>
            <div className="source-row">
              <span className="source-pill">{sourceLabel}</span>
              <span className={`badge ${stanceColorClass[stance]}`}>{stanceLabel[stance]}</span>
            </div>
          </div>
        </div>
        <div className="card-header-right">
          {confidence && <span className="confidence">Conf {confidence}</span>}
          {!confidence && stanceSource === "gold" && (
            <span className="confidence">Curated stance</span>
          )}
          {item.url && (
            <a className="ext" href={item.url} target="_blank" rel="noreferrer">Open ↗</a>
          )}
        </div>
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
        <div>
          <div className="muted">{item.timestamp ? new Date(item.timestamp).toLocaleString() : "Semeval / Reddit"}</div>
          <div className="muted lighter">#{item.id || "debate"}</div>
        </div>
        {onToggleSave && (
          <div className="actions">
            <button
              type="button"
              className={`save-btn ${saved ? "saved" : ""}`}
              onClick={() => onToggleSave(item, stance)}
            >
              {saved ? "Saved" : "Save"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


// Main component ---------------------------------------------------
export default function App() {
  const [query, setQuery] = useState("");
  const [nsfw, setNsfw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [clusters, setClusters] = useState([]); // [{stance, items:[...]}, ...]
  const [mode, setMode] = useState("grid"); // "grid" or "compare"
  const [selectedSource, setSelectedSource] = useState("all"); // "all" | "reddit" | "semeval"
  const [onlyStanceable, setOnlyStanceable] = useState(false);
  const [sortBy, setSortBy] = useState("confidence"); // "confidence" | "recent"
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [density, setDensity] = useState("cozy"); // "cozy" | "compact"
  const [savedItems, setSavedItems] = useState([]);
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

  async function doSearch(nextQuery) {
    const activeQuery = (typeof nextQuery === "string" ? nextQuery : query).trim();
    if (!activeQuery) return;
    if (nextQuery) {
      setQuery(activeQuery);
    }
    setLoading(true);
    setError("");
    try {
      const data = await fetchClustered({ query: activeQuery, k: 30, nsfw_ok: nsfw, only_stanceable: onlyStanceable });
      setClusters(
        (data?.clusters || [])
          .filter((c) => stanceOrder.includes(c.stance))
          .map((c) => ({
            ...c,
            items: (c.items || []).map((item) => ({
              ...item,
              conf: typeof item.stance_confidence === "number" ? item.stance_confidence : null,
            })),
          }))
      );
    } catch (e) {
      console.error(e);
      setError(e?.message || "Search failed");
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
  const sortLabel = sortBy === "recent" ? "Most recent" : "Confidence";
  const savedIds = useMemo(() => new Set(savedItems.map((it) => it.id)), [savedItems]);

  function handleToggleSave(item, stance) {
    if (!item?.id) return;
    setSavedItems((prev) => {
      const exists = prev.find((entry) => entry.id === item.id);
      if (exists) {
        return prev.filter((entry) => entry.id !== item.id);
      }
      return [...prev, { ...item, savedStance: stance }];
    });
  }

  return (
    <div className="app light">
      <header className="app-header">
        <div className="brand">
          <div className="logo" aria-hidden>💬</div>
          <div>
            <h1>Debate Search</h1>
            <p>Discover how different voices support or oppose the same topic.</p>
          </div>
        </div>
      </header>

      <main className={`container density-${density}`}>
        <section className="search-card card">
          <div className="search-box">
            <input
              ref={qref}
              className="input"
              placeholder="Search for a topic…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && doSearch()}
            />
            <button className="btn" disabled={loading} onClick={() => doSearch()}>
              {loading ? "Searching…" : "Search"}
            </button>
          </div>
          <p className="search-hint">Try questions like “carbon tax benefits” or “grad school rankings”.</p>
          {error && <div className="error" role="alert">{error}</div>}
        </section>

        <section className="controls card">
          <div className="control-row">
            <label>
              Source
              <select value={selectedSource} onChange={(e) => setSelectedSource(e.target.value)}>
                {sourceOptions.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </label>
            <label>
              Sort
              <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
                <option value="confidence">Confidence</option>
                <option value="recent">Most Recent</option>
              </select>
            </label>
          </div>
          <div className="control-row">
            <div className="pill-segmented" role="tablist" aria-label="View mode">
              <button
                type="button"
                className={mode === "grid" ? "active" : ""}
                onClick={() => setMode("grid")}
              >
                Grid
              </button>
              <button
                type="button"
                className={mode === "compare" ? "active" : ""}
                onClick={() => setMode("compare")}
              >
                Compare
              </button>
            </div>
            <div className="pill-segmented density-toggle" role="group" aria-label="Result density">
              <button
                type="button"
                className={density === "cozy" ? "active" : ""}
                onClick={() => setDensity("cozy")}
                aria-pressed={density === "cozy"}
              >
                Relaxed
              </button>
              <button
                type="button"
                className={density === "compact" ? "active" : ""}
                onClick={() => setDensity("compact")}
                aria-pressed={density === "compact"}
              >
                Compact
              </button>
            </div>
            <button
              type="button"
              className="command-advanced"
              onClick={() => setShowAdvanced((v) => !v)}
            >
              {showAdvanced ? "Hide advanced" : "Advanced filters"}
            </button>
          </div>
        </section>

        {showAdvanced && (
          <div className="advanced-panel card">
            <label className="switch">
              <input type="checkbox" checked={nsfw} onChange={(e) => setNsfw(e.target.checked)} />
              <div className="switch-text">
                <span className="switch-title">Include sensitive posts</span>
                <span className="switch-hint">NSFW or toxic language</span>
              </div>
            </label>
            <label className="switch">
              <input type="checkbox" checked={onlyStanceable} onChange={(e) => setOnlyStanceable(e.target.checked)} />
              <div className="switch-text">
                <span className="switch-title">Only show posts that pick a side</span>
                <span className="switch-hint">Relies on stance labels or curated tags</span>
              </div>
            </label>
          </div>
        )}

        {savedItems.length > 0 && (
          <section className="saved card">
            <div className="saved-header">
              <h3>Saved posts ({savedItems.length})</h3>
              <button type="button" onClick={() => setSavedItems([])}>Clear all</button>
            </div>
            <div className="saved-grid">
              {savedItems.map((entry) => (
                <ResultCard
                  key={`saved-${entry.id}`}
                  item={entry}
                  stance={entry.savedStance || entry.stance_label || "none"}
                  saved
                  onToggleSave={handleToggleSave}
                />
              ))}
            </div>
          </section>
        )}

        <section className="insights">
          <div className="insight-grid">
            <Metric label="Total Posts" value={counts.favor + counts.against + counts.none} hint="Across all stances" dist={dist} />
            <Metric label="Supporting" value={counts.favor} hint={`${(dist.favor * 100).toFixed(0)}% of debate`} dist={{ favor: 1, against: 0, none: 0 }} />
            <Metric label="Opposing" value={counts.against} hint={`${(dist.against * 100).toFixed(0)}% of debate`} dist={{ favor: 0, against: 1, none: 0 }} />
            <Metric label="Neutral" value={counts.none} hint={`${(dist.none * 100).toFixed(0)}% of debate`} dist={{ favor: 0, against: 0, none: 1 }} />
          </div>
          <div className="dist-card card">
            <div className="dist-header">
              <h3>Debate distribution</h3>
              <span className="muted small">Share of returned posts</span>
            </div>
            <div className="distbar large">
              <div className="seg favor" style={{ width: `${dist.favor * 100}%` }} />
              <div className="seg against" style={{ width: `${dist.against * 100}%` }} />
              <div className="seg none" style={{ width: `${dist.none * 100}%` }} />
            </div>
          </div>
        </section>

        {counts.favor + counts.against + counts.none === 0 ? (
          <div className="empty">Type a topic and hit Search to explore the debate.</div>
        ) : mode === "grid" ? (
          <>
            <div className="section-heading">
              <div>
                <p className="eyebrow">Debate threads</p>
                <h3>Scan every stance side-by-side</h3>
              </div>
              <span className="muted small">Sorted by {sortLabel}</span>
            </div>
            <div className="grid3">
              {stanceOrder.map((s) => (
                <div key={s} className={`stance-column stance-${s}`}>
                  <div className={`panel-h ${stanceColorClass[s]}`}>
                    <strong>{stanceLabel[s]}</strong> <Chip>{(rendered[s] || []).length} posts</Chip>
                  </div>
                  {(rendered[s] || []).map((it) => (
                    <ResultCard
                      key={it.id}
                      item={it}
                      stance={s}
                      saved={savedIds.has(it.id)}
                      onToggleSave={handleToggleSave}
                    />
                  ))}
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="compare2">
            <div className="stance-column stance-favor">
              <div className={`panel-h ${stanceColorClass["favor"]}`}>
                <strong>{stanceLabel["favor"]}</strong>{" "}
                <Chip>{(rendered["favor"] || []).length} posts</Chip>
              </div>
              {(rendered["favor"] || []).map((it) => (
                <ResultCard
                  key={it.id}
                  item={it}
                  stance="favor"
                  saved={savedIds.has(it.id)}
                  onToggleSave={handleToggleSave}
                />
              ))}
            </div>
            <div className="stance-column stance-against">
              <div className={`panel-h ${stanceColorClass["against"]}`}>
                <strong>{stanceLabel["against"]}</strong>{" "}
                <Chip>{(rendered["against"] || []).length} posts</Chip>
              </div>
              {(rendered["against"] || []).map((it) => (
                <ResultCard
                  key={it.id}
                  item={it}
                  stance="against"
                  saved={savedIds.has(it.id)}
                  onToggleSave={handleToggleSave}
                />
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
