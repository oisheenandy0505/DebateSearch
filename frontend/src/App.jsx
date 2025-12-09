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

function sourceLabelText(value) {
  const normalized = normalizeSource(value);
  if (normalized === "reddit") return "Reddit";
  if (normalized === "semeval") return "Twitter/SemEval";
  if (!value) return "Mixed sources";
  return String(value).replace(/^[a-z]/, (c) => c.toUpperCase());
}

function formatPercent(value, digits = 0) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
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

function Metric({
  label,
  value,
  description,
  hint,
  bar,
  delta,
  stats = [],
  active = false,
  onSelect,
  tooltip,
}) {
  const isInteractive = typeof onSelect === "function";
  const className = [
    "metric",
    active ? "active" : "",
    isInteractive ? "clickable" : "",
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <button
      type="button"
      className={className}
      onClick={onSelect}
      aria-pressed={active}
      title={tooltip}
    >
      <div className="metric-header">
        <div className="metric-label">{label}</div>
        {delta && <span className="metric-delta">{delta}</span>}
      </div>
      <div className="metric-value">{value}</div>
      {description && <div className="metric-desc">{description}</div>}
      {hint && <div className="metric-hint">{hint}</div>}
      {bar && (
        <div className="sparkbar" role="presentation">
          {bar.map((seg) => {
            const pct = Math.max(0, Math.min(100, (seg.percent || 0) * 100));
            return (
              <div
                key={seg.key}
                className={`seg ${seg.className || ""}`}
                style={{ width: `${pct}%` }}
                title={seg.label}
              />
            );
          })}
        </div>
      )}
      {stats.length > 0 && (
        <div className="metric-stats">
          {stats.map((stat) => (
            <div key={`${label}-${stat.label}`} className="metric-stat">
              <span className="metric-stat-label">{stat.label}</span>
              <span className="metric-stat-value">{stat.value}</span>
            </div>
          ))}
        </div>
      )}
    </button>
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

  const textClass = ["card-text"];
  if (isLong) {
    textClass.push(expanded ? "expanded" : "clamped");
  }

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

      <div className={textClass.join(" ")}>
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
  const [stanceFilter, setStanceFilter] = useState(null); // "favor" | "against" | "none" | null
  const qref = useRef(null);

  const counts = useMemo(() => {
    const map = { favor: 0, against: 0, none: 0 };
    (clusters || []).forEach((c) => (map[c.stance] = c.items?.length || 0));
    return map;
  }, [clusters]);

  const stanceBuckets = useMemo(() => {
    const map = {};
    (clusters || []).forEach((cluster) => {
      if (stanceOrder.includes(cluster.stance)) {
        map[cluster.stance] = cluster;
      }
    });
    return map;
  }, [clusters]);

  const datasetMix = useMemo(() => {
    const mix = { reddit: 0, semeval: 0, other: 0 };
    (clusters || []).forEach((cluster) => {
      (cluster.items || []).forEach((item) => {
        const src = normalizeSource(item.source);
        if (src === "reddit") mix.reddit += 1;
        else if (src === "semeval") mix.semeval += 1;
        else mix.other += 1;
      });
    });
    const total = mix.reddit + mix.semeval + mix.other || 1;
    return {
      counts: mix,
      percent: {
        reddit: mix.reddit / total,
        semeval: mix.semeval / total,
        other: mix.other / total,
      },
    };
  }, [clusters]);

  const stanceMeta = useMemo(() => {
    const meta = {};
    stanceOrder.forEach((stance) => {
      const bucket = stanceBuckets[stance];
      const items = bucket?.items || [];
      if (!items.length) {
        meta[stance] = { avgConf: null, topSource: "No posts", avgLength: null };
        return;
      }
      let confSum = 0;
      let confCount = 0;
      const sourceCounts = {};
      let lengthSum = 0;
      items.forEach((item) => {
        if (typeof item.conf === "number") {
          confSum += item.conf;
          confCount += 1;
        }
        const src = normalizeSource(item.source) || "other";
        sourceCounts[src] = (sourceCounts[src] || 0) + 1;
        const text = (item.text || item.body || "").trim();
        lengthSum += text.length;
      });
      const avgConf = confCount ? confSum / confCount : null;
      const topSourceKey = Object.entries(sourceCounts)
        .sort((a, b) => b[1] - a[1])[0]?.[0];
      meta[stance] = {
        avgConf,
        topSource: topSourceKey ? sourceLabelText(topSourceKey) : "Mixed sources",
        avgLength: lengthSum ? Math.round(lengthSum / items.length) : null,
      };
    });
    return meta;
  }, [stanceBuckets]);

  const dist = useMemo(() => {
    const total = counts.favor + counts.against + counts.none || 1;
    return {
      favor: counts.favor / total,
      against: counts.against / total,
      none: counts.none / total,
    };
  }, [counts]);
  const totalPosts = counts.favor + counts.against + counts.none;
  const stanceStats = stanceOrder.map((stance) => ({
    stance,
    label: stanceLabel[stance],
    count: counts[stance],
    share: dist[stance],
    avgConf: stanceMeta[stance]?.avgConf ?? null,
    avgLength: stanceMeta[stance]?.avgLength ?? null,
    topSource: stanceMeta[stance]?.topSource ?? "Mixed sources",
  }));
  const stanceStatsMap = useMemo(() => {
    const map = {};
    stanceStats.forEach((entry) => {
      map[entry.stance] = entry;
    });
    return map;
  }, [stanceStats]);
  const dominance = (() => {
    const sorted = [...stanceStats].sort((a, b) => b.count - a.count);
    const leader = sorted[0];
    const runnerUp = sorted[1];
    if (!leader || !runnerUp || leader.count === runnerUp.count) return { stance: null, message: null };
    return {
      stance: leader.stance,
      message: `+${leader.count - runnerUp.count} vs ${runnerUp.label}`,
    };
  })();

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
  const visibleStances = stanceFilter ? [stanceFilter] : stanceOrder;
  const savedIds = useMemo(() => new Set(savedItems.map((it) => it.id)), [savedItems]);

  function handleFocusStance(next) {
    setStanceFilter((prev) => (prev === next ? null : next));
  }

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
            <Metric
              label="Total posts"
              value={totalPosts}
              description="Across all stances"
              hint={`${counts.favor} supporting · ${counts.against} opposing · ${counts.none} neutral`}
              bar={stanceOrder.map((stance) => ({
                key: `total-${stance}`,
                percent: dist[stance],
                className: stanceColorClass[stance],
                label: `${stanceLabel[stance]} ${(dist[stance] * 100).toFixed(0)}%`,
              }))}
              stats={[
                { label: "Reddit", value: formatPercent(datasetMix.percent.reddit) },
                { label: "Twitter/SemEval", value: formatPercent(datasetMix.percent.semeval) },
                { label: "Other", value: formatPercent(datasetMix.percent.other) },
              ]}
              active={!stanceFilter}
              onSelect={() => handleFocusStance(null)}
              tooltip="Click to reset stance focus"
            />
            {stanceOrder.map((stance) => (
              <Metric
                key={`metric-${stance}`}
                label={stanceLabel[stance]}
                value={counts[stance]}
                description="Share of debate"
                hint={`${(dist[stance] * 100).toFixed(0)}% of ${totalPosts || 0} posts`}
                delta={dominance.stance === stance ? dominance.message : null}
                stats={[
                  {
                    label: "Avg conf",
                    value: stanceStatsMap[stance]?.avgConf != null ? formatPercent(stanceStatsMap[stance].avgConf, 0) : "—",
                  },
                  { label: "Top source", value: stanceStatsMap[stance]?.topSource || "Mixed" },
                ]}
                active={stanceFilter === stance}
                onSelect={() => handleFocusStance(stance)}
                tooltip="Toggle focus on this stance"
              />
            ))}
          </div>
          <div className="dist-card card">
            <div className="dist-header">
              <div>
                <h3>Stance momentum</h3>
                <span className="muted small">Share & average confidence</span>
              </div>
              <span className="muted small">{totalPosts} posts</span>
            </div>
            <div className="dist-mini-grid">
              {stanceStats.map((entry) => (
                <div key={`mini-${entry.stance}`} className="dist-mini">
                  <div className="dist-mini-top">
                    <span>{entry.label}</span>
                    <strong>{formatPercent(entry.share)}</strong>
                  </div>
                  <div className="dist-mini-bar" role="presentation" title={`${entry.label} share`}>
                    <div className={`seg ${stanceColorClass[entry.stance]}`} style={{ width: `${entry.share * 100}%` }} />
                  </div>
                  <div className="dist-mini-meta">
                    <span>{entry.count} posts</span>
                    <span>{entry.avgConf != null ? `Avg conf ${formatPercent(entry.avgConf, 0)}` : "No confidence data"}</span>
                  </div>
                </div>
              ))}
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
                <h3>{stanceFilter ? `Focus: ${stanceLabel[stanceFilter]}` : "Scan every stance side-by-side"}</h3>
              </div>
              <span className="muted small">
                {stanceFilter ? "Click Total to clear focus · " : ""}
                Sorted by {sortLabel}
              </span>
            </div>
            <div className="grid3">
              {visibleStances.map((s) => (
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
            {(stanceFilter ? [stanceFilter] : ["favor", "against"]).map((stance) => (
              <div key={`compare-${stance}`} className={`stance-column stance-${stance}`}>
                <div className={`panel-h ${stanceColorClass[stance]}`}>
                  <strong>{stanceLabel[stance]}</strong>{" "}
                  <Chip>{(rendered[stance] || []).length} posts</Chip>
                </div>
                {(rendered[stance] || []).map((it) => (
                  <ResultCard
                    key={it.id}
                    item={it}
                    stance={stance}
                    saved={savedIds.has(it.id)}
                    onToggleSave={handleToggleSave}
                  />
                ))}
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
