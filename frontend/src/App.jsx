// App.jsx — minimal edits

import { useState } from "react";
import axios from "axios";
const API_URL = "http://127.0.0.1:8000";

export default function App() {
  const [query, setQuery] = useState("");
  const [nsfwOk, setNsfwOk] = useState(false); // <-- new
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  async function handleSearch(e) {
    e.preventDefault();
    setLoading(true);
    setErr("");
    try {
      const res = await axios.post(`${API_URL}/search_clustered`, {
        query,
        k: 30,
        nsfw_ok: nsfwOk,     // <-- pass the toggle
      });
      setData(res.data);
    } catch (e) {
      console.error(e);
      setErr(e?.response?.data?.detail || "Search failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-semibold">Debate Search Engine</h1>

      <form onSubmit={handleSearch} className="space-y-3">
        <input
          className="w-full border rounded px-3 py-2"
          placeholder="Try: legalization of abortion"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={nsfwOk}
            onChange={(e) => setNsfwOk(e.target.checked)}
          />
          Show NSFW/toxic results
        </label>

        <button className="px-4 py-2 rounded bg-black text-white" disabled={loading}>
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      {err && <div className="text-red-600 text-sm">{err}</div>}

      {data && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {data.clusters.map((c) => (
            <div key={c.stance} className="border rounded p-3">
              <div className="font-semibold mb-2 capitalize">{c.stance}</div>
              <ul className="space-y-2">
                {c.items.slice(0, 10).map((it) => (
                  <li key={it.id} className="text-sm">
                    <div className="opacity-70">{it.text}</div>
                    <div className="text-xs mt-1 flex gap-2 opacity-60">
                      {it.source && <span>src: {it.source}</span>}
                      <span>conf: {it.conf.toFixed(2)}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}