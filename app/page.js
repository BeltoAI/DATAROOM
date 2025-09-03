"use client";

import { useMemo, useState } from "react";
import Papa from "papaparse";
import * as ss from "simple-statistics";
import dynamic from "next/dynamic";
import ClientOnly from "./components/ClientOnly";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Title);

// Client-only charts
const Scatter = dynamic(() => import("react-chartjs-2").then(m => m.Scatter), { ssr: false });

/* ---------- tiny, dependency-free k-means (Euclidean) ---------- */
function kmeansSimple(points, k, { maxIters = 100, seed = 42 } = {}) {
  if (points.length < k) return null;
  // seeded pseudo-RNG for deterministic init
  let s = seed >>> 0;
  const rand = () => (s = (1664525 * s + 1013904223) >>> 0) / 2**32;

  // init: pick k random distinct points
  const centroids = [];
  const used = new Set();
  while (centroids.length < k) {
    const idx = Math.floor(rand() * points.length);
    if (!used.has(idx)) { used.add(idx); centroids.push(points[idx].slice()); }
  }

  let clusters = new Array(points.length).fill(0);
  const dist2 = (a,b)=>a.reduce((acc,ai,i)=>acc+(ai-b[i])**2,0);

  for (let iter=0; iter<maxIters; iter++) {
    // assign
    let changed = false;
    for (let i=0;i<points.length;i++){
      let best=-1, bd=Infinity;
      for (let c=0;c<k;c++){
        const d=dist2(points[i], centroids[c]);
        if (d<bd){bd=d; best=c;}
      }
      if (clusters[i]!==best){clusters[i]=best; changed=true;}
    }
    // recompute
    const sums = Array.from({length:k}, ()=>Array(points[0].length).fill(0));
    const counts = new Array(k).fill(0);
    for (let i=0;i<points.length;i++){
      const cid = clusters[i];
      counts[cid]++;
      for (let d=0; d<points[i].length; d++) sums[cid][d]+=points[i][d];
    }
    for (let c=0;c<k;c++){
      if (counts[c]===0) continue; // keep old centroid if empty
      for (let d=0;d<sums[c].length;d++) centroids[c][d]=sums[c][d]/counts[c];
    }
    if (!changed) break;
  }
  return { clusters, centroids };
}
/* -------------------------------------------------------------- */

function makeGrid(rows, cols) {
  const grid = [];
  const header = Array.from({ length: cols }, (_, j) => `col_${j + 1}`);
  grid.push(header);
  for (let i = 1; i < rows; i++) grid.push(Array.from({ length: cols }, () => ""));
  return grid;
}
function parseNumericColumn(grid, colIdx) {
  const vals = [];
  for (let i = 1; i < grid.length; i++) {
    const v = Number(grid[i][colIdx]);
    if (Number.isFinite(v)) vals.push(v);
  }
  return vals;
}
function columnNames(grid) { return grid.length ? grid[0] : []; }

function download(filename, text, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

export default function Home() {
  const [data, setData] = useState(() => makeGrid(8, 4));
  const [xCol, setXCol] = useState(0);
  const [yCol, setYCol] = useState(1);
  const [k, setK] = useState(3);
  const [clusterCols, setClusterCols] = useState([0, 1]);
  const [aiQuestion, setAiQuestion] = useState("");
  const [aiAnswer, setAiAnswer] = useState("");
  const [busy, setBusy] = useState(false);

  const names = columnNames(data);

  const summary = useMemo(() => names.map((name, j) => {
    const vals = parseNumericColumn(data, j);
    if (!vals.length) return { name, count: 0 };
    return {
      name,
      count: vals.length,
      mean: ss.mean(vals),
      median: ss.median(vals),
      stdev: ss.standardDeviation(vals),
      min: ss.min(vals),
      max: ss.max(vals),
    };
  }), [data]);

  const regression = useMemo(() => {
    const aligned = [];
    for (let i = 1; i < data.length; i++) {
      const xv = Number(data[i][xCol]);
      const yv = Number(data[i][yCol]);
      if (Number.isFinite(xv) && Number.isFinite(yv)) aligned.push([xv, yv]);
    }
    if (aligned.length < 2) return null;
    const lr = ss.linearRegression(aligned);
    const r2 = ss.rSquared(aligned, ss.linearRegressionLine(lr));
    return { slope: lr.m, intercept: lr.b, r2, points: aligned };
  }, [data, xCol, yCol]);

  const clusters = useMemo(() => {
    const cols = clusterCols.filter(c => c >= 0 && c < names.length);
    if (!cols.length || data.length <= 1) return null;
    const matrix = [];
    for (let i = 1; i < data.length; i++) {
      const row = [];
      let ok = true;
      for (const c of cols) {
        const v = Number(data[i][c]);
        if (!Number.isFinite(v)) { ok = false; break; }
        row.push(v);
      }
      if (ok) matrix.push(row);
    }
    if (matrix.length < k) return null;
    try {
      const out = kmeansSimple(matrix, k, { seed: 42, maxIters: 100 });
      if (!out) return null;
      return { cols, clusters: out.clusters, centroids: out.centroids };
    } catch {
      return null;
    }
  }, [data, clusterCols, k, names.length]);

  const scatterData = useMemo(() => {
    if (!regression) return null;
    const xs = regression.points.map(p => p[0]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const yfit = (x) => regression.slope * x + regression.intercept;
    return {
      datasets: [
        { label: "data", data: regression.points.map(([x, y]) => ({ x, y })), showLine: false },
        { label: "fit", data: [{ x: minX, y: yfit(minX) }, { x: maxX, y: yfit(maxX) }], showLine: true }
      ],
    };
  }, [regression]);

  const clusterScatter = useMemo(() => {
    if (!clusters || clusters.cols.length < 2) return null;
    const c1 = clusters.cols[0], c2 = clusters.cols[1];
    const pts = [];
    for (let i = 1; i < data.length; i++) {
      const x = Number(data[i][c1]);
      const y = Number(data[i][c2]);
      if (Number.isFinite(x) && Number.isFinite(y)) pts.push({ x, y });
    }
    if (!clusters.clusters || pts.length !== clusters.clusters.length) return null;
    const grouped = {};
    clusters.clusters.forEach((cid, i) => { (grouped[cid] ||= []).push(pts[i]); });
    return {
      datasets: Object.keys(grouped).map(cid => ({
        label: `cluster ${Number(cid) + 1}`,
        data: grouped[cid],
        showLine: false,
      })),
    };
  }, [clusters, data]);

  function setCell(i, j, v) {
    setData(prev => { const next = prev.map(r => r.slice()); next[i][j] = v; return next; });
  }
  function addRow() { setData(prev => [...prev, Array.from({ length: prev[0].length }, () => "")]); }
  function delRow() { setData(prev => prev.length > 2 ? prev.slice(0, -1) : prev); }
  function addCol() { setData(prev => prev.map((row, i) => [...row, i === 0 ? `col_${row.length + 1}` : ""])); }
  function delCol() { setData(prev => prev[0].length > 1 ? prev.map(r => r.slice(0, -1)) : prev); }
  function newGrid() { setData(makeGrid(8, 4)); }

  function importCSV(file) {
    Papa.parse(file, {
      complete: (res) => {
        const rows = res.data.filter(r => r.length && r.some(x => String(x).trim() !== ""));
        if (!rows.length) return;
        const maxLen = Math.max(...rows.map(r => r.length));
        const grid = rows.map((r, i) => Array.from({ length: maxLen }, (_, j) => (r[j] ?? "").toString()));
        grid[0] = grid[0].map(h => String(h || "").trim() || "col");
        setData(grid);
      }
    });
  }
  function exportCSV() {
    const csv = Papa.unparse(data);
    download("dataset.csv", csv, "text/csv");
  }
  function exportReport() {
    const lines = [];
    lines.push("# Analysis Report", "", "## Schema");
    lines.push(names.map((n, i) => `- ${i}: ${n}`).join("\n"));
    lines.push("", "## Summary (numeric)");
    summary.forEach(s => { if (s.count) lines.push(`- ${s.name}: n=${s.count}, mean=${s.mean.toFixed(4)}, med=${s.median.toFixed(4)}, sd=${s.stdev.toFixed(4)}, min=${s.min}, max=${s.max}`); });
    if (regression) {
      lines.push("", "## Linear regression");
      lines.push(`- X=${names[xCol]}  Y=${names[yCol]}`);
      lines.push(`- slope=${regression.slope.toFixed(6)}  intercept=${regression.intercept.toFixed(6)}  R^2=${regression.r2.toFixed(6)}`);
    }
    if (clusters) {
      lines.push("", "## K-means");
      lines.push(`- features=${clusters.cols.map(c => names[c]).join(", ")}  k=${k}`);
    }
    download("analysis_report.md", lines.join("\n"));
  }

  async function askAI() {
    setBusy(true); setAiAnswer("");
    try {
      const schema = names.map((n, i) => `${i}:${n}`).join(", ");
      const numericBrief = summary.filter(s => s.count).map(s => `${s.name}{n:${s.count},mean:${s.mean.toFixed(3)},sd:${s.stdev.toFixed(3)}}`).join("; ");
      const regBrief = regression ? `regression X=${names[xCol]} Y=${names[yCol]} slope=${regression.slope.toFixed(4)} intercept=${regression.intercept.toFixed(4)} r2=${regression.r2.toFixed(4)}` : "no regression";
      const cluBrief = clusters ? `kmeans k=${k} features=${clusters.cols.map(c => names[c]).join(",")}` : "no clustering";
      const csvPreview = Papa.unparse(data.slice(0, Math.min(20, data.length)));

      const prompt = `
You are a data analyst. Given:
schema: ${schema}
summary: ${numericBrief}
${regBrief}
${cluBrief}
csv_preview:
${csvPreview}

User question:
${aiQuestion}

Answer clearly and concisely. Provide specific next steps if needed.
`.trim();

      const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt }) });
      const json = await res.json();
      setAiAnswer(json.text || json.error || "no response");
    } catch (e) { setAiAnswer(String(e)); }
    finally { setBusy(false); }
  }

  const chartOptions = { responsive: true, maintainAspectRatio: false, scales: { x: { type: "linear", position: "bottom" }, y: { type: "linear" } } };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-white">
      <div className="max-w-7xl mx-auto px-4 py-6">
        <header className="mb-6 flex items-center justify-between">
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">DATAROOM — build datasets, analyze, ask</h1>
          <div className="flex gap-2">
            <button onClick={newGrid} className="px-3 py-2 rounded-xl bg-slate-700 hover:bg-slate-600">new</button>
            <button onClick={exportCSV} className="px-3 py-2 rounded-xl bg-slate-700 hover:bg-slate-600">export csv</button>
            <button onClick={exportReport} className="px-3 py-2 rounded-xl bg-slate-700 hover:bg-slate-600">export report</button>
          </div>
        </header>

        <section className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-4">
              <div className="flex flex-wrap items-center gap-2 mb-4">
                <label className="px-3 py-2 rounded-xl bg-slate-800/80 cursor-pointer">
                  import csv
                  <input type="file" accept=".csv" className="hidden" onChange={(e) => e.target.files?.[0] && importCSV(e.target.files[0])} />
                </label>
                <button onClick={addRow} className="px-3 py-2 rounded-xl bg-slate-800/80">+ row</button>
                <button onClick={delRow} className="px-3 py-2 rounded-xl bg-slate-800/80">- row</button>
                <button onClick={addCol} className="px-3 py-2 rounded-xl bg-slate-800/80">+ col</button>
                <button onClick={delCol} className="px-3 py-2 rounded-xl bg-slate-800/80">- col</button>
              </div>

              <div className="overflow-auto border border-white/10 rounded-xl">
                <table className="min-w-full text-sm">
                  <thead className="bg-slate-800/70 sticky top-0">
                    <tr>
                      {data[0].map((h, j) => (
                        <th key={j} className="px-3 py-2 border-b border-white/10">
                          <input className="w-32 bg-transparent outline-none" value={h} onChange={(e) => setCell(0, j, e.target.value)} />
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.slice(1).map((row, i) => (
                      <tr key={i} className="odd:bg-slate-900/40">
                        {row.map((cell, j) => (
                          <td key={j} className="px-3 py-1 border-b border-white/5">
                            <input className="w-32 bg-transparent outline-none" value={cell} onChange={(e) => setCell(i + 1, j, e.target.value)} placeholder="..." />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-4">
              <h2 className="font-semibold mb-3">summary</h2>
              <div className="space-y-2 max-h-64 overflow-auto pr-1">
                {summary.map((s, idx) => (
                  <div key={idx} className="text-xs">
                    <div className="font-medium">{s.name}</div>
                    {s.count ? (
                      <div className="opacity-80">n={s.count} mean={s.mean.toFixed(3)} med={s.median.toFixed(3)} sd={s.stdev.toFixed(3)} min={s.min} max={s.max}</div>
                    ) : <div className="opacity-60">non-numeric/empty</div>}
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-4">
              <h2 className="font-semibold mb-3">ask the analysis</h2>
              <textarea className="w-full h-24 rounded-xl bg-slate-800/80 p-2 outline-none" placeholder="e.g., is Y increasing with X? which cluster is most spread?" value={aiQuestion} onChange={(e) => setAiQuestion(e.target.value)} />
              <button onClick={askAI} disabled={busy || !aiQuestion.trim()} className="mt-2 px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50">
                {busy ? "thinking..." : "ask"}
              </button>
              {aiAnswer && (<pre className="mt-3 whitespace-pre-wrap text-xs bg-slate-800/60 p-2 rounded-xl">{aiAnswer.trim()}</pre>)}
            </div>
          </div>
        </section>

        <section className="mt-6 grid lg:grid-cols-2 gap-6">
          <div className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-4">
            <h2 className="font-semibold mb-3">linear regression</h2>
            <div className="flex gap-2 mb-3">
              <select value={xCol} onChange={(e) => setXCol(Number(e.target.value))} className="px-2 py-1 rounded bg-slate-800/80">
                {names.map((n, i) => (<option key={i} value={i}>X: {n}</option>))}
              </select>
              <select value={yCol} onChange={(e) => setYCol(Number(e.target.value))} className="px-2 py-1 rounded bg-slate-800/80">
                {names.map((n, i) => (<option key={i} value={i}>Y: {n}</option>))}
              </select>
            </div>
            {regression ? (
              <>
                <div className="text-sm opacity-80 mb-3">
                  slope={regression.slope.toFixed(6)} intercept={regression.intercept.toFixed(6)} R^2={regression.r2.toFixed(6)}
                </div>
                <div className="bg-slate-800/40 rounded-xl p-2 h-64">
                  <ClientOnly>
                    {scatterData && <Scatter data={scatterData} options={{ responsive: true, maintainAspectRatio: false, scales: { x: { type: "linear" }, y: { type: "linear" } } }} />}
                  </ClientOnly>
                </div>
              </>
            ) : <div className="text-sm opacity-60">need at least 2 numeric rows for both columns</div>}
          </div>

          <div className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-4">
            <h2 className="font-semibold mb-3">k-means clustering</h2>
            <div className="flex flex-wrap gap-2 mb-3">
              <div className="flex items-center gap-1">
                <span className="text-sm opacity-80">k</span>
                <input type="number" min="2" max="10" value={k} onChange={(e) => setK(Number(e.target.value) || 2)} className="w-16 px-2 py-1 rounded bg-slate-800/80" />
              </div>
              <div className="flex flex-wrap gap-1">
                {names.map((n, i) => (
                  <label key={i} className="text-xs px-2 py-1 rounded bg-slate-800/80 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={clusterCols.includes(i)}
                      onChange={(e) => {
                        if (e.target.checked) setClusterCols(prev => [...new Set([...prev, i])]);
                        else setClusterCols(prev => prev.filter(x => x !== i));
                      }}
                      className="mr-1 align-middle"
                    />
                    {n}
                  </label>
                ))}
              </div>
            </div>
            <div className="bg-slate-800/40 rounded-xl p-2 h-64">
              <ClientOnly>
                {clusterScatter && <Scatter data={clusterScatter} options={{ responsive: true, maintainAspectRatio: false, scales: { x: { type: "linear" }, y: { type: "linear" } } }} />}
              </ClientOnly>
            </div>
          </div>
        </section>

        <footer className="mt-10 text-center text-xs opacity-60">
          data lives in the browser until you export. non-numeric values are ignored in math.
        </footer>
      </div>
    </main>
  );
}
