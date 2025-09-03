"use client";

import { useMemo, useState } from "react";
import Papa from "papaparse";
import * as ss from "simple-statistics";
import dynamic from "next/dynamic";
import ClientOnly from "./components/ClientOnly";
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Title,
} from "chart.js";
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Title);
const Scatter = dynamic(() => import("react-chartjs-2").then(m => m.Scatter), { ssr: false });

/* ---------- tiny dependency-free k-means ---------- */
function kmeansSimple(points, k, { maxIters = 100, seed = 42 } = {}) {
  if (points.length < k) return null;
  let s = seed >>> 0;
  const rand = () => (s = (1664525 * s + 1013904223) >>> 0) / 2**32;
  const centroids = []; const used = new Set();
  while (centroids.length < k) { const i = Math.floor(rand() * points.length); if (!used.has(i)) { used.add(i); centroids.push(points[i].slice()); } }
  let clusters = new Array(points.length).fill(0);
  const d2 = (a,b)=>a.reduce((acc,ai,i)=>acc+(ai-b[i])**2,0);
  for (let it=0; it<maxIters; it++){
    let changed=false;
    for (let i=0;i<points.length;i++){
      let best=-1,bd=Infinity;
      for (let c=0;c<k;c++){ const d=d2(points[i],centroids[c]); if(d<bd){bd=d; best=c;} }
      if (clusters[i]!==best){clusters[i]=best; changed=true;}
    }
    const sums=Array.from({length:k},()=>Array(points[0].length).fill(0));
    const cnt=new Array(k).fill(0);
    for (let i=0;i<points.length;i++){ const cid=clusters[i]; cnt[cid]++; for(let d=0;d<points[i].length;d++) sums[cid][d]+=points[i][d]; }
    for (let c=0;c<k;c++){ if(!cnt[c]) continue; for(let d=0;d<sums[c].length;d++) centroids[c][d]=sums[c][d]/cnt[c]; }
    if (!changed) break;
  }
  // compute SSE/inertia
  const inertia = points.reduce((acc, p, i) => {
    const c = clusters[i]; const center = centroids[c];
    const sse = p.reduce((a, pv, d) => a + (pv - center[d])**2, 0);
    return acc + sse;
  }, 0);
  return { clusters, centroids, inertia };
}
/* -------------------------------------------------- */

function makeGrid(rows, cols) {
  const header = Array.from({ length: cols }, (_, j) => `col_${j+1}`);
  const grid = [header];
  for (let i=1;i<rows;i++) grid.push(Array.from({length: cols}, () => ""));
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
const columnNames = (grid) => (grid.length ? grid[0] : []);
const download = (filename, text, mime="text/plain") => {
  const blob = new Blob([text], { type: mime }); const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
};

const EXAMPLES = [
  "SaaS subscriptions (monthly): user_id, signup_date, plan {free,pro,business}, monthly_fee, seats, active_flag, churn_date, region",
  "E-commerce orders: order_id, order_date, customer_segment, items_count, subtotal, shipping_cost, discount, tax, total, country",
  "A/B experiment results: user_id, variant {A,B}, session_date, clicks, time_on_site_sec, converted {0,1}, revenue",
  "IoT sensor telemetry (hourly): device_id, timestamp, temp_c, humidity_pct, vibration_mm_s, status {ok,warning,fail}",
  "Education LMS: student_id, course_id, assignment, due_date, submitted_at, grade_pct, late_flag, ai_usage_score",
  "Finance daily prices: date, ticker, open, high, low, close, volume, sector",
];

export default function Home() {
  // Full dataset lives in fullData; we render only first 10 rows in table
  const [fullData, setFullData] = useState(() => makeGrid(8,4));
  const [xCol, setXCol] = useState(0);
  const [yCol, setYCol] = useState(1);
  const [k, setK] = useState(3);
  const [clusterCols, setClusterCols] = useState([0,1]);
  const [aiQuestion, setAiQuestion] = useState("");
  const [aiAnswer, setAiAnswer] = useState("");
  const [busyAsk, setBusyAsk] = useState(false);

  // AI dataset generation UI state
  const [genPrompt, setGenPrompt] = useState(EXAMPLES[0]);
  const [genRows, setGenRows] = useState(1000);
  const [busyGen, setBusyGen] = useState(false);
  const [genError, setGenError] = useState("");

  // VIEW: first 10 rows only (plus header)
  const viewData = useMemo(() => {
    const take = Math.min(fullData.length, 11); // header + 10 rows
    return fullData.slice(0, take);
  }, [fullData]);

  const names = columnNames(fullData);
  const totalRows = Math.max(fullData.length - 1, 0);

  const summary = useMemo(() => names.map((name, j) => {
    const vals = parseNumericColumn(fullData, j);
    if (!vals.length) return { name, count: 0 };
    return { name, count: vals.length, mean: ss.mean(vals), median: ss.median(vals), stdev: ss.standardDeviation(vals), min: ss.min(vals), max: ss.max(vals) };
  }), [fullData]);

  // ---- Linear Regression (with idiot-safe outputs) ----
  const regression = useMemo(() => {
    const aligned = [];
    const xs = [], ys = [];
    for (let i = 1; i < fullData.length; i++) {
      const xv = Number(fullData[i][xCol]);
      const yv = Number(fullData[i][yCol]);
      if (Number.isFinite(xv) && Number.isFinite(yv)) { aligned.push([xv, yv]); xs.push(xv); ys.push(yv); }
    }
    if (aligned.length < 2) return null;
    const lr = ss.linearRegression(aligned); // { m, b }
    const line = ss.linearRegressionLine(lr);
    const r2 = ss.rSquared(aligned, line);
    const r = ss.sampleCorrelation(xs, ys); // Pearson r
    return { slope: lr.m, intercept: lr.b, r, r2, line, points: aligned };
  }, [fullData, xCol, yCol]);

  // ---- K-means ----
  const clusters = useMemo(() => {
    const cols = clusterCols.filter(c => c >= 0 && c < names.length);
    if (!cols.length || fullData.length <= 1) return null;
    const matrix = [];
    const rowIndexMap = []; // to map back for centroids table if needed
    for (let i = 1; i < fullData.length; i++) {
      const row = [];
      let ok = true;
      for (const c of cols) {
        const v = Number(fullData[i][c]);
        if (!Number.isFinite(v)) { ok = false; break; }
        row.push(v);
      }
      if (ok) { matrix.push(row); rowIndexMap.push(i); }
    }
    if (matrix.length < k) return null;
    try {
      const out = kmeansSimple(matrix, k, { seed: 42, maxIters: 100 });
      if (!out) return null;
      // cluster sizes
      const sizes = new Array(k).fill(0);
      out.clusters.forEach(cid => sizes[cid]++);
      return { cols, clusters: out.clusters, centroids: out.centroids, inertia: out.inertia, sizes, rowIndexMap };
    } catch { return null; }
  }, [fullData, clusterCols, k, names.length]);

  // Charts — brighter lines/points + readable grid on dark theme
  const axisColor = "rgba(255,255,255,0.7)";
  const gridColor = "rgba(255,255,255,0.10)";
  const chartOptions = {
    responsive:true, maintainAspectRatio:false,
    scales:{ x:{ type:"linear", position:"bottom", ticks:{ color:axisColor }, grid:{ color:gridColor } },
            y:{ type:"linear", ticks:{ color:axisColor }, grid:{ color:gridColor } } },
    plugins:{
      legend:{ labels:{ color:axisColor } },
      title:{ color:axisColor }
    },
    elements:{ point:{ radius:3, hitRadius:6 }, line:{ borderWidth:2 } }
  };

  function setCell(i, j, v) {
    setFullData(prev => { const next = prev.map(r => r.slice()); next[i][j] = v; return next; });
  }
  function addRow() { setFullData(prev => [...prev, Array.from({ length: prev[0].length }, () => "")]); }
  function delRow() { setFullData(prev => prev.length > 2 ? prev.slice(0, -1) : prev); }
  function addCol() { setFullData(prev => prev.map((row, i) => [...row, i === 0 ? `col_${row.length + 1}` : ""])); }
  function delCol() { setFullData(prev => prev[0].length > 1 ? prev.map(r => r.slice(0, -1)) : prev); }
  function newGrid() { setFullData(makeGrid(8, 4)); }

  function importCSV(file) {
    Papa.parse(file, {
      complete: (res) => {
        const rows = res.data.filter(r => r.length && r.some(x => String(x).trim() !== ""));
        if (!rows.length) return;
        const maxLen = Math.max(...rows.map(r => r.length));
        const grid = rows.map((r, i) => Array.from({ length: maxLen }, (_, j) => (r[j] ?? "").toString()));
        grid[0] = grid[0].map(h => String(h || "").trim() || "col");
        setFullData(grid);
      }
    });
  }
  function exportCSV() {
    const csv = Papa.unparse(fullData);
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
      lines.push(`- equation: y = ${regression.slope.toFixed(6)} * x + ${regression.intercept.toFixed(6)}`);
      lines.push(`- r (Pearson) = ${regression.r.toFixed(6)}  R^2 = ${regression.r2.toFixed(6)}`);
    }
    if (clusters) {
      lines.push("", "## K-means");
      lines.push(`- features=${clusters.cols.map(c => names[c]).join(", ")}  k=${k}`);
      lines.push(`- inertia (SSE) = ${clusters.inertia.toFixed(4)}`);
      lines.push(`- sizes = ${clusters.sizes.join(", ")}`);
      // centroids preview for first two features to keep it readable
      const f1 = names[clusters.cols[0]], f2 = clusters.cols[1] != null ? names[clusters.cols[1]] : null;
      if (f2) {
        lines.push("- centroids (first two features):");
        clusters.centroids.forEach((cen, idx)=> lines.push(`  - C${idx+1}: ${f1}=${cen[0].toFixed(4)}, ${f2}=${cen[1].toFixed(4)}`));
      }
    }
    download("analysis_report.md", lines.join("\n"));
  }

  async function askAI() {
    setBusyAsk(true); setAiAnswer("");
    try {
      const schema = names.map((n, i) => `${i}:${n}`).join(", ");
      const numericBrief = summary.filter(s => s.count).map(s => `${s.name}{n:${s.count},mean:${s.mean.toFixed(3)},sd:${s.stdev.toFixed(3)}}`).join("; ");
      const regBrief = regression ? `regression X=${names[xCol]} Y=${names[yCol]} slope=${regression.slope.toFixed(4)} intercept=${regression.intercept.toFixed(4)} r=${regression.r.toFixed(4)} r2=${regression.r2.toFixed(4)} eq:y=${regression.slope.toFixed(4)}*x+${regression.intercept.toFixed(4)}` : "no regression";
      const cluBrief = clusters ? `kmeans k=${k} features=${clusters.cols.map(c=>names[c]).join(",")} inertia=${clusters.inertia.toFixed(2)} sizes=${clusters.sizes.join("/")}` : "no clustering";
      const csvPreview = Papa.unparse(fullData.slice(0, Math.min(20, fullData.length)));
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

Answer clearly with concrete, beginner-safe steps and short formulas when relevant.
`.trim();
      const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt }) });
      const json = await res.json();
      setAiAnswer(json.text || json.error || "no response");
    } catch (e) { setAiAnswer(String(e)); }
    finally { setBusyAsk(false); }
  }

  // ---- Render ----
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-white">
      <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        {/* AI-centered generate */}
        <section className="rounded-2xl bg-slate-900/70 ring-1 ring-white/10 p-5">
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight mb-2">DATAROOM — AI dataset lab</h1>
          <p className="text-sm opacity-80 mb-4">Describe the data you want. We’ll generate thousands of rows, but only show the first 10 here.</p>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="md:col-span-2 space-y-2">
              <textarea
                className="w-full h-28 rounded-xl bg-slate-800/80 p-3 outline-none"
                placeholder="Describe the dataset you want…"
                value={genPrompt}
                onChange={(e)=>setGenPrompt(e.target.value)}
              />
              <div className="flex flex-wrap items-center gap-3">
                <label className="text-sm opacity-80">target rows</label>
                <input type="number" min="100" max="5000" step="100"
                       className="w-28 px-2 py-1 rounded bg-slate-800/80"
                       value={genRows} onChange={(e)=>setGenRows(Number(e.target.value)||1000)} />
                <button onClick={async ()=>{
                  setBusyGen(true); setGenError("");
                  try {
                    const res = await fetch("/api/generate", { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify({ prompt: genPrompt, rows: genRows }) });
                    if (!res.ok) throw new Error(`generate failed: ${res.status}`);
                    const csvText = await res.text();
                    const parsed = Papa.parse(csvText.trim());
                    if (!parsed.data?.length) throw new Error("empty CSV");
                    const maxLen = Math.max(...parsed.data.map(r => r.length));
                    const grid = parsed.data.map((r, i) => Array.from({ length: maxLen }, (_, j) => (r[j] ?? "").toString()));
                    grid[0] = grid[0].map(h => String(h || "").trim() || "col");
                    setFullData(grid);
                  } catch (e) { setGenError(String(e)); } finally { setBusyGen(false); }
                }} disabled={busyGen}
                        className="px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50">
                  {busyGen ? "generating…" : "generate dataset"}
                </button>
                <button onClick={exportCSV} className="px-3 py-2 rounded-xl bg-slate-700 hover:bg-slate-600">export csv</button>
              </div>
              {genError && <div className="text-xs text-red-300">{genError}</div>}
            </div>
            <div className="space-y-2">
              <div className="text-sm font-medium opacity-90">example prompts</div>
              <ul className="text-xs space-y-1">
                {EXAMPLES.map((ex, i)=>(
                  <li key={i}>
                    <button
                      className="text-left w-full px-2 py-1 rounded bg-slate-800/70 hover:bg-slate-800/90"
                      onClick={()=>setGenPrompt(ex)}
                    >
                      {ex}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        {/* Dataset preview (first 10 rows) */}
        <section className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">dataset preview</h2>
            <div className="text-xs opacity-70">showing first 10 of {totalRows} rows • analytics run on full dataset • export for all rows</div>
          </div>
          <div className="overflow-auto border border-white/10 rounded-xl">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-800/70 sticky top-0">
                <tr>
                  {viewData[0].map((h, j) => (
                    <th key={j} className="px-3 py-2 border-b border-white/10">
                      <input className="w-40 bg-transparent outline-none" value={h} onChange={(e) => setCell(0, j, e.target.value)} />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {viewData.slice(1).map((row, i) => (
                  <tr key={i} className="odd:bg-slate-900/40">
                    {row.map((cell, j) => (
                      <td key={j} className="px-3 py-1 border-b border-white/5">
                        <input className="w-40 bg-transparent outline-none" value={cell} onChange={(e) => setCell(i + 1, j, e.target.value)} placeholder="..." />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Quick summary cards */}
        <section className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {summary.filter(s=>s.count).slice(0,4).map((s,idx)=>(
            <div key={idx} className="rounded-xl bg-slate-900/70 ring-1 ring-white/10 p-4">
              <div className="text-xs opacity-70">{s.name}</div>
              <div className="mt-1 text-lg font-semibold">μ {s.mean.toFixed(2)} · σ {s.stdev.toFixed(2)}</div>
              <div className="text-xs opacity-70">n={s.count} · min {s.min} · max {s.max}</div>
            </div>
          ))}
        </section>

        {/* Analytics */}
        <section className="grid lg:grid-cols-2 gap-6">
          {/* Regression */}
          <div className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-5">
            <h2 className="font-semibold mb-3">linear regression</h2>
            <div className="flex gap-2 mb-3">
              <select value={xCol} onChange={(e)=>setXCol(Number(e.target.value))} className="px-2 py-1 rounded bg-slate-800/80">
                {names.map((n,i)=>(<option key={i} value={i}>X: {n}</option>))}
              </select>
              <select value={yCol} onChange={(e)=>setYCol(Number(e.target.value))} className="px-2 py-1 rounded bg-slate-800/80">
                {names.map((n,i)=>(<option key={i} value={i}>Y: {n}</option>))}
              </select>
            </div>

            {regression ? (
              <>
                {/* Friendly stat cards */}
                <div className="grid sm:grid-cols-2 gap-3 mb-3">
                  <div className="rounded-xl bg-slate-800/60 p-3">
                    <div className="text-xs opacity-70">equation</div>
                    <div className="text-sm font-semibold">y = {regression.slope.toFixed(6)} · x + {regression.intercept.toFixed(6)}</div>
                  </div>
                  <div className="rounded-xl bg-slate-800/60 p-3">
                    <div className="text-xs opacity-70">fit</div>
                    <div className="text-sm font-semibold">r = {regression.r.toFixed(4)} · R² = {regression.r2.toFixed(4)}</div>
                  </div>
                </div>

                <div className="bg-slate-800/40 rounded-xl p-2 h-64">
                  <ClientOnly>
                    <Scatter
                      data={{
                        datasets: [
                          {
                            label:"data",
                            data: regression.points.map(([x,y])=>({x,y})),
                            showLine:false,
                            borderColor:"rgba(99,102,241,0.9)", // indigo-500
                            backgroundColor:"rgba(99,102,241,0.9)"
                          },
                          {
                            label:"fit",
                            data:(()=>{
                              const xs = regression.points.map(p=>p[0]);
                              const minX = Math.min(...xs), maxX = Math.max(...xs);
                              return [{x:minX, y:regression.slope*minX+regression.intercept},
                                      {x:maxX, y:regression.slope*maxX+regression.intercept}];
                            })(),
                            showLine:true,
                            borderColor:"rgba(236,72,153,0.9)", // pink-500
                            backgroundColor:"rgba(236,72,153,0.9)"
                          }
                        ]
                      }}
                      options={chartOptions}
                    />
                  </ClientOnly>
                </div>
              </>
            ) : (
              <div className="text-sm opacity-80">
                Pick numeric columns for X and Y. You need at least 2 rows with numbers in both columns.
              </div>
            )}
          </div>

          {/* K-means */}
          <div className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-5">
            <h2 className="font-semibold mb-3">k-means clustering</h2>
            <div className="flex flex-wrap gap-2 mb-3">
              <div className="flex items-center gap-1">
                <span className="text-sm opacity-80">k</span>
                <input type="number" min="2" max="10" value={k} onChange={(e)=>setK(Number(e.target.value)||2)} className="w-16 px-2 py-1 rounded bg-slate-800/80" />
              </div>
              <div className="flex flex-wrap gap-1">
                {names.map((n,i)=>(
                  <label key={i} className="text-xs px-2 py-1 rounded bg-slate-800/80 cursor-pointer select-none">
                    <input type="checkbox" checked={clusterCols.includes(i)}
                      onChange={(e)=>{ if(e.target.checked) setClusterCols(p=>[...new Set([...p,i])]); else setClusterCols(p=>p.filter(x=>x!==i)); }}
                      className="mr-1 align-middle" />
                    {n}
                  </label>
                ))}
              </div>
            </div>

            {clusters && clusters.cols.length>=2 ? (
              <>
                {/* Friendly stat cards */}
                <div className="grid sm:grid-cols-3 gap-3 mb-3">
                  <div className="rounded-xl bg-slate-800/60 p-3">
                    <div className="text-xs opacity-70">features</div>
                    <div className="text-sm font-semibold">{clusters.cols.map(c=>names[c]).join(", ")}</div>
                  </div>
                  <div className="rounded-xl bg-slate-800/60 p-3">
                    <div className="text-xs opacity-70">sizes</div>
                    <div className="text-sm font-semibold">{clusters.sizes.join(" · ")}</div>
                  </div>
                  <div className="rounded-xl bg-slate-800/60 p-3">
                    <div className="text-xs opacity-70">inertia (SSE)</div>
                    <div className="text-sm font-semibold">{clusters.inertia.toFixed(2)}</div>
                  </div>
                </div>

                {/* Centroids (first two features) */}
                {clusters.cols.length >= 2 && (
                  <div className="text-xs opacity-90 mb-3">
                    <div className="mb-1">centroids (for first two selected features):</div>
                    <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
                      {clusters.centroids.map((cen, idx)=>(
                        <div key={idx} className="rounded-lg bg-slate-800/60 p-2">
                          <div className="opacity-70">cluster {idx+1}</div>
                          <div>{names[clusters.cols[0]]}: <span className="font-semibold">{cen[0].toFixed(3)}</span></div>
                          <div>{names[clusters.cols[1]]}: <span className="font-semibold">{cen[1].toFixed(3)}</span></div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="bg-slate-800/40 rounded-xl p-2 h-64">
                  <ClientOnly>
                    <Scatter
                      data={(function(){
                        const c1=clusters.cols[0], c2=clusters.cols[1];
                        const pts=[]; for(let i=1;i<fullData.length;i++){ const x=Number(fullData[i][c1]); const y=Number(fullData[i][c2]); if(Number.isFinite(x)&&Number.isFinite(y)) pts.push({x,y}); }
                        const grouped={}; clusters.clusters.forEach((cid, i)=>{ (grouped[cid] ||= []).push(pts[i]); });
                        return {
                          datasets: Object.keys(grouped).map(cid=>({
                            label:`cluster ${Number(cid)+1}`,
                            data: grouped[cid],
                            showLine:false,
                            borderColor:"rgba(34,197,94,0.9)", // green-500-ish (all groups share color; simplicity)
                            backgroundColor:"rgba(34,197,94,0.9)"
                          }))
                        };
                      })()}
                      options={chartOptions}
                    />
                  </ClientOnly>
                </div>
              </>
            ) : (
              <div className="text-sm opacity-80">
                Select at least two numeric features and ensure rows contain numbers. Increase k if clusters look mixed.
              </div>
            )}
          </div>
        </section>

        {/* Ask the analysis */}
        <section className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-5">
          <h2 className="font-semibold mb-3">ask the analysis</h2>
          <textarea className="w-full h-24 rounded-xl bg-slate-800/80 p-2 outline-none"
                    placeholder="e.g., which features correlate with revenue? segment drivers of churn? is B statistically better than A?"
                    value={aiQuestion} onChange={(e)=>setAiQuestion(e.target.value)} />
          <button onClick={askAI} disabled={busyAsk || !aiQuestion.trim()} className="mt-2 px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50">
            {busyAsk ? "thinking…" : "ask"}
          </button>
          {aiAnswer && (<pre className="mt-3 whitespace-pre-wrap text-xs bg-slate-800/60 p-2 rounded-xl">{aiAnswer.trim()}</pre>)}
        </section>

      </div>
    </main>
  );
}
