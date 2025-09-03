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
  const [fullData, setFullData] = useState(() => makeGrid(8,4));
  const [xCol, setXCol] = useState(0);
  const [yCol, setYCol] = useState(1);
  const [k, setK] = useState(3);
  const [clusterCols, setClusterCols] = useState([0,1]);
  const [aiQuestion, setAiQuestion] = useState("");
  const [aiAnswer, setAiAnswer] = useState("");
  const [busyAsk, setBusyAsk] = useState(false);

  const [genPrompt, setGenPrompt] = useState(EXAMPLES[0]);
  const [genRows, setGenRows] = useState(1000);
  const [busyGen, setBusyGen] = useState(false);
  const [genError, setGenError] = useState("");

  const [corrTarget, setCorrTarget] = useState(0);        // correlation target column
  const [clusterExplain, setClusterExplain] = useState(""); // LLM explanation

  const viewData = useMemo(() => fullData.slice(0, Math.min(fullData.length, 11)), [fullData]);
  const names = columnNames(fullData);
  const totalRows = Math.max(fullData.length - 1, 0);

  const summary = useMemo(() => names.map((name, j) => {
    const vals = parseNumericColumn(fullData, j);
    if (!vals.length) return { name, count: 0 };
    return { name, count: vals.length, mean: ss.mean(vals), median: ss.median(vals), stdev: ss.standardDeviation(vals), min: ss.min(vals), max: ss.max(vals) };
  }), [fullData]);

  // Regression
  const regression = useMemo(() => {
    const aligned = [], xs=[], ys=[];
    for (let i = 1; i < fullData.length; i++) {
      const xv = Number(fullData[i][xCol]); const yv = Number(fullData[i][yCol]);
      if (Number.isFinite(xv) && Number.isFinite(yv)) { aligned.push([xv, yv]); xs.push(xv); ys.push(yv); }
    }
    if (aligned.length < 2) return null;
    const lr = ss.linearRegression(aligned); const line = ss.linearRegressionLine(lr);
    const r2 = ss.rSquared(aligned, line); const r = ss.sampleCorrelation(xs, ys);
    return { slope: lr.m, intercept: lr.b, r, r2, line, points: aligned };
  }, [fullData, xCol, yCol]);

  // Correlation vs chosen target (top 5)
  const topCorr = useMemo(() => {
    const targetVals = parseNumericColumn(fullData, corrTarget);
    if (targetVals.length < 2) return [];
    const out = [];
    for (let j=0; j<names.length; j++){
      if (j===corrTarget) continue;
      const vals = parseNumericColumn(fullData, j);
      const n = Math.min(targetVals.length, vals.length);
      if (n < 2) continue;
      // align by rows
      const a=[], b=[];
      for (let i=1; i<fullData.length; i++){
        const v1 = Number(fullData[i][corrTarget]);
        const v2 = Number(fullData[i][j]);
        if (Number.isFinite(v1) && Number.isFinite(v2)) { a.push(v1); b.push(v2); }
      }
      if (a.length >= 2) {
        try { out.push({ name: names[j], r: ss.sampleCorrelation(a,b) }); } catch {}
      }
    }
    return out.sort((x,y)=>Math.abs(y.r)-Math.abs(x.r)).slice(0,5);
  }, [fullData, corrTarget, names]);

  // K-means
  const clusters = useMemo(() => {
    const cols = clusterCols.filter(c => c >= 0 && c < names.length);
    if (!cols.length || fullData.length <= 1) return null;
    const matrix = [], rowIndexMap=[];
    for (let i=1;i<fullData.length;i++){
      const row=[]; let ok=true;
      for (const c of cols){ const v=Number(fullData[i][c]); if(!Number.isFinite(v)){ ok=false; break; } row.push(v); }
      if (ok){ matrix.push(row); rowIndexMap.push(i); }
    }
    if (matrix.length < k) return null;
    try{
      const out = kmeansSimple(matrix, k, { seed:42, maxIters:100 });
      if (!out) return null;
      const sizes = new Array(k).fill(0); out.clusters.forEach(cid=>sizes[cid]++);
      return { cols, clusters: out.clusters, centroids: out.centroids, inertia: out.inertia, sizes, rowIndexMap };
    } catch { return null; }
  }, [fullData, clusterCols, k, names.length]);

  // Chart styling for dark UI — brighter
  const axisColor = "rgba(255,255,255,0.85)";
  const gridColor = "rgba(255,255,255,0.12)";
  const chartOptions = {
    responsive:true, maintainAspectRatio:false,
    scales:{ x:{ type:"linear", position:"bottom", ticks:{ color:axisColor }, grid:{ color:gridColor } },
            y:{ type:"linear", ticks:{ color:axisColor }, grid:{ color:gridColor } } },
    plugins:{ legend:{ labels:{ color:axisColor } }, title:{ color:axisColor } },
    elements:{ point:{ radius:3, hitRadius:6 }, line:{ borderWidth:2 } }
  };
  const clusterPalette = [
    "rgba(99,102,241,0.9)",   // indigo
    "rgba(16,185,129,0.9)",   // emerald
    "rgba(234,179,8,0.9)",    // amber
    "rgba(244,63,94,0.9)",    // rose
    "rgba(59,130,246,0.9)",   // blue
    "rgba(217,70,239,0.9)",   // fuchsia
    "rgba(34,197,94,0.9)",    // green
    "rgba(250,204,21,0.9)",   // yellow
    "rgba(168,85,247,0.9)",   // violet
    "rgba(236,72,153,0.9)"    // pink
  ];

  // IO helpers
  function setCell(i, j, v) {
    setFullData(prev => { const next = prev.map(r => r.slice()); next[i][j] = v; return next; });
  }
  function addRow() { setFullData(prev => [...prev, Array.from({ length: prev[0].length }, () => "")]); }
  function delRow() { setFullData(prev => prev.length > 2 ? prev.slice(0, -1) : prev); }
  function addCol() { setFullData(prev => prev.map((row, i) => [...row, i === 0 ? `col_${row.length + 1}` : ""])); }
  function delCol() { setFullData(prev => prev[0].length > 1 ? prev.map(r => r.slice(0, -1)) : prev); }
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
  const exportCSV = () => download("dataset.csv", Papa.unparse(fullData), "text/csv");

  const buildContextForLLM = () => {
    const schema = names.map((n, i) => `${i}:${n}`).join(", ");
    const numericBrief = summary.filter(s=>s.count).map(s=>`${s.name}{n:${s.count},mean:${s.mean.toFixed(3)},sd:${s.stdev.toFixed(3)}}`).join("; ");
    const regBrief = regression ? `regression X=${names[xCol]} Y=${names[yCol]} slope=${regression.slope.toFixed(4)} intercept=${regression.intercept.toFixed(4)} r=${regression.r.toFixed(4)} r2=${regression.r2.toFixed(4)}` : "no regression";
    const cluBrief = clusters ? `kmeans k=${k} features=${clusters.cols.map(c=>names[c]).join(",")} inertia=${clusters.inertia.toFixed(2)} sizes=${clusters.sizes.join("/")}; centroids=${JSON.stringify(clusters.centroids)}` : "no clustering";
    const csvPreview = Papa.unparse(fullData.slice(0, Math.min(20, fullData.length)));
    return `schema: ${schema}\nsummary: ${numericBrief}\n${regBrief}\n${cluBrief}\ncsv_preview:\n${csvPreview}`;
  };

  async function explainClusters() {
    setClusterExplain("thinking…");
    try {
      const ctx = buildContextForLLM();
      const prompt = `
Explain the k-means result in plain language. 
- Describe each cluster using the selected feature names and centroid values.
- Compare cluster sizes and what that implies.
- Mention inertia and how changing k might help.
- No code. 
`.trim();
      const res = await fetch("/api/ask", { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify({ prompt, context: ctx }) });
      const json = await res.json();
      setClusterExplain(json.text || json.error || "no response");
    } catch (e) { setClusterExplain(String(e)); }
  }

  async function askAI() {
    setBusyAsk(true); setAiAnswer("");
    try {
      const ctx = buildContextForLLM();
      const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: aiQuestion, context: ctx }) });
      const json = await res.json();
      setAiAnswer(json.text || json.error || "no response");
    } catch (e) { setAiAnswer(String(e)); }
    finally { setBusyAsk(false); }
  }

  // ---- Render ----
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-white">
      <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">

        {/* DATASET PREVIEW + GENERATION BAR */}
        <section className="rounded-2xl bg-slate-900/70 ring-1 ring-white/10 p-5">
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight mb-2">DATAROOM — AI dataset lab</h1>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="md:col-span-2 space-y-2">
              <textarea className="w-full h-24 rounded-xl bg-slate-800/80 p-3 outline-none"
                        placeholder="Describe the dataset you want…"
                        value={genPrompt} onChange={(e)=>setGenPrompt(e.target.value)} />
              <div className="flex flex-wrap items-center gap-3">
                <label className="text-sm opacity-80">target rows</label>
                <input type="number" min="100" max="5000" step="100" className="w-28 px-2 py-1 rounded bg-slate-800/80"
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
                }} disabled={busyGen} className="px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50">
                  {busyGen ? "generating…" : "generate dataset"}
                </button>
                <button onClick={exportCSV} className="px-3 py-2 rounded-xl bg-slate-700 hover:bg-slate-600">export csv</button>
                <label className="px-3 py-2 rounded-xl bg-slate-800/80 cursor-pointer">
                  import csv
                  <input type="file" accept=".csv" className="hidden" onChange={(e) => e.target.files?.[0] && importCSV(e.target.files[0])} />
                </label>
              </div>
              {genError && <div className="text-xs text-red-300">{genError}</div>}
            </div>
            <div className="space-y-2">
              <div className="text-sm font-medium opacity-90">example prompts</div>
              <ul className="text-xs space-y-1">{EXAMPLES.map((ex,i)=>(
                <li key={i}><button className="text-left w-full px-2 py-1 rounded bg-slate-800/70 hover:bg-slate-800/90" onClick={()=>setGenPrompt(ex)}>{ex}</button></li>
              ))}</ul>
            </div>
          </div>
        </section>

        {/* PREVIEW (first 10 rows) */}
        <section className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">dataset preview</h2>
            <div className="text-xs opacity-70">showing first 10 of {totalRows} rows • analytics run on full dataset • export for all rows</div>
          </div>
          <div className="overflow-auto border border-white/10 rounded-xl">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-800/70 sticky top-0">
                <tr>{viewData[0].map((h, j) => (
                  <th key={j} className="px-3 py-2 border-b border-white/10">
                    <input className="w-40 bg-transparent outline-none" value={h} onChange={(e) => setCell(0, j, e.target.value)} />
                  </th>))}
                </tr>
              </thead>
              <tbody>{viewData.slice(1).map((row, i) => (
                <tr key={i} className="odd:bg-slate-900/40">
                  {row.map((cell, j) => (
                    <td key={j} className="px-3 py-1 border-b border-white/5">
                      <input className="w-40 bg-transparent outline-none" value={cell} onChange={(e) => setCell(i + 1, j, e.target.value)} placeholder="..." />
                    </td>
                  ))}
                </tr>
              ))}</tbody>
            </table>
          </div>
        </section>

        {/* QUICK CORRELATION */}
        <section className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-5">
          <div className="flex items-center gap-3 mb-2">
            <h2 className="font-semibold">quick correlation to a target</h2>
            <select value={corrTarget} onChange={(e)=>setCorrTarget(Number(e.target.value))} className="px-2 py-1 rounded bg-slate-800/80">
              {names.map((n,i)=>(<option key={i} value={i}>{n}</option>))}
            </select>
          </div>
          {topCorr.length ? (
            <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-2">
              {topCorr.map((c,idx)=>(
                <div key={idx} className="rounded-xl bg-slate-800/60 p-3">
                  <div className="text-xs opacity-70">{c.name}</div>
                  <div className="text-sm font-semibold">r = {c.r.toFixed(3)}</div>
                  <div className="text-[11px] opacity-60">{Math.abs(c.r) < 0.2 ? "weak" : Math.abs(c.r) < 0.5 ? "moderate" : "strong"} correlation</div>
                </div>
              ))}
            </div>
          ) : <div className="text-sm opacity-70">Pick a numeric target column. We’ll show the top 5 absolute correlations.</div>}
        </section>

        {/* ANALYTICS */}
        <section className="grid lg:grid-cols-2 gap-6">
          {/* REGRESSION */}
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
                          { label:"data", data: regression.points.map(([x,y])=>({x,y})), showLine:false, borderColor:"rgba(99,102,241,0.95)", backgroundColor:"rgba(99,102,241,0.95)" },
                          { label:"fit", data:(()=>{ const xs=regression.points.map(p=>p[0]); const minX=Math.min(...xs), maxX=Math.max(...xs); return [{x:minX,y:regression.slope*minX+regression.intercept},{x:maxX,y:regression.slope*maxX+regression.intercept}]; })(), showLine:true, borderColor:"rgba(236,72,153,0.95)", backgroundColor:"rgba(236,72,153,0.95)" }
                        ]
                      }}
                      options={chartOptions}
                    />
                  </ClientOnly>
                </div>
              </>
            ) : <div className="text-sm opacity-80">Pick numeric columns for X and Y (need ≥ 2 numeric rows).</div>}
          </div>

          {/* K-MEANS */}
          <div className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-5">
            <h2 className="font-semibold mb-3">k-means clustering</h2>
            <p className="text-xs opacity-70 mb-2">Choose numeric features. k-means groups rows by proximity; centroids are the “typical row” per group. Lower inertia means tighter clusters.</p>
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
                <div className="grid sm:grid-cols-3 gap-3 mb-3">
                  <div className="rounded-xl bg-slate-800/60 p-3"><div className="text-xs opacity-70">features</div><div className="text-sm font-semibold">{clusters.cols.map(c=>names[c]).join(", ")}</div></div>
                  <div className="rounded-xl bg-slate-800/60 p-3"><div className="text-xs opacity-70">sizes</div><div className="text-sm font-semibold">{clusters.sizes.join(" · ")}</div></div>
                  <div className="rounded-xl bg-slate-800/60 p-3"><div className="text-xs opacity-70">inertia (SSE)</div><div className="text-sm font-semibold">{clusters.inertia.toFixed(2)}</div></div>
                </div>

                {/* Full centroid table for selected features (limited to first 6 features for readability) */}
                <div className="overflow-auto border border-white/10 rounded-xl mb-3">
                  <table className="min-w-full text-xs">
                    <thead className="bg-slate-800/70 sticky top-0">
                      <tr>
                        <th className="px-3 py-2 border-b border-white/10 text-left">cluster</th>
                        {clusters.cols.slice(0,6).map((c,idx)=>(
                          <th key={idx} className="px-3 py-2 border-b border-white/10 text-left">{names[c]}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {clusters.centroids.map((cen, i)=>(
                        <tr key={i} className="odd:bg-slate-900/40">
                          <td className="px-3 py-2 border-b border-white/5">#{i+1} (n={clusters.sizes[i]})</td>
                          {clusters.cols.slice(0,6).map((c,idx)=>(
                            <td key={idx} className="px-3 py-2 border-b border-white/5">{Number(cen[idx]).toFixed(3)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="bg-slate-800/40 rounded-xl p-2 h-64 mb-3">
                  <ClientOnly>
                    <Scatter
                      data={(function(){
                        const c1=clusters.cols[0], c2=clusters.cols[1];
                        const pts=[]; for(let i=1;i<fullData.length;i++){ const x=Number(fullData[i][c1]); const y=Number(fullData[i][c2]); if(Number.isFinite(x)&&Number.isFinite(y)) pts.push({x,y}); }
                        const grouped={}; clusters.clusters.forEach((cid, i)=>{ (grouped[cid] ||= []).push(pts[i]); });
                        return {
                          datasets: Object.keys(grouped).map((cid, idx)=>({
                            label:`cluster ${Number(cid)+1}`,
                            data: grouped[cid],
                            showLine:false,
                            borderColor: clusterPalette[idx % clusterPalette.length],
                            backgroundColor: clusterPalette[idx % clusterPalette.length]
                          }))
                        };
                      })()}
                      options={chartOptions}
                    />
                  </ClientOnly>
                </div>

                <button onClick={explainClusters} className="px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500">explain clusters</button>
                {clusterExplain && <pre className="mt-3 whitespace-pre-wrap text-xs bg-slate-800/60 p-2 rounded-xl">{clusterExplain}</pre>}
              </>
            ) : <div className="text-sm opacity-80">Select at least two numeric features. Tip: standardize wildly different scales before clustering (not implemented here).</div>}
          </div>
        </section>

        {/* ASK */}
        <section className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-5">
          <h2 className="font-semibold mb-3">ask the analysis</h2>
          <textarea className="w-full h-24 rounded-xl bg-slate-800/80 p-2 outline-none"
                    placeholder="e.g., which features drive monthly_fee? how do clusters differ by seats vs monthly_fee?"
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
