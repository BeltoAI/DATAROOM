"use client";

import { useMemo, useRef, useState, useEffect } from "react";
import Papa from "papaparse";
import * as ss from "simple-statistics";
import kmeans from "ml-kmeans";
import dynamic from "next/dynamic";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

// Lazy load chart component client-only
const Scatter = dynamic(() => import("react-chartjs-2").then(m => m.Scatter), { ssr: false });

function makeGrid(rows, cols) {
  const grid = [];
  // Row 0 is header
  const header = Array.from({ length: cols }, (_, j) => `col_${j+1}`);
  grid.push(header);
  for (let i = 1; i < rows; i++) grid.push(Array.from({ length: cols }, () => ""));
  return grid;
}

function parseNumericColumn(grid, colIdx) {
  const vals = [];
  for (let i = 1; i < grid.length; i++) {
    const v = Number(grid[i][colIdx]);
    if (!Number.isNaN(v) && Number.isFinite(v)) vals.push(v);
  }
  return vals;
}

function columnNames(grid) {
  return grid.length ? grid[0] : [];
}

function download(filename, text, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function Home() {
  const [data, setData] = useState(() => makeGrid(8, 4));
  const [xCol, setXCol] = useState(0);
  const [yCol, setYCol] = useState(1);
  const [k, setK] = useState(3);
  const [clusterCols, setClusterCols] = useState([0,1]);
  const [aiQuestion, setAiQuestion] = useState("");
  const [aiAnswer, setAiAnswer] = useState("");
  const [busy, setBusy] = useState(false);

  // Basic summary stats per numeric column
  const summary = useMemo(() => {
    const names = columnNames(data);
    return names.map((name, j) => {
      const vals = parseNumericColumn(data, j);
      if (vals.length === 0) return { name, count: 0 };
      return {
        name,
        count: vals.length,
        mean: ss.mean(vals),
        median: ss.median(vals),
        stdev: ss.standardDeviation(vals),
        min: ss.min(vals),
        max: ss.max(vals)
      };
    });
  }, [data]);

  // Regression
  const regression = useMemo(() => {
    const xs = parseNumericColumn(data, xCol);
    const ys = parseNumericColumn(data, yCol);
    const n = Math.min(xs.length, ys.length);
    if (n < 2) return null;
    const points = [];
    for (let i = 0, c = 0; i < data.length-1 && c < n; i++) {
      // Rebuild aligned pairs from grid
    }
    // Build aligned pairs by scanning rows
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

  // K-means clustering
  const clusters = useMemo(() => {
    const cols = clusterCols.filter((c) => c >= 0 && c < columnNames(data).length);
    if (cols.length === 0 || data.length <= 1) return null;
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
      const result = kmeans(matrix, k, { seed: 42 });
      return { cols, ...result };
    } catch {
      return null;
    }
  }, [data, clusterCols, k]);

  // Scatter data for regression
  const scatterData = useMemo(() => {
    if (!regression) return null;
    const [minX, maxX] = (() => {
      const xs = regression.points.map(p => p[0]);
      return [Math.min(...xs), Math.max(...xs)];
    })();
    const line = (x) => regression.slope * x + regression.intercept;

    return {
      datasets: [
        {
          label: "data",
          data: regression.points.map(([x, y]) => ({ x, y })),
          showLine: false,
        },
        {
          label: "fit",
          data: [{ x: minX, y: line(minX) }, { x: maxX, y: line(maxX) }],
          showLine: true,
        },
      ],
    };
  }, [regression]);

  // Scatter for clusters (first two selected columns)
  const clusterScatter = useMemo(() => {
    if (!clusters || clusters.cols.length < 2) return null;
    const c1 = clusters.cols[0];
    const c2 = clusters.cols[1];
    const points = [];
    for (let i = 1; i < data.length; i++) {
      const x = Number(data[i][c1]);
      const y = Number(data[i][c2]);
      if (Number.isFinite(x) && Number.isFinite(y)) points.push({ x, y });
    }
    if (points.length !== clusters.clusters.length) return null;
    const grouped = {};
    clusters.clusters.forEach((cid, i) => {
      grouped[cid] ||= [];
      grouped[cid].push(points[i]);
    });
    return {
      datasets: Object.keys(grouped).map(cid => ({
        label: `cluster ${Number(cid)+1}`,
        data: grouped[cid],
        showLine: false
      }))
    };
  }, [clusters, data]);

  // Helpers
  const names = columnNames(data);

  function setCell(i, j, v) {
    setData(prev => {
      const next = prev.map(r => r.slice());
      next[i][j] = v;
      return next;
    });
  }

  function addRow() {
    setData(prev => [...prev, Array.from({ length: prev[0].length }, () => "")]);
  }

  function addCol() {
    setData(prev => prev.map((row, i) => [...row, i === 0 ? `col_${row.length+1}` : ""]));
  }

  function delRow() {
    setData(prev => prev.length > 2 ? prev.slice(0, -1) : prev);
  }

  function delCol() {
    setData(prev => prev[0].length > 1 ? prev.map(r => r.slice(0, -1)) : prev);
  }

  function newGrid() {
    setData(makeGrid(8, 4));
  }

  function importCSV(file) {
    Papa.parse(file, {
      complete: (res) => {
        const rows = res.data.filter(r => r.length && r.some(x => String(x).trim() !== ""));
        if (rows.length === 0) return;
        // Ensure rectangular
        const maxLen = Math.max(...rows.map(r => r.length));
        const grid = rows.map((r, i) => {
          const row = Array.from({ length: maxLen }, (_, j) => (r[j] ?? "").toString());
          if (i === 0) {
            // Normalize headers
            return row.map(h => String(h || "").trim() || "col");
          }
          return row;
        });
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
    lines.push("# Analysis Report");
    lines.push("");
    lines.push("## Schema");
    lines.push(columnNames(data).map((n, i) => `- ${i}: ${n}`).join("\n"));
    lines.push("");
    lines.push("## Summary stats (numeric columns)");
    summary.forEach(s => {
      if (!s.count) return;
      lines.push(`- ${s.name}: n=${s.count}, mean=${s.mean?.toFixed(4)}, median=${s.median?.toFixed(4)}, stdev=${s.stdev?.toFixed(4)}, min=${s.min}, max=${s.max}`);
    });
    lines.push("");
    if (regression) {
      lines.push("## Linear regression");
      lines.push(`- X = ${names[xCol]} | Y = ${names[yCol]}`);
      lines.push(`- slope = ${regression.slope.toFixed(6)}`);
      lines.push(`- intercept = ${regression.intercept.toFixed(6)}`);
      lines.push(`- R^2 = ${regression.r2.toFixed(6)}`);
      lines.push("");
    }
    if (clusters) {
      lines.push("## K-means clustering");
      lines.push(`- features = ${clusters.cols.map(c => names[c]).join(", ")}`);
      lines.push(`- k = ${k}`);
      lines.push(`- inertia = ${clusters.computeInformation().withinss?.reduce((a,b)=>a+b,0) ?? "n/a"}`);
      lines.push("");
    }
    download("analysis_report.md", lines.join("\n"));
  }

  async function askAI() {
    setBusy(true);
    setAiAnswer("");
    try {
      // Build compact context
      const schema = columnNames(data).map((n,i)=>`${i}:${n}`).join(", ");
      const numericBrief = summary.filter(s=>s.count).map(s=>`${s.name}{n:${s.count},mean:${s.mean?.toFixed(3)},sd:${s.stdev?.toFixed(3)}}`).join("; ");
      const regBrief = regression ? `regression X=${names[xCol]} Y=${names[yCol]} slope=${regression.slope.toFixed(4)} intercept=${regression.intercept.toFixed(4)} r2=${regression.r2.toFixed(4)}` : "no regression";
      const cluBrief = clusters ? `kmeans k=${k} features=${clusters.cols.map(c=>names[c]).join(",")}` : "no clustering";

      const csvPreview = Papa.unparse(data.slice(0, Math.min(20, data.length))); // cap preview

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

Answer clearly and concisely. If needed, propose next steps (new features, more tests) without hand-waving.
      `.trim();

      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
      });
      const json = await res.json();
      setAiAnswer(json.text || json.error || "no response");
    } catch (e) {
      setAiAnswer(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-white">
      <div className="max-w-7xl mx-auto px-4 py-6">
        <header className="mb-6 flex items-center justify-between">
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">
            DATAROOM — build datasets, analyze, ask
          </h1>
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
                          <input
                            className="w-32 bg-transparent outline-none"
                            value={h}
                            onChange={(e) => setCell(0, j, e.target.value)}
                          />
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.slice(1).map((row, i) => (
                      <tr key={i} className="odd:bg-slate-900/40">
                        {row.map((cell, j) => (
                          <td key={j} className="px-3 py-1 border-b border-white/5">
                            <input
                              className="w-32 bg-transparent outline-none"
                              value={cell}
                              onChange={(e) => setCell(i+1, j, e.target.value)}
                              placeholder="..."
                            />
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
                      <div className="opacity-80">
                        n={s.count} mean={s.mean.toFixed(3)} med={s.median.toFixed(3)} sd={s.stdev.toFixed(3)} min={s.min} max={s.max}
                      </div>
                    ) : (
                      <div className="opacity-60">non-numeric/empty</div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-4">
              <h2 className="font-semibold mb-3">ask the analysis</h2>
              <textarea
                className="w-full h-24 rounded-xl bg-slate-800/80 p-2 outline-none"
                placeholder="e.g., is Y increasing with X? which cluster is most spread?"
                value={aiQuestion}
                onChange={(e)=>setAiQuestion(e.target.value)}
              />
              <button onClick={askAI} disabled={busy || !aiQuestion.trim()} className="mt-2 px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50">
                {busy ? "thinking..." : "ask"}
              </button>
              {aiAnswer && (
                <pre className="mt-3 whitespace-pre-wrap text-xs bg-slate-800/60 p-2 rounded-xl">{aiAnswer.trim()}</pre>
              )}
            </div>
          </div>
        </section>

        <section className="mt-6 grid lg:grid-cols-2 gap-6">
          <div className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-4">
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
                <div className="text-sm opacity-80 mb-3">
                  slope={regression.slope.toFixed(6)} intercept={regression.intercept.toFixed(6)} R^2={regression.r2.toFixed(6)}
                </div>
                <div className="bg-slate-800/40 rounded-xl p-2">
                  {typeof window !== "undefined" && scatterData && <Scatter data={scatterData} options={{ scales: { x: { type: 'linear' }, y: { type: 'linear' }}}} />}
                </div>
              </>
            ) : <div className="text-sm opacity-60">need at least 2 numeric rows for both columns</div>}
          </div>

          <div className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-4">
            <h2 className="font-semibold mb-3">k-means clustering</h2>
            <div className="flex flex-wrap gap-2 mb-3">
              <div className="flex items-center gap-1">
                <span className="text-sm opacity-80">k</span>
                <input type="number" min="2" max="10" value={k} onChange={(e)=>setK(Number(e.target.value)||2)} className="w-16 px-2 py-1 rounded bg-slate-800/80" />
              </div>
              <div className="flex flex-wrap gap-1">
                {names.map((n,i)=>(
                  <label key={i} className="text-xs px-2 py-1 rounded bg-slate-800/80 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={clusterCols.includes(i)}
                      onChange={(e)=>{
                        setClusterCols(prev=>{
                          if (e.target.checked) return [...new Set([...prev, i])];
                          return prev.filter(x=>x!==i);
                        });
                      }}
                      className="mr-1 align-middle"
                    />
                    {n}
                  </label>
                ))}
              </div>
            </div>
            {clusterScatter ? (
              <div className="bg-slate-800/40 rounded-xl p-2">
                {typeof window !== "undefined" && <Scatter data={clusterScatter} options={{ scales: { x: { type: 'linear' }, y: { type: 'linear' }}}} />}
              </div>
            ) : <div className="text-sm opacity-60">select at least 2 numeric features and ensure rows are numeric</div>}
          </div>
        </section>

        <footer className="mt-10 text-center text-xs opacity-60">
          data lives in the browser until you export. keep rows clean; non-numeric values are ignored in math.
        </footer>
      </div>
    </main>
  );
}
