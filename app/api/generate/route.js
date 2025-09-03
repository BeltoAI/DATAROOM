export const runtime = "nodejs";

/* ===== deterministic PRNG ===== */
function makeRNG(seed = 42) {
  let s = seed >>> 0;
  return () => (s = (1664525 * s + 1013904223) >>> 0) / 2 ** 32;
}

/* ===== helpers ===== */
const toSnake = (s) =>
  String(s).trim().toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .replace(/_{2,}/g, "_");

function parseEnumList(prompt) {
  // look for {...} blocks → enums
  const enums = {};
  const re = /([a-zA-Z0-9_ ]+)\s*\{([^}]+)\}/g;
  let m;
  while ((m = re.exec(prompt))) {
    const key = toSnake(m[1]);
    const vals = m[2].split(/[,|]/).map(v => toSnake(v));
    if (key && vals.length) enums[key] = vals;
  }
  return enums;
}

function inferColumnsFromPrompt(prompt) {
  // try to read "name: a, b, c, d" after the first colon
  const afterColon = prompt.split(":").slice(1).join(":");
  const raw = afterColon ? afterColon : prompt;
  const tokens = raw.split(/[,|\n]/).map(x => x.trim()).filter(Boolean);

  // remove any enum braces from tokens
  const cols = [];
  for (let t of tokens) {
    t = t.replace(/\{[^}]*\}/g, "").trim();
    if (!t) continue;
    cols.push(toSnake(t));
  }
  // uniq + filter junk
  const seen = new Set();
  const out = [];
  for (const c of cols) {
    if (!c || c.length > 40) continue;
    if (!seen.has(c)) { seen.add(c); out.push(c); }
  }
  // sensible defaults if nothing found
  return out.length ? out : ["user_id", "signup_date", "plan", "monthly_fee", "seats", "active_flag", "churn_date", "region"];
}

function typeForCol(col, enums) {
  if (enums[col]) return { kind: "enum", values: enums[col] };
  if (/^(.*_)?id$/.test(col) || /_id$/.test(col)) return { kind: "id" };
  if (/date|timestamp/i.test(col)) return { kind: "date" };
  if (/flag|bool|active/.test(col)) return { kind: "bool" };
  if (/plan|segment|variant|status|region|country|state|city|ticker|sector/.test(col)) return { kind: "enum_guess" };
  if (/price|fee|amount|revenue|subtotal|total|tax|cost|open|close|high|low|volume/.test(col)) return { kind: "float" };
  if (/count|seats|items|clicks|time|age/.test(col)) return { kind: "int" };
  return { kind: "string" };
}

function synthValue(kind, rng) {
  const r = rng();
  const randInt = (a, b) => a + Math.floor(r * (b - a + 1));
  const randFloat = (a, b, d=2) => Number((a + r * (b - a)).toFixed(d));
  switch (kind) {
    case "id": return randInt(100000, 999999);
    case "date": {
      // between 2021-01-01 and 2025-09-01
      const start = new Date("2021-01-01").getTime();
      const end = new Date("2025-09-01").getTime();
      const ts = start + Math.floor(r * (end - start));
      const d = new Date(ts);
      return d.toISOString().slice(0,10);
    }
    case "bool": return r < 0.8 ? 1 : 0; // skewed active
    case "int": return randInt(0, 100);
    case "float": return randFloat(0, 1000, 2);
    case "enum_guess": {
      const pools = [
        ["free","pro","business"],
        ["A","B"],
        ["ok","warning","fail"],
        ["americas","emea","apac"],
        ["student","enterprise","smb"],
      ];
      const pool = pools[randInt(0, pools.length - 1)];
      return pool[randInt(0, pool.length - 1)];
    }
    case "string": {
      const nouns = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel","india","juliet"];
      return nouns[randInt(0, nouns.length - 1)];
    }
    default: return "";
  }
}

function generateCSVFallback(prompt, rows=1000, seed=42) {
  const rng = makeRNG(seed);
  const enums = parseEnumList(prompt);
  const cols = inferColumnsFromPrompt(prompt);
  const schema = cols.map(c => ({ name: c, type: typeForCol(c, enums) }));

  const header = cols.join(",");
  const lines = [header];

  for (let i = 0; i < rows; i++) {
    const vals = schema.map((col) => {
      if (col.type.kind === "enum") {
        const v = col.type.values[Math.floor(rng()*col.type.values.length)];
        return v;
      }
      return synthValue(col.type.kind, rng);
    });
    // churn_date logic for SaaS: if active_flag=0, add a later date
    const ai = cols.indexOf("active_flag");
    const si = cols.indexOf("signup_date");
    const ci = cols.indexOf("churn_date");
    if (ai !== -1 && si !== -1 && ci !== -1) {
      const active = Number(vals[ai]);
      if (!active) {
        // churn after signup
        const base = new Date(vals[si]).getTime();
        const churn = new Date(base + Math.floor(rng()*200)*86400000);
        vals[ci] = churn.toISOString().slice(0,10);
      } else {
        vals[ci] = "";
      }
    }
    // quote only if necessary (avoid commas inside values)
    const safe = vals.map(v => {
      const s = String(v ?? "");
      return /[",\n]/.test(s) ? `"${s.replace(/"/g,'""')}"` : s;
    });
    lines.push(safe.join(","));
  }
  return lines.join("\n");
}

function buildLLMPrompt({ userPrompt, rows }) {
  const maxRows = Math.min(Math.max(Number(rows)||1000, 50), 5000);
  return `
You are a data generator. Output CSV ONLY (no prose, no code fences).
- Rows: about ${maxRows} lines INCLUDING header row.
- First row MUST be headers with short snake_case names.
- Use realistic, coherent values; avoid commas inside fields unless quoted.
- If dataset implies dates, use ISO date "YYYY-MM-DD".
- If dataset implies booleans, use 0/1.
Topic:
${userPrompt}
CSV ONLY.
`.trim();
}

export async function POST(req) {
  try {
    const { prompt, rows } = await req.json();
    const endpoint = process.env.LLM_ENDPOINT || "http://minibelto.duckdns.org:8007/v1/completions";

    // 1) Ask the LLM
    const upstream = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "local",
        prompt: buildLLMPrompt({ userPrompt: prompt || "generic dataset", rows }),
        max_tokens: 120000,
        temperature: 0.2
      })
    });

    let csv = "";
    if (upstream.ok) {
      const data = await upstream.json();
      // be tolerant to shapes: {choices:[{text}]} OR {choices:[{message:{content}}]}
      const parts = (data?.choices || []).map(c => (c?.text ?? c?.message?.content ?? "")).filter(Boolean);
      csv = (parts.join("") || "").trim();
      // unwrap ```csv ... ```
      const fence = csv.match(/```(?:csv)?\s*([\s\S]*?)```/i);
      if (fence) csv = fence[1].trim();
    }

    // 2) Validate LLM output looks like CSV with header and at least 2 rows
    const looksCSV = (txt) => {
      if (!txt) return false;
      const lines = txt.split(/\r?\n/).filter(Boolean);
      if (lines.length < 2) return false;
      // header must have at least 2 columns
      const cols = lines[0].split(",");
      return cols.length >= 2;
    };

    if (!looksCSV(csv)) {
      // 3) Fallback: synthesize CSV server-side
      const n = Math.min(Math.max(Number(rows)||1000, 50), 5000);
      csv = generateCSVFallback(prompt || "", n, 1337);
    }

    return new Response(csv, { status: 200, headers: { "Content-Type": "text/csv; charset=utf-8" }});
  } catch (err) {
    return new Response(JSON.stringify({ error: String(err) }), { status: 500, headers: { "Content-Type": "application/json" }});
  }
}
