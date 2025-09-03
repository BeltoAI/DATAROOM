export const runtime = "nodejs";

function buildPrompt({ prompt, context }) {
  return `
You are a data analyst living inside a browser app. You must answer ONLY about the dataset summary provided. 
Absolutely DO NOT output Python, shell commands, or library install instructions. No "pip", no code blocks, no notebooks.
If the user asks for code, politely say this app already does the analysis and then explain the result in plain language.

CONTEXT (schema, numeric summary, regression, clustering, sample rows):
${context}

USER QUESTION:
${prompt}

RULES:
- Give short, concrete explanations in plain English.
- If statistics are present, restate key numbers: coefficients, r, R², cluster sizes, centroids, SSE.
- Offer 1–3 next steps relevant to this dataset (e.g., try different k, normalize features, remove outliers).
- No code. No imports. No pip. No generic “read CSV” instructions.
`.trim();
}

export async function POST(req) {
  try {
    const { prompt, context } = await req.json();
    const endpoint = process.env.LLM_ENDPOINT || 'http://minibelto.duckdns.org:8007/v1/completions';

    const upstream = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'local',
        prompt: buildPrompt({ prompt, context }),
        max_tokens: 600,
        temperature: 0.2
      })
    });

    if (!upstream.ok) {
      const txt = await upstream.text();
      throw new Error(`Upstream ${upstream.status}: ${txt}`);
    }

    const data = await upstream.json();
    const text = (data?.choices?.[0]?.text ?? '').trim();
    return new Response(JSON.stringify({ text }), { status: 200, headers: { 'Content-Type': 'application/json' }});
  } catch (err) {
    return new Response(JSON.stringify({ error: String(err) }), { status: 500, headers: { 'Content-Type': 'application/json' }});
  }
}
