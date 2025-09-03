export const runtime = "nodejs";

function buildGenerationPrompt({ userPrompt, rows }) {
  const maxRows = Math.min(Math.max(Number(rows)||300, 50), 5000);
  return `
You are a data generator. Produce a synthetic dataset as CSV ONLY — no commentary, no markdown fences, no code block ticks.
- Row count: about ${maxRows} rows (header + data).
- First row MUST be headers with short, clean names (snake_case).
- Data MUST be coherent and realistic.
- Include a mix of numeric, categorical, dates if relevant to the topic.
- Avoid commas inside fields; if necessary, quote fields correctly.
- Decimal separator = dot.

Topic/instructions:
${userPrompt}

Return CSV ONLY.
`.trim();
}

export async function POST(req) {
  try {
    const { prompt, rows } = await req.json();
    const endpoint = process.env.LLM_ENDPOINT || "http://minibelto.duckdns.org:8007/v1/completions";

    const upstream = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "local",
        prompt: buildGenerationPrompt({ userPrompt: prompt||"generic dataset", rows }),
        max_tokens: 120000,  // your server will ignore/clip as needed
        temperature: 0.2
      })
    });

    if (!upstream.ok) {
      const txt = await upstream.text();
      throw new Error(`Upstream ${upstream.status}: ${txt}`);
    }
    const data = await upstream.json();
    let text = (data?.choices?.[0]?.text ?? "").trim();

    // If the model wraps CSV in code fences, try to unwrap.
    const fence = text.match(/```(?:csv)?\s*([\s\S]*?)```/i);
    if (fence) text = fence[1].trim();

    // Return raw csv text; client will parse
    return new Response(text, { status: 200, headers: { "Content-Type": "text/csv; charset=utf-8" }});
  } catch (err) {
    return new Response(JSON.stringify({ error: String(err) }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
