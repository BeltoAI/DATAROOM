export const runtime = "nodejs";

export async function POST(req) {
  try {
    const { prompt } = await req.json();
    const endpoint = process.env.LLM_ENDPOINT || 'http://minibelto.duckdns.org:8007/v1/completions';

    const upstream = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'local',
        prompt,
        max_tokens: 400,
        temperature: 0.2
      })
    });

    if (!upstream.ok) {
      const txt = await upstream.text();
      throw new Error(`Upstream ${upstream.status}: ${txt}`);
    }

    const data = await upstream.json();
    const text = data?.choices?.[0]?.text ?? '';
    return new Response(JSON.stringify({ text }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: String(err) }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}
