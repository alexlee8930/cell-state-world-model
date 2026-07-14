/**
 * Local harness proxy for the Cell-State World Model demo.
 *
 * The browser must NOT hold the Claude API key, and browsers can't call the
 * Modal endpoints cross-origin. This tiny Node server (no external deps) sits
 * between them:
 *   - POST /api/interpret  {text}  -> calls Claude to turn a free-text "state"
 *                                     into a structured plan (which endpoint +
 *                                     params to call).
 *   - GET  /api/cell|trajectory|intervene|generate  -> forwarded server-side to
 *                                     the Modal deployment (no browser CORS).
 *
 * Secrets come from the environment only (see .env.example); nothing is
 * committed. Run:  node server/proxy.mjs   (reads .env if present)
 */
import { createServer } from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

// --- minimal .env loader (no dependency) -----------------------------------
const __dir = dirname(fileURLToPath(import.meta.url));
const envPath = join(__dir, "..", ".env");
if (existsSync(envPath)) {
  for (const line of readFileSync(envPath, "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, "");
  }
}

const PORT = Number(process.env.PROXY_PORT || 8787);
const MODAL_BASE = (process.env.MODAL_BASE || "https://alexlee--cell-world-model-demo-web.modal.run").replace(/\/$/, "");
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY || "";
const MODEL = process.env.ANTHROPIC_MODEL || "claude-haiku-4-5-20251001";

const cors = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
  "access-control-allow-headers": "content-type",
};
const json = (res, code, obj) =>
  res.writeHead(code, { "content-type": "application/json", ...cors }).end(JSON.stringify(obj));

// --- Claude: free text -> structured plan ----------------------------------
const PLAN_SCHEMA = `Return ONLY a JSON object (no prose) with this shape:
{
  "mode": "cell" | "trajectory" | "intervene",
  "cellId": <int 0-199999>,        // the cell to show / start from
  "end": <int 0-199999>,           // trajectory/intervene destination cell
  "steps": <int 2-16>,             // number of frames
  "w": <number 0-5>,               // CFG guidance weight (default 3)
  "intervene": { "at": <int frame index>, "toward": <int 0-199999>, "strength": <number 0-1> },
  "rationale": "<one short sentence>"
}
Rules:
- "cell": a single static cell. "trajectory": roll the cell state through time
  from cellId to end. "intervene": a trajectory that is pushed toward another
  cell partway through (use the intervene block).
- The deployment indexes cells by integer id, so map the described phenotype to a
  STABLE id by hashing the wording (same text -> same id). Be deterministic.
- If the user mentions change/time/"over time"/days -> trajectory. If they
  mention a perturbation/knockout/adding a signal midway -> intervene.`;

async function interpret(text) {
  if (!ANTHROPIC_KEY) throw new Error("ANTHROPIC_API_KEY not set (put it in cell-world-model/.env)");
  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": ANTHROPIC_KEY,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 400,
      system:
        "You translate a biologist's free-text description of a cell state into control parameters for a cell-state world-model demo. " +
        PLAN_SCHEMA,
      messages: [{ role: "user", content: text }],
    }),
  });
  if (!r.ok) throw new Error(`Claude ${r.status}: ${(await r.text()).slice(0, 300)}`);
  const data = await r.json();
  const raw = (data.content || []).map((b) => b.text || "").join("");
  const jsonStr = raw.slice(raw.indexOf("{"), raw.lastIndexOf("}") + 1);
  return JSON.parse(jsonStr);
}

/** Answer a question about the current cell, grounded in its state. */
async function ask(question, context) {
  if (!ANTHROPIC_KEY) throw new Error("ANTHROPIC_API_KEY not set (put it in cell-world-model/.env)");
  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 220,
      system:
        "You are annotating a cell-state world model. Answer the user's question about the cell in ONE or TWO short " +
        "sentences, grounded in the numbers you are given. Be precise and honest: size/brightness are validated signal " +
        "(r≈0.66–0.68 vs a shuffle control), while fine texture from the diffusion decoder is illustrative. Never invent numbers.",
      messages: [{ role: "user", content: `Cell state: ${JSON.stringify(context)}\n\nQuestion: ${question}` }],
    }),
  });
  if (!r.ok) throw new Error(`Claude ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const d = await r.json();
  return (d.content || []).map((b) => b.text || "").join("").trim();
}

// --- forward a GET to the Modal deployment ---------------------------------
async function forward(path, query) {
  const url = `${MODAL_BASE}/${path}${query ? "?" + query : ""}`;
  const r = await fetch(url, { signal: AbortSignal.timeout(120000) });
  const body = await r.text();
  return { status: r.status, body };
}

const server = createServer(async (req, res) => {
  const u = new URL(req.url, `http://localhost:${PORT}`);
  const path = u.pathname.replace(/^\/api\//, "");
  if (req.method === "OPTIONS") return res.writeHead(204, cors).end();

  try {
    if (u.pathname === "/api/health")
      return json(res, 200, { ok: true, modalBase: MODAL_BASE, hasKey: !!ANTHROPIC_KEY, model: MODEL });

    if (u.pathname === "/api/interpret" && req.method === "POST") {
      let body = "";
      for await (const c of req) body += c;
      const { text } = JSON.parse(body || "{}");
      if (!text) return json(res, 400, { error: "missing text" });
      return json(res, 200, { plan: await interpret(text) });
    }

    if (u.pathname === "/api/ask" && req.method === "POST") {
      let body = "";
      for await (const c of req) body += c;
      const { question, context } = JSON.parse(body || "{}");
      if (!question) return json(res, 400, { error: "missing question" });
      return json(res, 200, { answer: await ask(question, context ?? {}) });
    }

    if (["cell", "trajectory", "intervene", "generate"].includes(path) && req.method === "GET") {
      const { status, body } = await forward(path, u.searchParams.toString());
      return res.writeHead(status, { "content-type": "application/json", ...cors }).end(body);
    }

    return json(res, 404, { error: "not found", path: u.pathname });
  } catch (e) {
    return json(res, 500, { error: String(e.message || e) });
  }
});

server.listen(PORT, () => {
  console.log(`[proxy] http://localhost:${PORT}  ->  Modal ${MODAL_BASE}`);
  console.log(`[proxy] Claude key: ${ANTHROPIC_KEY ? "loaded" : "MISSING (set ANTHROPIC_API_KEY in .env)"} · model ${MODEL}`);
});
