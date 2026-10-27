#!/usr/bin/env node
// brain-bench: same model, same prompt, same questions; only the "brain" (how knowledge reaches the model) changes.
// No dependencies. Node 18+. Works with any OpenAI-compatible chat API (default: local Ollama).
//
//   node bench.mjs                       # sample-kb + questions.json, all setups
//   node bench.mjs --kb ./my-notes --questions ./my-questions.json --setups none,files,bm25,embed,long
//
// Env: BENCH_API (default http://127.0.0.1:11434/v1), BENCH_MODEL (default gpt-oss:120b-cloud),
//      BENCH_API_KEY (optional), BENCH_EMBED_MODEL (default nomic-embed-text, via Ollama /api/embed; skipped if missing).
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const KB_DIR = arg('kb', 'sample-kb');
const QUESTIONS = JSON.parse(readFileSync(arg('questions', 'questions.json'), 'utf8'));
const SETUP_IDS = arg('setups', 'none,files,bm25,embed,long').split(',');
const API = process.env.BENCH_API ?? 'http://127.0.0.1:11434/v1';
const MODEL = process.env.BENCH_MODEL ?? 'gpt-oss:120b-cloud';
const EMBED_MODEL = process.env.BENCH_EMBED_MODEL ?? 'nomic-embed-text';
const OLLAMA = API.replace(/\/v1\/?$/, '');

// The one prompt every setup gets.
const SYSTEM = "You are an assistant for a small business. Answer the question using only the provided context. If the context does not contain the answer, reply exactly: I don't know. Answer in one short sentence.";
const userMessage = (q, context) => (context == null ? '' : `Context:\n<<<\n${context}\n>>>\n\n`) + `Question: ${q}`;

// ---------- knowledge base ----------
function walk(dir) {
  return readdirSync(dir).flatMap((n) => { const p = join(dir, n); return statSync(p).isDirectory() ? walk(p) : /\.(md|txt)$/i.test(n) ? [p] : []; });
}
const KB = walk(KB_DIR).map((p) => ({ path: relative(KB_DIR, p).replace(/\\/g, '/'), text: readFileSync(p, 'utf8') }));
const STOP = new Set('a an the and or of to in on for is are was were be by with what which who how much many does do did at as it its this that from why when whats'.split(' '));
const terms = (s) => s.toLowerCase().match(/[a-z0-9$%]+/g)?.filter((t) => !STOP.has(t) && t.length > 1) ?? [];
const block = (path, text) => `### ${path}\n${text.trim()}\n`;

// ---------- the brains ----------
// files: whole notes ranked by keyword hits (what "search my Obsidian vault" usually means), capped at 12k chars.
function files(q, cap = 12000) {
  const qt = [...new Set(terms(q))];
  const ranked = KB.map((d) => {
    const body = d.text.toLowerCase(); let distinct = 0, hits = 0;
    for (const t of qt) { const n = body.split(t).length - 1; if (n) distinct++; hits += n; }
    return { ...d, score: distinct * 100 + Math.min(hits, 50) };
  }).filter((d) => d.score > 0).sort((a, b) => b.score - a.score);
  let out = '';
  for (const d of ranked) { const b = block(d.path, d.text); if (out.length + b.length > cap) { out += b.slice(0, Math.max(0, cap - out.length)); break; } out += b + '\n'; }
  return out.trim();
}

// chunks shared by both RAG setups: ~800 chars with 100 overlap
const CHUNKS = KB.flatMap((d) => { const out = []; for (let i = 0; i < d.text.length; i += 700) { out.push({ path: d.path, text: d.text.slice(i, i + 800) }); if (i + 800 >= d.text.length) break; } return out; });

// bm25: keyword RAG, top 6 chunks (k1 1.5, b 0.75)
const BM = (() => {
  const docs = CHUNKS.map((c) => terms(`${c.path} ${c.text}`)); const df = new Map();
  const tfs = docs.map((d) => { const tf = new Map(); d.forEach((t) => tf.set(t, (tf.get(t) ?? 0) + 1)); tf.forEach((_, t) => df.set(t, (df.get(t) ?? 0) + 1)); return tf; });
  return { N: docs.length, avg: docs.reduce((n, d) => n + d.length, 0) / docs.length, lens: docs.map((d) => d.length), df, tfs };
})();
function bm25(q, k = 6) {
  const qt = [...new Set(terms(q))];
  return BM.tfs.map((tf, i) => {
    let s = 0;
    for (const t of qt) { const f = tf.get(t); if (!f) continue; const n = BM.df.get(t); s += Math.log(1 + (BM.N - n + 0.5) / (n + 0.5)) * (f * 2.5) / (f + 1.5 * (0.25 + 0.75 * BM.lens[i] / BM.avg)); }
    return { i, s };
  }).filter((x) => x.s > 0).sort((a, b) => b.s - a.s).slice(0, k).map((x) => block(CHUNKS[x.i].path, CHUNKS[x.i].text)).join('\n');
}

// embed: embedding RAG via Ollama, top 6 chunks by cosine similarity
let VECS = null;
const unit = (v) => { const n = Math.hypot(...v); return v.map((x) => x / n); };
async function embedAll(inputs) {
  const r = await fetch(`${OLLAMA}/api/embed`, { method: 'POST', body: JSON.stringify({ model: EMBED_MODEL, input: inputs }) });
  if (!r.ok) throw new Error(`embedding model "${EMBED_MODEL}" not available (${r.status})`);
  return (await r.json()).embeddings.map(unit);
}
async function embed(q, k = 6) {
  if (!VECS) { VECS = []; for (let i = 0; i < CHUNKS.length; i += 32) VECS.push(...await embedAll(CHUNKS.slice(i, i + 32).map((c) => `search_document: ${c.path}\n${c.text}`))); }
  const [qv] = await embedAll([`search_query: ${q}`]);
  return VECS.map((v, i) => ({ i, s: v.reduce((n, x, j) => n + x * qv[j], 0) })).sort((a, b) => b.s - a.s).slice(0, k).map((x) => block(CHUNKS[x.i].path, CHUNKS[x.i].text)).join('\n');
}

// long: paste every note into the prompt
const long = () => KB.map((d) => block(d.path, d.text)).join('\n');

const SETUPS = {
  none: { name: 'No brain', context: async () => null },
  files: { name: 'Notes + file search', context: async (q) => files(q) },
  bm25: { name: 'Keyword RAG (BM25)', context: async (q) => bm25(q) },
  embed: { name: 'Embedding RAG', context: async (q) => embed(q) },
  long: { name: 'Long context (paste everything)', context: async () => long() },
};

// ---------- model call + grading ----------
async function ask(user) {
  for (let attempt = 0; attempt < 5; attempt++) {
    const t0 = Date.now();
    const r = await fetch(`${API}/chat/completions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', ...(process.env.BENCH_API_KEY ? { Authorization: `Bearer ${process.env.BENCH_API_KEY}` } : {}) },
      body: JSON.stringify({ model: MODEL, temperature: 0, messages: [{ role: 'system', content: SYSTEM }, { role: 'user', content: user }] }),
    }).catch((e) => ({ ok: false, status: 0, text: async () => e.message }));
    if (r.ok) { const j = await r.json(); return { answer: (j.choices?.[0]?.message?.content ?? '').trim(), ms: Date.now() - t0 }; }
    if (r.status && r.status < 429) throw new Error(`model call failed: ${r.status} ${await r.text()}`);
    await new Promise((res) => setTimeout(res, 2000 * (attempt + 1)));
  }
  throw new Error('model call kept failing');
}
const normal = (s) => s.toLowerCase().normalize('NFKC').replace(/[‐-―−]/g, '-').replace(/[‘’]/g, "'").replace(/\*\*/g, '');
function grade(q, answer) {
  const a = normal(answer); const idk = /\bi don'?t know\b/.test(a);
  if (q.unanswerable) return idk ? 'correct' : 'made_up';
  const ok = (q.all ?? []).every((r) => new RegExp(r, 'i').test(a)) && !(q.none ?? []).some((r) => new RegExp(r, 'i').test(a));
  return ok ? 'correct' : idk ? 'dont_know' : 'made_up';
}

// ---------- run ----------
const results = { model: MODEL, kb: { notes: KB.length, chars: KB.reduce((n, d) => n + d.text.length, 0) }, setups: {} };
for (const id of SETUP_IDS) {
  const setup = SETUPS[id];
  if (!setup) { console.error(`unknown setup "${id}"`); continue; }
  const rows = [];
  try {
    for (const q of QUESTIONS) {
      const context = await setup.context(q.q);
      const { answer, ms } = await ask(userMessage(q.q, context));
      rows.push({ id: q.id, answer, grade: grade(q, answer), ms, contextChars: context?.length ?? 0 });
      process.stdout.write('.');
    }
  } catch (e) { console.error(`\n${setup.name}: skipped (${e.message})`); continue; }
  const count = (g) => rows.filter((r) => r.grade === g).length;
  results.setups[id] = { name: setup.name, correct: count('correct'), madeUp: count('made_up'), dontKnow: count('dont_know'),
    avgSeconds: +(rows.reduce((n, r) => n + r.ms, 0) / rows.length / 1000).toFixed(2), avgContextChars: Math.round(rows.reduce((n, r) => n + r.contextChars, 0) / rows.length), rows };
  console.log(`\n${setup.name}: ${count('correct')}/${QUESTIONS.length} correct, ${count('made_up')} made up`);
}
writeFileSync('results.json', JSON.stringify(results, null, 2));
console.log('\n| Setup | Correct | Made up | I don\'t know | Avg s | Context chars |\n|---|---|---|---|---|---|');
for (const s of Object.values(results.setups).sort((a, b) => b.correct - a.correct)) {
  console.log(`| ${s.name} | ${s.correct}/${QUESTIONS.length} | ${s.madeUp} | ${s.dontKnow} | ${s.avgSeconds} | ${s.avgContextChars} |`);
}
console.log('\nFull answers and grades: results.json');
