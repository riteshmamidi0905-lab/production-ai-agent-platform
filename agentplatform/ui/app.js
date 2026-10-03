/* Operations UI. Renders the real event stream (SSE) of a run. All dynamic text is set via textContent. */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const KEY = (() => { try { return localStorage.getItem('ap_api_key') || ''; } catch (e) { return ''; } })();
  const api = async (path, opts = {}) => {
    const r = await fetch(path, { ...opts, headers: { 'Content-Type': 'application/json', ...(KEY ? { 'X-API-Key': KEY } : {}), ...(opts.headers || {}) } });
    if (!r.ok) { let d = ''; try { d = (await r.json()).detail; } catch (e) { } throw new Error(typeof d === 'string' && d ? d : `HTTP ${r.status}`); }
    return r.json();
  };
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) { if (v == null || v === false) continue; if (k === 'class') e.className = v; else if (k === 'on') for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn); else e.setAttribute(k, v === true ? '' : v); }
    for (const k of kids.flat()) { if (k == null || k === false) continue; e.append(k.nodeType ? k : document.createTextNode(String(k))); }
    return e;
  }
  const fmt = (n, d = 0) => Number(n).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
  const short = (o, n = 90) => { const s = typeof o === 'string' ? o : JSON.stringify(o); return s.length > n ? s.slice(0, n - 1) + '…' : s; };
  const EXAMPLES = [
    'Compare how llm-eval-framework and genai-doc-assistant evaluate faithfulness.',
    'How does the real-time streaming pipeline handle late events and bound state?',
    'What does the lakehouse do for data quality; and how does the streaming pipeline store its results?',
    'If control converts 500 of 5000 visitors and the variant 600 of 5000, what is the lift in percentage points ((600/5000 - 500/5000) * 100)? Which test does expkit use?',
    'What is the airspeed velocity of an unladen swallow?',
  ];
  const STAGES = ['Request', 'Plan', 'Execute', 'Evidence', 'Verify', 'Result'];
  let es = null, run = null, tick = null;

  /* ---------- status + sidebar ---------- */
  async function loadStatus() {
    const box = $('#status'); box.replaceChildren();
    try {
      const r = await fetch('/readyz'); const s = await r.json();
      box.append(h('span', { class: 'pill ' + (s.database ? 'ok' : 'bad') }, 'postgres+pgvector ' + (s.database ? '✓' : '✗')), h('span', { class: 'pill ok' }, 'cache: ' + s.cache));
      const p = await api('/v1/providers');
      const sel = $('#provider'); sel.replaceChildren(...p.providers.map((x) => h('option', { value: x.name }, `${x.name} · ${x.model}`))); sel.value = p.default;
      const det = p.providers.find((x) => x.name === p.default);
      box.append(h('span', { class: 'pill ' + (det && det.deterministic ? 'warn' : 'ok'), title: det && det.deterministic ? 'The offline provider is a deterministic heuristic, not a language model. Scores describe the platform plumbing.' : '' }, det && det.deterministic ? 'offline heuristic provider (not an LLM)' : 'provider: ' + p.default));
    } catch (e) { box.append(h('span', { class: 'pill bad' }, 'backend unreachable')); }
  }
  async function loadRuns() {
    const ul = $('#runs');
    try {
      const rs = await api('/v1/runs?limit=30');
      ul.replaceChildren(...rs.map((r) => h('li', null, h('button', { type: 'button', 'aria-current': run && run.id === r.id, on: { click: () => openRun(r.id) } },
        h('span', { class: 'dot ' + (['completed', 'failed', 'cancelled', 'interrupted'].includes(r.state) ? r.state : 'running') }), h('span', { class: 'q', title: r.query }, r.query),
        h('span', { class: 'm' }, r.latency_ms ? fmt(r.latency_ms) + 'ms' : r.state)))));
      if (!rs.length) ul.append(h('li', { class: 'muted' }, 'No runs yet.'));
    } catch (e) { ul.replaceChildren(h('li', { class: 'muted' }, e.message)); }
  }

  /* ---------- live run model ---------- */
  function resetRun(id, query) {
    run = { id, query, state: 'created', plan: null, steps: {}, tools: [], evidence: {}, usage: { prompt_tokens: 0, completion_tokens: 0, cost_usd: 0, calls: 0 }, claims: [], limitations: [], title: null, verification: null, failure: null, t0: null, tEnd: null, events: [], approvals: {}, stage: -1, evCount: 0 };
    $('#liveEmpty').hidden = true; $('#live').hidden = false; $('#approvalCard').hidden = true;
    for (const id of ['plan', 'evidence', 'events']) $('#' + id).replaceChildren();
    $('#tools tbody').replaceChildren(); $('#result').replaceChildren(h('p', { class: 'muted' }, 'Waiting for the report…')); $('#verification').replaceChildren(); $('#planSummary').textContent = '—';
    renderStepper(); renderMeters();
    clearInterval(tick); tick = setInterval(renderMeters, 200);
  }

  function stageFromState(s) { return { created: 0, guarding: 0, planning: 1, executing: 2, synthesizing: 4, verifying: 4, completed: 5 }[s] ?? run.stage; }
  function renderStepper() {
    const li = STAGES.map((n, i) => {
      let c = '';
      if (run.state === 'failed' || run.state === 'interrupted') c = i === Math.max(0, run.stage) ? 'fail' : i < run.stage ? 'done' : '';
      else if (run.state === 'cancelled') c = i < run.stage ? 'done' : '';
      else if (run.state === 'completed') c = 'done';
      else if (run.pendingApproval && i === 2) c = 'wait';
      else c = i < run.stage ? 'done' : i === run.stage ? 'active' : '';
      return h('li', { class: c }, n);
    });
    $('#stepper').replaceChildren(...li);
  }

  function handle(ev) {
    const d = ev.data; run.events.push(ev);
    if (run.t0 == null) run.t0 = ev.ts;
    switch (ev.type) {
      case 'run.created': run.query = d.query; break;
      case 'state.changed': run.state = d.to; run.stage = Math.max(run.stage, stageFromState(d.to)); break;
      case 'plan.created': run.plan = d.plan; renderPlan(); break;
      case 'plan.fallback': $('#planSummary').textContent = 'Planner output rejected (' + d.reason + ') — using the safe fallback plan.'; break;
      case 'step.started': run.steps[d.step_id] = { status: 'run', ...d }; renderPlan(); break;
      case 'step.completed': if (run.steps[d.step_id]) run.steps[d.step_id].status = d.status === 'ok' ? 'ok' : 'bad'; renderPlan(); break;
      case 'tool.requested': run.tools.push({ id: d.call_id, tool: d.tool, args: d.args, status: 'running', ms: null, attempts: 1, risk: d.risk }); renderTools(); break;
      case 'tool.approval_required': run.pendingApproval = d; showApproval(d); break;
      case 'tool.approved': case 'tool.denied': run.pendingApproval = null; $('#approvalCard').hidden = true; setTool(d.call_id, { status: ev.type === 'tool.denied' ? 'denied' : 'approved' }); break;
      case 'tool.completed': setTool(d.call_id, { status: 'ok', ms: d.duration_ms, attempts: d.attempts }); break;
      case 'tool.failed': setTool(d.call_id, { status: d.status, ms: null, attempts: d.attempts || 1, error: d.error }); break;
      case 'tool.retrying': setTool(d.call_id, { attempts: d.attempt + 1 }); break;
      case 'evidence.added': run.evidence[d.id] = d; run.evCount++; run.stage = Math.max(run.stage, 3); addEvidence(d); break;
      case 'guardrail.evidence': { const el = $('#ev-' + d.evidence_id); if (el) el.classList.add('flag'); break; }
      case 'usage.updated': run.usage = d.total; break;
      case 'verification.completed': run.verification = d; renderVerification(); break;
      case 'report.title': run.title = d.title; renderResult(); break;
      case 'report.claim': run.claims.push(d); renderResult(); break;
      case 'report.limitation': run.limitations.push(d.text); renderResult(); break;
      case 'report.final': run.finalAnswer = d.answer; renderResult(); break;
      case 'run.completed': run.state = 'completed'; run.tEnd = ev.ts; run.usage = d.usage; finish(); break;
      case 'run.failed': run.state = 'failed'; run.tEnd = ev.ts; run.failure = d; run.usage = d.usage || run.usage; finish(); renderResult(); break;
      case 'run.cancelled': run.state = 'cancelled'; run.tEnd = ev.ts; finish(); break;
    }
    addEventRow(ev); renderStepper(); renderMeters();
  }
  function finish() { clearInterval(tick); renderMeters(); loadRuns(); $('#go').disabled = false; }

  /* ---------- renderers ---------- */
  function renderPlan() {
    if (!run.plan) return;
    $('#planSummary').textContent = run.plan.summary;
    $('#plan').replaceChildren(...run.plan.steps.map((s) => { const st = run.steps[s.id]; return h('li', { class: st ? st.status : '' }, h('span', { class: 'id' }, s.id), h('div', null, s.goal, h('small', null, `${s.tool}(${short(s.args, 70)})` + (s.depends_on.length ? ' ← ' + s.depends_on.join(', ') : ''))), h('span', { class: 'st' }, st ? (st.status === 'run' ? 'running' : st.status === 'ok' ? 'done' : 'failed') : 'pending')); }));
  }
  function setTool(id, patch) { const t = run.tools.find((x) => x.id === id); if (t) Object.assign(t, patch); renderTools(); }
  function renderTools() {
    $('#tools tbody').replaceChildren(...run.tools.map((t) => h('tr', null, h('td', null, h('b', null, t.tool), ' ', h('span', { class: 'tag ' + t.risk }, t.risk)), h('td', { class: 'args', title: JSON.stringify(t.args) }, short(t.args, 60)), h('td', null, h('span', { class: 'tag ' + (t.status === 'running' || t.status === 'approved' ? '' : t.status) }, t.status), t.error ? h('div', { class: 'muted', style: 'font-size:11px' }, short(t.error, 60)) : null), h('td', null, t.ms == null ? '…' : fmt(t.ms, 1)), h('td', null, t.attempts))));
    const ok = run.tools.filter((t) => t.status === 'ok').length; $('#toolSummary').textContent = run.tools.length ? `${ok}/${run.tools.length} succeeded` : '';
  }
  function addEvidence(e) {
    const bar = (label, v, cls = '') => h('div', { class: 'bar ' + cls }, `${label} ${v == null ? '—' : v.toFixed(2)}`, h('i', null, h('u', { style: `width:${Math.max(0, Math.min(1, v || 0)) * 100}%` })));
    const card = h('div', { class: 'ev' + (e.untrusted_flags.length ? ' flag' : ''), id: 'ev-' + e.id }, h('div', { class: 'evh' }, h('span', null, h('b', null, e.id), '  ', e.doc_id, '  ', e.source !== 'corpus' ? e.source : ''), h('span', null, e.untrusted_flags.length ? '⚠ ' + e.untrusted_flags.join(', ') : 'step ' + (e.step_id || ''))),
      h('p', null, e.text), h('div', { class: 'bars' }, bar('vector', e.vector_score), bar('lexical', e.lexical_score), bar('fused', e.fused_score, 'f')));
    card.addEventListener('click', () => card.classList.toggle('open'));
    $('#evidence').append(card); $('#evSummary').textContent = `${run.evCount} passage(s) · ${new Set(Object.values(run.evidence).map((x) => x.doc_id)).size} document(s)`;
  }
  function cite(text) {
    const frag = document.createDocumentFragment();
    String(text).split(/(\[e\d+\])/).forEach((p) => { const m = /^\[(e\d+)\]$/.exec(p); if (m) frag.append(h('span', { class: 'cite', role: 'button', tabindex: 0, on: { click: () => flash(m[1]), keydown: (ev) => { if (ev.key === 'Enter') flash(m[1]); } } }, m[1])); else frag.append(p); });
    return frag;
  }
  function flash(id) { const el = $('#ev-' + id); if (!el) return; el.scrollIntoView({ block: 'center', behavior: 'smooth' }); el.classList.add('flash', 'open'); setTimeout(() => el.classList.remove('flash'), 1600); }
  function renderResult() {
    const box = $('#result'); box.replaceChildren();
    if (run.failure) { box.append(h('p', { class: 'err' }, `Run failed · ${run.failure.failure_class}: ${run.failure.error}`)); return; }
    if (run.title) box.append(h('h3', null, run.title));
    if (run.claims.length) box.append(h('ul', null, run.claims.map((c) => h('li', null, cite(c.text), ' ', c.evidence_ids.map((i) => cite(`[${i}]`))))));
    else if (run.finalAnswer) box.append(h('p', null, run.finalAnswer));
    run.limitations.forEach((l) => box.append(h('p', { class: 'lim' }, '⚠ ' + l)));
    if (!box.children.length) box.append(h('p', { class: 'muted' }, 'Waiting for the report…'));
  }
  function renderVerification() {
    const v = run.verification; if (!v) return;
    $('#verification').replaceChildren(h('div', { class: 'vbox' }, h('b', { class: v.ok ? 'ok' : 'bad' }, v.ok ? 'VERIFIED' : 'NOT VERIFIED'), `  groundedness ${v.groundedness.toFixed(2)} · citation validity ${v.citation_validity.toFixed(2)}` + (v.repaired ? ' · repaired' : ''),
      v.unsupported_claims.length ? h('div', null, 'unsupported: ' + short(v.unsupported_claims, 140)) : null, v.invalid_citations.length ? h('div', null, 'invalid citations: ' + v.invalid_citations.join(', ')) : null, v.notes.length ? h('div', null, v.notes.join('; ')) : null));
  }
  function renderMeters() {
    if (!run) return;
    const end = run.tEnd || Date.now() / 1000, elapsed = run.t0 ? (end - run.t0) * 1000 : 0, u = run.usage;
    const ok = run.tools.filter((t) => t.status === 'ok').length, tot = run.tools.length;
    const m = (v, l) => h('div', { class: 'meter' }, h('b', null, v), h('span', null, l));
    $('#meters').replaceChildren(m(fmt(elapsed) + ' ms', run.state === 'completed' || run.state === 'failed' ? 'total latency' : 'elapsed'), m(fmt(u.prompt_tokens + u.completion_tokens), 'tokens' + (u.estimated ? ' (est.)' : '')), m('$' + (u.cost_usd || 0).toFixed(4), 'est. cost'), m(tot ? `${ok}/${tot}` : '—', 'tool success'),
      m(String(u.calls || 0), 'llm calls'), m(String(run.evCount), 'evidence'), m(run.verification ? run.verification.groundedness.toFixed(2) : '—', 'groundedness'), m(run.state, 'state'));
  }
  function addEventRow(ev) {
    const raw = $('#rawToggle').checked, rel = run.t0 ? Math.round((ev.ts - run.t0) * 1000) : 0;
    const li = h('li', null, h('span', { class: 'n' }, '+' + rel), h('span', { class: 'ty' }, ev.type + '  ' + (raw ? '' : short(Object.fromEntries(Object.entries(ev.data).filter(([k]) => !['text', 'plan', 'answer', 'args'].includes(k))), 80))), raw ? h('span', { class: 'raw' }, JSON.stringify(ev.data)) : null);
    const ol = $('#events'); ol.append(li); ol.scrollTop = ol.scrollHeight;
  }
  function showApproval(d) {
    const body = $('#approvalBody'), card = $('#approvalCard'), id = d.approval_id;
    const decide = async (approve) => { try { await api('/v1/approvals/' + id, { method: 'POST', body: JSON.stringify({ approve, reason: approve ? 'approved in UI' : 'denied in UI' }) }); } catch (e) { body.append(h('p', { class: 'err' }, e.message)); } };
    body.replaceChildren(h('div', { class: 'approval' }, h('p', null, 'The agent wants to call a ', h('b', null, d.risk), ' tool: ', h('code', null, d.tool), '. The run is paused until you decide (expires in ' + fmt(d.expires_in_s) + 's → denied).'),
      h('pre', null, JSON.stringify(d.args, null, 2)), h('div', { class: 'btns' }, h('button', { class: 'yes', type: 'button', on: { click: () => decide(true) } }, 'Approve'), h('button', { class: 'no', type: 'button', on: { click: () => decide(false) } }, 'Deny'))));
    card.hidden = false; card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  /* ---------- streaming ---------- */
  function stream(id) {
    if (es) es.close();
    es = new EventSource(`/v1/runs/${id}/events` + (KEY ? `?api_key=${encodeURIComponent(KEY)}` : ''));
    const types = ['run.created', 'state.changed', 'guardrail.input', 'guardrail.evidence', 'plan.created', 'plan.fallback', 'step.started', 'step.completed', 'tool.requested', 'tool.approval_required', 'tool.approved', 'tool.denied', 'tool.completed', 'tool.failed', 'tool.retrying', 'evidence.added', 'evidence.pruned', 'usage.updated', 'provider.retrying', 'provider.error', 'provider.fallback', 'verification.completed', 'report.title', 'report.claim', 'report.limitation', 'report.final', 'report.saved', 'report.not_saved', 'run.completed', 'run.failed', 'run.cancelled'];
    types.forEach((t) => es.addEventListener(t, (m) => { handle(JSON.parse(m.data)); if (['run.completed', 'run.failed', 'run.cancelled'].includes(t)) es.close(); }));
    es.onerror = () => { if (es.readyState === EventSource.CLOSED) return; };
  }
  async function openRun(id) { resetRun(id, ''); stream(id); loadRuns(); }

  async function submit(e) {
    e.preventDefault(); $('#formErr').textContent = '';
    const q = $('#query').value.trim(); if (q.length < 3) { $('#formErr').textContent = 'Enter a request (3+ characters).'; return; }
    const mw = parseInt($('#maxwords').value, 10);
    const body = { query: q, provider: $('#provider').value, constraints: { require_approval_for: $('#approval').value, save_report: $('#saveReport').checked, ...(mw ? { max_words: mw } : {}) } };
    $('#go').disabled = true;
    try { const r = await api('/v1/runs', { method: 'POST', body: JSON.stringify(body) }); resetRun(r.run_id, q); stream(r.run_id); loadRuns(); }
    catch (err) { $('#formErr').textContent = err.message; $('#go').disabled = false; }
  }

  /* ---------- evaluation tab ---------- */
  const METRICS = [['groundedness', 'groundedness'], ['citation_f1', 'citation F1'], ['task_completion', 'task completion'], ['tool_success', 'tool success'], ['instruction_adherence', 'instruction adherence'], ['consistency', 'consistency'], ['hallucinated_number_rate', 'unsupported numbers', true], ['latency_p50_ms', 'latency p50 ms', true], ['latency_p95_ms', 'latency p95 ms', true]];
  function evalGrid(m, base) {
    return h('div', { class: 'kvgrid' }, METRICS.map(([k, label, lower]) => {
      const v = m[k]; const b = base && base[k]; let delta = null;
      if (v != null && b != null && !k.startsWith('latency')) { const dd = v - b; if (Math.abs(dd) >= 0.0005) delta = h('span', { class: 'd ' + ((lower ? dd < 0 : dd > 0) ? 'up' : 'down') }, (dd > 0 ? '+' : '') + dd.toFixed(3)); }
      return h('div', null, h('b', null, v == null ? '—' : k.startsWith('latency') ? fmt(v, 1) : v.toFixed(3), delta), h('span', null, label));
    }));
  }
  async function loadEvals() {
    const tb = $('#evalList tbody'); tb.replaceChildren();
    try {
      const list = await api('/v1/evals');
      list.forEach((r) => tb.append(h('tr', { style: 'cursor:pointer', on: { click: () => showEval(r.id) } }, h('td', null, new Date(r.created_at * 1000).toLocaleString()), h('td', null, r.provider), h('td', null, fmt(r.metrics.groundedness, 3)), h('td', null, fmt(r.metrics.citation_f1, 3)), h('td', null, fmt(r.metrics.task_completion, 3)), h('td', null, fmt(r.metrics.tool_success, 3)), h('td', null, fmt(r.metrics.latency_p95_ms, 1)),
        h('td', null, r.gate ? h('span', { class: 'gate ' + (r.gate.passed ? 'pass' : 'fail') }, r.gate.passed ? 'PASS' : 'FAIL') : '—'))));
      if (list.length) showEval(list[0].id); else $('#evalDetail').replaceChildren(h('div', { class: 'card muted' }, 'No evaluation has been run yet. Press “Run evaluation”.'));
    } catch (e) { $('#evalDetail').replaceChildren(h('div', { class: 'card err' }, e.message)); }
  }
  async function showEval(id) {
    const r = await api('/v1/evals/' + id); const g = r.gate;
    $('#evalDetail').replaceChildren(h('div', { class: 'card' }, h('div', { class: 'cardhead' }, h('h2', null, `${r.dataset} · ${r.provider} · ${r.metrics.cases} cases · ${r.config.repeats} repeats`), g ? h('span', { class: 'gate ' + (g.passed ? 'pass' : 'fail') }, g.passed ? 'REGRESSION GATE PASS' : 'REGRESSION GATE FAIL') : null),
      r.config.deterministic_provider ? h('p', { class: 'muted' }, 'Provider is the deterministic offline heuristic (not an LLM): these numbers measure retrieval, verification and orchestration quality. Consistency is 1.0 by construction for a deterministic provider.') : null,
      evalGrid(r.metrics, g && g.baseline), g && g.failures.length ? h('p', { class: 'err' }, g.failures.map((f) => `${f.metric}: ${f.current} vs baseline ${f.baseline}`).join(' · ')) : null,
      h('div', { class: 'tablewrap' }, h('table', null, h('thead', null, h('tr', null, ['case', 'state', 'groundedness', 'citation F1', 'task completion', 'adherence', 'ms', 'notes'].map((x) => h('th', null, x)))),
        h('tbody', null, r.cases.map((c) => h('tr', null, h('td', null, c.id), h('td', null, c.state + (c.failure_class ? ' · ' + c.failure_class : '')), ...['groundedness', 'citation_f1', 'task_completion', 'instruction_adherence'].map((k) => h('td', null, c.metrics[k] == null ? '—' : c.metrics[k].toFixed(2))), h('td', null, fmt(c.latency_ms, 1)), h('td', { class: 'muted' }, c.notes.join('; ') || (c.facts_hit && c.facts_hit.some((x) => !x) ? 'missing fact' : '')))))))));
  }
  async function runEval() {
    const btn = $('#runEval'); btn.disabled = true; btn.textContent = 'Running…';
    try {
      const before = (await api('/v1/evals')).length; await api(`/v1/evals/run?repeats=${$('#repeats').value}&provider=${$('#provider').value}`, { method: 'POST' });
      for (let i = 0; i < 120; i++) { await new Promise((r) => setTimeout(r, 1500)); if ((await api('/v1/evals')).length > before) break; }
      await loadEvals();
    } catch (e) { $('#evalNote').textContent = e.message; } finally { btn.disabled = false; btn.textContent = 'Run evaluation'; }
  }

  /* ---------- knowledge + system ---------- */
  async function loadKnowledge() { const tb = $('#docs tbody'); const d = await api('/v1/documents'); tb.replaceChildren(...d.map((x) => h('tr', null, h('td', null, x.id), h('td', null, x.title), h('td', null, x.chunks)))); }
  async function loadSystem() {
    const [t, p] = await Promise.all([api('/v1/tools'), api('/v1/prompts')]);
    $('#toolList tbody').replaceChildren(...t.map((x) => h('tr', null, h('td', null, h('code', null, x.name)), h('td', null, h('span', { class: 'tag ' + x.risk }, x.risk)), h('td', null, x.description))));
    $('#promptList tbody').replaceChildren(...p.map((x) => h('tr', null, h('td', null, x.name), h('td', null, x.version), h('td', null, h('code', null, x.sha)), h('td', { class: 'muted' }, x.notes))));
  }

  /* ---------- wiring ---------- */
  function tab(name) {
    document.querySelectorAll('.tabs button').forEach((b) => b.setAttribute('aria-selected', b.dataset.tab === name));
    document.querySelectorAll('.tab').forEach((s) => { s.hidden = s.id !== 'tab-' + name; });
    if (name === 'eval') loadEvals(); if (name === 'knowledge') loadKnowledge(); if (name === 'system') loadSystem();
  }
  document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => tab(b.dataset.tab)));
  $('#runForm').addEventListener('submit', submit);
  $('#refreshRuns').addEventListener('click', loadRuns);
  $('#runEval').addEventListener('click', runEval);
  $('#docForm').addEventListener('submit', async (e) => { e.preventDefault(); try { const r = await api('/v1/documents', { method: 'POST', body: JSON.stringify({ id: $('#docId').value, title: $('#docTitle').value, text: $('#docText').value }) }); $('#docMsg').textContent = `${r.status}: ${r.chunks} chunk(s)`; loadKnowledge(); } catch (err) { $('#docMsg').textContent = err.message; } });
  $('#examples').replaceChildren(...EXAMPLES.map((q) => h('button', { type: 'button', title: q, on: { click: () => { $('#query').value = q; $('#query').focus(); } } }, q.length > 46 ? q.slice(0, 44) + '…' : q)));
  $('#query').value = EXAMPLES[0];
  loadStatus(); loadRuns();
  const params = new URLSearchParams(location.search); if (params.get('run')) openRun(params.get('run'));
})();
