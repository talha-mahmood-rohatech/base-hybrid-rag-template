'use strict';
/* Voice RAG Console.
 *
 * Turn-taking follows the Leap voice agent: the browser streams 16 kHz PCM only while
 * listening and makes no speech decisions (the server's Silero VAD sends speech_started /
 * speech_ended); the mic never streams while the assistant speaks; when playback ends the mic
 * reopens at once with a vad_hold for the echo tail; tapping the mic while it speaks = barge-in.
 */

const $ = (sel, el = document) => el.querySelector(sel);
const mic = new MicStream();
const player = new Audio();

const state = {
  ws: null,
  apiKey: '',
  echoHoldMs: 500,
  micPhase: 'ready',        // ready | listening | thinking
  hearing: false,           // VAD says the user is speaking
  playing: false,
  interruptible: false,
  queue: [],                // audio pieces waiting to play
  echoGuardUntil: 0,
  turn: null,               // the turn currently being answered
  // The mic stays off until the user taps it; from then on it reopens after every answer
  // (hands-free conversation) until they tap it off again.
  autoListen: false,
};

// =============================================================== setup / connection
const storage = {
  get(k, session = false) { try { return (session ? sessionStorage : localStorage).getItem(k) || ''; } catch { return ''; } },
  set(k, v, session = false) { try { (session ? sessionStorage : localStorage).setItem(k, v); } catch { /* blocked */ } },
};

$('#api-key').value = storage.get('rag_api_key', true);
$('#vocab').value = storage.get('rag_vocab');
$('#top-k').value = storage.get('rag_top_k') || '8';
$('#speak-answers').checked = storage.get('rag_speak') !== 'off';

let kbTimer = null;
$('#api-key').addEventListener('input', () => { clearTimeout(kbTimer); kbTimer = setTimeout(loadKbs, 400); });
if ($('#api-key').value) loadKbs();

async function loadKbs() {
  const key = $('#api-key').value.trim();
  const sel = $('#kb');
  sel.disabled = true;
  $('#connect-btn').disabled = true;
  if (!key) { sel.innerHTML = '<option>Enter an API key first</option>'; setupMsg(''); return; }
  setupMsg('Loading knowledge bases…');
  try {
    const r = await fetch('/v1/knowledge-bases', { headers: { 'X-API-Key': key } });
    if (r.status === 401) { sel.innerHTML = '<option>Invalid API key</option>'; return setupMsg('That API key was rejected.', true); }
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const kbs = await r.json();
    if (!kbs.length) { sel.innerHTML = '<option>No knowledge bases</option>'; return setupMsg('This tenant has no knowledge bases yet.', true); }
    const last = storage.get('rag_kb');
    sel.innerHTML = kbs.map((k) => `<option value="${k.id}" ${k.id === last ? 'selected' : ''}>${esc(k.name)}</option>`).join('');
    sel.disabled = false;
    $('#connect-btn').disabled = false;
    storage.set('rag_api_key', key, true);
    setupMsg(`${kbs.length} knowledge base${kbs.length > 1 ? 's' : ''} available.`);
  } catch (e) {
    setupMsg(`Could not reach the API (${e.message}).`, true);
  }
}

function setupMsg(text, isError = false) {
  const el = $('#setup-msg');
  el.textContent = text;
  el.style.color = isError ? 'var(--err)' : '';
}

$('#connect-btn').addEventListener('click', async () => {
  storage.set('rag_vocab', $('#vocab').value.trim());
  storage.set('rag_top_k', $('#top-k').value);
  storage.set('rag_kb', $('#kb').value);
  storage.set('rag_speak', $('#speak-answers').checked ? 'on' : 'off');
  if (!mic.ready) {
    try {
      mic.onChunk = (pcm) => { if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(pcm); };
      mic.onLevel = (peak) => $('#mic').style.setProperty('--level', mic.streaming ? Math.min(1, peak * 2.5).toFixed(2) : 0);
      await mic.open();
    } catch (e) {
      // Still connect: typed questions work without a microphone.
      toast(`Microphone unavailable (${e.message}). You can still type questions.`);
    }
  }
  connect();
});

$('#settings-toggle').addEventListener('click', () => { $('#setup').hidden = !$('#setup').hidden; });
$('#speak-answers').addEventListener('change', (e) => storage.set('rag_speak', e.target.checked ? 'on' : 'off'));

function connect() {
  if (state.ws) { state.ws.onclose = null; state.ws.close(); }
  state.apiKey = $('#api-key').value.trim();
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/v1/voice/ws`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;
  setPill('Connecting…', false);
  ws.onopen = () => {
    const start = { type: 'start', api_key: state.apiKey, knowledge_base_id: $('#kb').value };
    const vocab = $('#vocab').value.trim();
    if (vocab) start.stt_prompt = vocab;
    const k = parseInt($('#top-k').value, 10);
    if (k > 0) start.top_k = k;
    ws.send(JSON.stringify(start));
  };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => {
    setPill('Disconnected', false);
    mic.pause();
    haltAudio();
    setMicPhase('ready');
    $('#connect-btn').textContent = 'Reconnect';
  };
}

function setPill(text, on) {
  const p = $('#conn-pill');
  p.textContent = text;
  p.className = `pill ${on ? 'on' : 'off'}`;
}

// =============================================================== server messages
function handle(msg) {
  switch (msg.type) {
    case 'ready': {
      state.echoHoldMs = msg.config.echo_hold_ms ?? 500;
      const tts = msg.config.tts.enabled ? `${msg.config.tts.voice}` : 'text only';
      setPill(`${msg.config.knowledge_base} · ${msg.config.stt.model} · ${tts}`, true);
      $('#setup').hidden = true;
      $('#settings-toggle').hidden = false;
      $('#talk').hidden = false;
      $('#connect-btn').textContent = 'Reconnect';
      setupMsg('');
      renderMic();
      break;
    }
    case 'listen': setMicPhase(state.autoListen ? 'listening' : 'ready'); break;
    case 'speech_started': state.hearing = true; resetSteps(); markStep('heard', 'active'); renderMic(); break;
    case 'speech_ended':
      state.hearing = false;
      mic.pause();
      newTurn({ spoken: true, utteranceMs: msg.duration_ms });
      markStep('heard', 'done', fmt(msg.duration_ms));
      markStep('transcribed', 'active');
      setMicPhase('thinking');
      break;
    case 'thinking':
      setMicPhase('thinking');
      if (!state.turn) newTurn({ spoken: true });
      break;
    case 'transcript': onTranscript(msg); break;
    case 'answer': onAnswer(msg); break;
    case 'say': onSay(msg); break;
    case 'stop_audio': haltAudio(); break;
    case 'error': onError(msg); break;
  }
}

// =============================================================== turns
function newTurn({ spoken, utteranceMs = null, text = '' }) {
  $('#empty')?.remove();
  const el = $('#turn-tpl').content.firstElementChild.cloneNode(true);
  $('.q-text', el).textContent = text || (spoken ? 'Transcribing…' : '');
  $('#log').append(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'end' });
  state.turn = {
    el, spoken, utteranceMs, t0: performance.now(), sttMs: null, ragMs: null, ttsMs: null,
    firstAudioAt: null, audio: [], citations: [], traceId: null, done: false,
  };
  if (!spoken) { resetSteps(); markStep('heard', 'skip'); markStep('transcribed', 'skip'); markStep('answered', 'active'); }
  return state.turn;
}

function onTranscript(msg) {
  const t = state.turn || newTurn({ spoken: msg.latency_ms != null });
  const q = $('.q', t.el);
  if (msg.latency_ms != null) t.sttMs = msg.latency_ms;
  if (!msg.reliable) {
    q.classList.add('unreliable');
    $('.q-text', q).textContent = msg.text ? `“${msg.text}”` : '(silence or noise)';
    $('.q-meta', q).textContent = 'not understood — mic reopened, please repeat';
    $('.a', t.el)?.remove();
    markStep('transcribed', 'fail', 'unclear');
    state.turn = null;
    return;
  }
  $('.q-text', q).textContent = msg.text;
  $('.q-meta', q).textContent = t.sttMs != null ? `heard in ${fmt(t.sttMs)}${msg.language ? ' · ' + msg.language : ''}` : 'typed';
  if (t.sttMs != null) markStep('transcribed', 'done', fmt(t.sttMs));
  markStep('answered', 'active');
}

function onAnswer(msg) {
  const t = state.turn || newTurn({ spoken: false });
  t.ragMs = msg.latency_ms;
  t.citations = msg.citations || [];
  t.traceId = msg.trace_id;
  const a = $('.a', t.el);
  a.classList.remove('pending');
  $('.a-text', a).innerHTML = renderMarkdown(msg.text || '(no answer)');
  renderSources(t);
  $('.a-actions', a).hidden = false;
  $('.replay', a).hidden = true;
  $('.inspect', a).onclick = () => openTrace(t.traceId);
  a.querySelectorAll('.cite').forEach((c) => c.addEventListener('click', () => focusSource(t, c.dataset.cid)));
  markStep('answered', 'done', fmt(t.ragMs));
  renderTiming(t);
  renderMic();
  if (!$('#speak-answers').checked) finishTurn(t);
  t.el.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function onSay(msg) {
  const t = state.turn;
  if (!t) return;
  if (!msg.audio_b64) { finishTurn(t); return; }   // TTS off/failed: text answer only
  if (t.firstAudioAt == null) {
    t.firstAudioAt = performance.now();
    t.ttsMs = msg.latency_ms;
    markStep('speaking', 'active', fmt(t.ttsMs));
  }
  t.audio.push(msg);
  const rp = $('.replay', t.el);
  rp.hidden = false;
  rp.onclick = () => replay(t);
  renderTiming(t);
  if ($('#speak-answers').checked) enqueue(msg); else finishTurn(t);
}

function onError(msg) {
  toast(msg.message);
  const t = state.turn;
  const a = t && $('.a', t.el);
  if (a && a.classList.contains('pending')) {
    a.classList.remove('pending');
    a.classList.add('error');
    $('.a-text', a).textContent = `Something went wrong: ${msg.message}`;
    markStep('answered', 'fail', 'error');
    finishTurn(t);
  }
}

function finishTurn(t) {
  if (t.done) return;
  t.done = true;
  renderTiming(t);
  if (state.turn === t) state.turn = null;
}

// =============================================================== rendering
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* Minimal, safe markdown (input is escaped first): paragraphs, lists, **bold**, citation chips. */
function renderMarkdown(text) {
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]/g, (_, ids) =>
      ids.split(/\s*,\s*/).map((id) => `<button type="button" class="cite" data-cid="${id}" title="Show source ${id}">${id}</button>`).join(''));
  const out = [];
  let list = null;
  for (const raw of text.replace(/\r/g, '').split('\n')) {
    const line = raw.trim();
    const bullet = line.match(/^[-*•]\s+(.*)/);
    const numbered = line.match(/^\d{1,3}[.)]\s+(.*)/);
    if (bullet || numbered) {
      const tag = bullet ? 'ul' : 'ol';
      if (!list || list !== tag) { if (list) out.push(`</${list}>`); out.push(`<${tag}>`); list = tag; }
      out.push(`<li>${inline((bullet || numbered)[1])}</li>`);
      continue;
    }
    if (list) { out.push(`</${list}>`); list = null; }
    if (line) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push(`</${list}>`);
  return out.join('');
}

function renderSources(t) {
  const box = $('.sources', t.el);
  if (!t.citations.length) { box.hidden = true; return; }
  box.innerHTML = t.citations.map((c) => {
    const section = (c.section || '').split(' > ').slice(-2).join(' › ');
    const where = [section, c.page ? `p. ${c.page}` : ''].filter(Boolean).join(' · ');
    return `<details data-cid="${c.citation_id}">
      <summary><span class="src-id">[${c.citation_id}]</span>${esc(c.document_name)}${where ? ` <span class="src-where">— ${esc(where)}</span>` : ''}</summary>
      <div class="src-snippet">${esc(c.snippet)}${c.snippet && c.snippet.length >= 300 ? '…' : ''}</div>
    </details>`;
  }).join('');
  box.hidden = false;
}

function focusSource(t, cid) {
  const d = $(`.sources details[data-cid="${cid}"]`, t.el);
  if (!d) return;
  d.open = true;
  d.classList.add('flash');
  d.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  setTimeout(() => d.classList.remove('flash'), 1200);
}

function renderTiming(t) {
  const parts = [
    ['stt', 'Speech-to-text', t.sttMs],
    ['rag', 'Retrieval + answer', t.ragMs],
    ['tts', 'First audio', t.ttsMs],
  ].filter(([, , v]) => v != null);
  if (!parts.length) return;
  const total = parts.reduce((s, [, , v]) => s + v, 0) || 1;
  const box = $('.timing', t.el);
  $('.bar', box).innerHTML = parts.map(([k, label, v]) => `<span class="seg-${k}" style="width:${(v / total) * 100}%" title="${label} ${fmt(v)}"></span>`).join('');
  const legend = parts.map(([k, label, v]) => `<span><i class="seg-${k}"></i>${label} ${fmt(v)}</span>`);
  if (t.utteranceMs) legend.unshift(`<span><i class="seg-vad"></i>Question ${fmt(t.utteranceMs)}</span>`);
  if (t.firstAudioAt != null) legend.push(`<span><b>${fmt(t.firstAudioAt - t.t0)}</b> to first audio</span>`);
  $('.legend', box).innerHTML = legend.join('');
  box.hidden = false;
}

function resetSteps() { document.querySelectorAll('#steps li').forEach((li) => { li.className = ''; $('b', li).textContent = ''; }); }
function markStep(name, status, value = '') {
  const li = $(`#steps li[data-step="${name}"]`);
  if (!li) return;
  li.className = status === 'skip' ? '' : status;
  $('b', li).textContent = status === 'skip' ? '—' : value;
}

// =============================================================== mic state
function setMicPhase(phase) {
  if (phase !== 'listening' && mic.streaming) abandonCapture();
  state.micPhase = phase;
  if (phase === 'listening' && !state.playing && state.ws?.readyState === WebSocket.OPEN && mic.ready && !mic.streaming) {
    const hold = Math.max(0, state.echoGuardUntil - Date.now());
    if (hold > 0) state.ws.send(JSON.stringify({ type: 'vad_hold', ms: Math.round(hold) }));   // before the first chunk
    mic.resume();
  }
  renderMic();
}

function abandonCapture() {
  mic.pause();
  state.hearing = false;
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ type: 'vad_reset' }));
}

function renderMic() {
  const phase = state.playing ? 'speaking' : state.micPhase;
  const btn = $('#mic');
  btn.className = phase === 'listening' && state.hearing ? 'hearing' : phase;
  btn.setAttribute('aria-pressed', String(phase === 'listening'));
  $('#mic-label').textContent = {
    ready: mic.ready ? 'Tap the mic and ask a question' : 'Microphone unavailable — type below',
    listening: state.hearing ? 'Hearing you…' : 'Listening — go ahead',
    thinking: state.turn && state.turn.ragMs != null ? 'Preparing the spoken answer…' : 'Thinking…',
    speaking: 'Speaking — tap to interrupt',
  }[phase];
}

$('#mic').addEventListener('click', () => {
  if (state.ws?.readyState !== WebSocket.OPEN) return toast('Not connected');
  if (!mic.ready) return toast('Microphone unavailable — type your question instead');
  if (state.playing) {
    if (!state.interruptible) return shake();
    haltAudio();
    state.ws.send(JSON.stringify({ type: 'barge_in' }));   // server cancels the turn, then sends listen
    return;
  }
  if (state.micPhase === 'listening') { state.autoListen = false; return setMicPhase('ready'); }
  if (state.micPhase === 'ready') { state.autoListen = true; return setMicPhase('listening'); }
  shake();   // thinking: wait for the answer
});

function shake() { const b = $('#mic'); b.classList.add('shake'); setTimeout(() => b.classList.remove('shake'), 300); }

// =============================================================== typed questions
$('#ask-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const text = $('#ask-input').value.trim();
  if (!text) return;
  if (state.ws?.readyState !== WebSocket.OPEN) return toast('Connect first');
  $('#ask-input').value = '';
  haltAudio();
  newTurn({ spoken: false, text });
  setMicPhase('thinking');
  state.ws.send(JSON.stringify({ type: 'text', text }));
});

// =============================================================== playback
function enqueue(msg) {
  state.queue.push(msg);
  if (!state.playing) playNext();
}

function playNext() {
  const next = state.queue.shift();
  if (!next) {
    const wasPlaying = state.playing;
    state.playing = false;
    state.interruptible = false;
    if (wasPlaying) {
      state.echoGuardUntil = Date.now() + state.echoHoldMs;   // must precede reopening the mic
      if (state.turn) { markStep('speaking', 'done', $('#steps li[data-step="speaking"] b').textContent); finishTurn(state.turn); }
    }
    if (state.micPhase === 'listening' || (state.autoListen && state.micPhase === 'ready')) setMicPhase('listening');
    else renderMic();
    return;
  }
  state.playing = true;
  state.interruptible = next.interruptible !== false;
  if (mic.streaming) abandonCapture();
  renderMic();
  player.src = `data:${next.mime || 'audio/mpeg'};base64,${next.audio_b64}`;
  player.onended = playNext;
  player.play().catch((e) => {
    if (e.name === 'AbortError') return;
    console.warn('playback failed', e);
    setTimeout(playNext, 50);   // never leave the mic stuck behind a failed clip
  });
}

function haltAudio() {
  const was = state.playing;
  state.queue = [];
  player.onended = null;
  player.pause();
  state.playing = false;
  state.interruptible = false;
  if (was) state.echoGuardUntil = Date.now() + state.echoHoldMs;
  renderMic();
}

function replay(t) {
  if (state.micPhase === 'thinking') return toast('Wait for the current answer first');
  haltAudio();
  if (mic.streaming) abandonCapture();
  state.micPhase = 'ready';
  state.queue.push(...t.audio);
  playNext();
}

// =============================================================== trace inspector
async function openTrace(traceId) {
  const body = $('#trace-body');
  body.innerHTML = '<p class="muted">Loading trace…</p>';
  $('#trace').showModal();
  try {
    const r = await fetch(`/v1/traces/${traceId}`, { headers: { 'X-API-Key': state.apiKey } });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const t = await r.json();
    const picked = new Set((t.selected_chunks || []).map((c) => c.chunk_id));
    const counts = t.timings_ms?.counts || {};
    const kv = [
      ['Query', t.query],
      ['Total', fmt(t.total_latency_ms)],
      ['Rerank', fmt(t.timings_ms?.rerank_ms)],
      ['LLM', t.llm?.model ? `${t.llm.model} · ${fmt(t.timings_ms?.generate_ms)}` : '—'],
      ['Tokens', t.llm?.usage ? `${t.llm.usage.prompt_tokens} in / ${t.llm.usage.completion_tokens} out` : '—'],
      ['Candidates', `${counts.dense_candidates ?? '?'} dense · ${counts.sparse_candidates ?? '?'} sparse · ${counts.rrf_candidates ?? '?'} fused`],
      ['Reranker', t.config?.reranker?.enabled ? t.config.reranker.model : 'off'],
      ['Context', `${t.context_tokens} tokens · ${picked.size} chunks`],
    ];
    const rows = (t.reranked_results || []).map((c) => `
      <tr class="${picked.has(c.chunk_id) ? 'picked' : ''}">
        <td class="num">${c.final_rank ?? ''}</td>
        <td>${esc(c.document_name || '')}<br><small>${esc((c.section || '').split(' > ').slice(-2).join(' › '))}</small></td>
        <td class="num">${c.dense_rank ?? '–'}</td>
        <td class="num">${c.sparse_rank ?? '–'}</td>
        <td class="num">${c.rrf_score != null ? c.rrf_score.toFixed(4) : '–'}</td>
        <td class="num">${c.reranker_score != null ? c.reranker_score.toFixed(3) : '–'}</td>
        <td>${esc((c.preview || '').slice(0, 140))}…</td>
      </tr>`).join('');
    body.innerHTML = `
      <div class="kv">${kv.map(([k, v]) => `<div><b>${k}</b>${esc(v)}</div>`).join('')}</div>
      <p class="muted">Reranked candidates — highlighted rows were sent to the LLM as context.</p>
      <div class="table-wrap"><table>
        <thead><tr><th>#</th><th>Source</th><th>Dense</th><th>Sparse</th><th>RRF</th><th>Rerank</th><th>Preview</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="7">No candidates</td></tr>'}</tbody>
      </table></div>`;
  } catch (e) {
    body.innerHTML = `<p style="color:var(--err)">Could not load the trace (${esc(e.message)}).</p>`;
  }
}
$('#trace-close').addEventListener('click', () => $('#trace').close());
$('#trace').addEventListener('click', (e) => { if (e.target.id === 'trace') $('#trace').close(); });

// =============================================================== helpers
function fmt(ms) {
  if (ms == null || Number.isNaN(ms)) return '–';
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

let toastTimer = null;
function toast(text) {
  const el = $('#toast');
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 6000);
}
