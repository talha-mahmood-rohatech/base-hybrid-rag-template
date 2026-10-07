'use strict';
/* Simple voice assistant page (end users). Developer details live on /voice/console.html.
 *
 * Connects to the public assistant (VOICE__PUBLIC_KNOWLEDGE_BASE_ID) without any key, or to a
 * link shared as /voice/?key=...&kb=... . Turn-taking is the same as the console: the server's
 * Silero VAD decides turns, the mic is paused while the assistant speaks, and tapping the mic
 * while it speaks interrupts it.
 */

const $ = (s) => document.querySelector(s);
const mic = new MicStream();
const player = new Audio();
const st = {
  ws: null, ready: false, phase: 'idle', hearing: false, playing: false,
  queue: [], echoMs: 500, echoUntil: 0, autoListen: false, pending: null,
};

// ------------------------------------------------------------------ connect
(async function init() {
  let start = null;
  try {
    const cfg = await (await fetch('/v1/voice/public')).json();
    if (cfg.title) { $('#title').textContent = cfg.title; document.title = cfg.title; }
    if (cfg.enabled) start = { type: 'start' };
  } catch { /* fall through to a shared link */ }

  if (!start) {
    // A shared link (/voice/?key=..&kb=..): remember it for this tab and hide it from the URL.
    const q = new URLSearchParams(location.search);
    if (q.get('key') && q.get('kb')) {
      try { sessionStorage.setItem('va_link', JSON.stringify({ key: q.get('key'), kb: q.get('kb') })); } catch { /* blocked */ }
      history.replaceState(null, '', location.pathname);
    }
    let link = null;
    try { link = JSON.parse(sessionStorage.getItem('va_link') || 'null'); } catch { /* blocked */ }
    if (link) start = { type: 'start', api_key: link.key, knowledge_base_id: link.kb };
  }

  if (!start) return status('The assistant is not available right now.');
  connect(start);
})();

function connect(start) {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/v1/voice/ws`);
  ws.binaryType = 'arraybuffer';
  st.ws = ws;
  ws.onopen = () => ws.send(JSON.stringify(start));
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => {
    st.ready = false;
    mic.pause();
    stopAudio();
    $('#mic').disabled = true;
    status('Connection lost. Refresh the page to try again.');
  };
}

// ------------------------------------------------------------------ server messages
function onMessage(m) {
  switch (m.type) {
    case 'ready':
      st.ready = true;
      st.echoMs = m.config.echo_hold_ms ?? 500;
      $('#mic').disabled = false;
      $('#type-form').hidden = false;
      render();
      break;
    case 'listen': setPhase(st.autoListen ? 'listening' : 'idle'); break;
    case 'speech_started': st.hearing = true; render(); break;
    case 'speech_ended': st.hearing = false; mic.pause(); setPhase('thinking'); break;
    case 'thinking': setPhase('thinking'); break;
    case 'transcript':
      if (!m.reliable) { status('Sorry, I didn’t catch that. Please try again.', 4000); break; }
      if (!st.pending) { add('me', m.text); st.pending = add('bot wait', null); }
      break;
    case 'answer':
      if (!st.pending) st.pending = add('bot', null);
      st.pending.className = 'msg bot';
      st.pending.innerHTML = render_md(m.text || 'Sorry, I don’t have an answer for that.');
      st.pending.scrollIntoView({ behavior: 'smooth', block: 'end' });
      st.pending = null;
      break;
    case 'say': if (m.audio_b64) { st.queue.push(m); if (!st.playing) playNext(); } break;
    case 'stop_audio': stopAudio(); break;
    case 'error':
      if (st.pending) { st.pending.className = 'msg bot'; st.pending.textContent = 'Sorry, something went wrong. Please try again.'; st.pending = null; }
      break;
  }
}

// ------------------------------------------------------------------ mic
$('#mic').addEventListener('click', async () => {
  if (!st.ready) return;
  if (st.playing) {                       // interrupt the answer
    stopAudio();
    st.ws.send(JSON.stringify({ type: 'barge_in' }));
    return;
  }
  if (st.phase === 'thinking') return;
  if (st.phase === 'listening') { st.autoListen = false; return setPhase('idle'); }
  if (!mic.ready) {
    try {
      mic.onChunk = (pcm) => { if (st.ws?.readyState === WebSocket.OPEN) st.ws.send(pcm); };
      mic.onLevel = (p) => $('#mic').style.setProperty('--level', mic.streaming ? Math.min(1, p * 2.5).toFixed(2) : 0);
      await mic.open();
    } catch {
      return status('Microphone access is blocked. You can type your question below.');
    }
  }
  st.autoListen = true;                   // keep listening after each answer until tapped off
  setPhase('listening');
});

function pauseCapture() {
  if (!mic.streaming) return;
  mic.pause();
  st.hearing = false;
  if (st.ws?.readyState === WebSocket.OPEN) st.ws.send(JSON.stringify({ type: 'vad_reset' }));
}

function setPhase(phase) {
  if (phase !== 'listening') pauseCapture();
  st.phase = phase;
  if (phase === 'listening' && !st.playing && mic.ready && !mic.streaming && st.ws?.readyState === WebSocket.OPEN) {
    const hold = Math.max(0, st.echoUntil - Date.now());
    if (hold) st.ws.send(JSON.stringify({ type: 'vad_hold', ms: Math.round(hold) }));
    mic.resume();
  }
  render();
}

function render() {
  const phase = st.playing ? 'speaking' : st.phase;
  $('#mic').className = phase === 'idle' ? '' : phase;
  const text = {
    idle: 'Tap to ask a question',
    listening: st.hearing ? 'I’m listening…' : 'Listening… go ahead',
    thinking: 'Thinking…',
    speaking: 'Tap to interrupt',
  }[phase];
  $('#mic').setAttribute('aria-label', text);
  status(text);
}

let statusTimer = null;
function status(text, holdMs = 0) {
  clearTimeout(statusTimer);
  $('#status').textContent = text;
  if (holdMs) statusTimer = setTimeout(render, holdMs);
}

// ------------------------------------------------------------------ typing
$('#type-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const text = $('#type-input').value.trim();
  if (!text || !st.ready) return;
  $('#type-input').value = '';
  stopAudio();
  add('me', text);
  st.pending = add('bot wait', null);
  setPhase('thinking');
  st.ws.send(JSON.stringify({ type: 'text', text }));
});

// ------------------------------------------------------------------ playback
function playNext() {
  const next = st.queue.shift();
  if (!next) {
    if (st.playing) st.echoUntil = Date.now() + st.echoMs;   // before the mic reopens
    st.playing = false;
    setPhase(st.phase === 'thinking' ? 'thinking' : st.autoListen ? 'listening' : 'idle');
    return;
  }
  st.playing = true;
  pauseCapture();   // never listen to ourselves
  render();
  player.src = `data:${next.mime || 'audio/mpeg'};base64,${next.audio_b64}`;
  player.onended = playNext;
  player.play().catch((e) => { if (e.name !== 'AbortError') setTimeout(playNext, 50); });
}

function stopAudio() {
  const was = st.playing;
  st.queue = [];
  player.onended = null;
  player.pause();
  st.playing = false;
  if (was) st.echoUntil = Date.now() + st.echoMs;
  render();
}

// ------------------------------------------------------------------ chat
function add(kind, text) {
  $('#hello')?.remove();
  const el = document.createElement('div');
  el.className = `msg ${kind}`;
  if (text == null) el.innerHTML = '<span class="dots"><i></i><i></i><i></i></span>';
  else el.textContent = text;
  $('#chat').append(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return el;
}

/* Plain, readable answer: citation markers removed; paragraphs, lists and bold kept. */
function render_md(text) {
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const inline = (s) => esc(s.replace(/\s*\[\d{1,3}(?:\s*,\s*\d{1,3})*\]/g, '')).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  const out = [];
  let list = null;
  for (const raw of text.replace(/\r/g, '').split('\n')) {
    const line = raw.trim();
    const item = line.match(/^(?:[-*•]|(\d{1,3})[.)])\s+(.*)/);
    if (item) {
      const tag = item[1] ? 'ol' : 'ul';
      if (list !== tag) { if (list) out.push(`</${list}>`); out.push(`<${tag}>`); list = tag; }
      out.push(`<li>${inline(item[2])}</li>`);
      continue;
    }
    if (list) { out.push(`</${list}>`); list = null; }
    if (line) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push(`</${list}>`);
  return out.join('');
}
