"""Embedded web chat UI served at http://localhost:7777/.

Single self-contained page (no external assets — works offline) matching
the Sparsify brand: slate ground, amber accent, mono-led type. Talks to
the same OpenAI-compatible endpoints third-party clients use.

Conversations and projects live in the *browser* (localStorage): the
server stays stateless on purpose — it is a runtime, not a database.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sparsify</title>
<style>
  :root {
    --ground:#0B0E14; --panel:#121826; --panel2:#0E1420; --line:#232B3C;
    --ink:#E7EBF2; --soft:#9AA5B8; --faint:#5B6577; --accent:#E8A33D;
    --good:#3FB27F; color-scheme: dark;
  }
  @media (prefers-color-scheme: light) {
    :root { --ground:#F2F4F8; --panel:#FFF; --panel2:#E9EDF3; --line:#D8DFEA;
      --ink:#1A2130; --soft:#4C586E; --faint:#8B96A9; --accent:#9A6A14;
      --good:#1D7A52; color-scheme: light; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--ground); color:var(--ink); height:100vh;
    display:flex; font:15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .mono { font-family: ui-monospace,"SF Mono",Menlo,Consolas,monospace; }

  /* ── sidebar ── */
  aside { width:250px; min-width:250px; border-right:1px solid var(--line);
    background:var(--panel2); display:flex; flex-direction:column; }
  aside.hidden { display:none; }
  .side-top { padding:12px; display:flex; gap:8px; }
  .side-top button { flex:1; }
  .side-scroll { flex:1; overflow-y:auto; padding:4px 8px 12px; }
  .proj { margin-top:10px; }
  .proj-head { display:flex; align-items:center; gap:6px; padding:4px 8px;
    font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--faint); }
  .proj-head .grow { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .iconbtn { background:none; border:0; color:var(--faint); cursor:pointer;
    font-size:12px; padding:2px 4px; border-radius:4px; opacity:0; }
  .proj-head:hover .iconbtn, .chat-item:hover .iconbtn { opacity:1; }
  .iconbtn:hover { color:var(--accent); background:var(--panel); }
  .chat-item { display:flex; align-items:center; gap:6px; padding:7px 10px;
    border-radius:8px; cursor:pointer; font-size:13.5px; color:var(--soft); }
  .chat-item .grow { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .chat-item:hover { background:var(--panel); }
  .chat-item.active { background:var(--panel); color:var(--ink);
    border-left:2px solid var(--accent); padding-left:8px; }
  .side-note { padding:8px 12px; font-size:10.5px; color:var(--faint);
    border-top:1px solid var(--line); }

  /* ── main column ── */
  .maincol { flex:1; display:flex; flex-direction:column; min-width:0; }
  header { display:flex; align-items:center; gap:12px; padding:10px 18px;
    border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header svg { width:20px; height:20px; }
  header .brand { font-weight:650; letter-spacing:.02em; }
  header .dot { width:8px; height:8px; border-radius:50%; background:var(--good); }
  header .dot.err { background:#C0504A; }
  header .spacer { flex:1; }
  select, input[type=number] {
    background:var(--panel2); color:var(--ink); border:1px solid var(--line);
    border-radius:8px; padding:6px 10px; font-size:13px; max-width:340px;
  }
  label.set { font-size:12px; color:var(--faint); display:flex; gap:6px; align-items:center; }
  button { background:var(--accent); border:0; color:var(--ground);
    font-weight:650; border-radius:8px; padding:8px 16px; cursor:pointer; font-size:14px; }
  button.ghost { background:transparent; color:var(--soft); border:1px solid var(--line); }
  button:disabled { opacity:.45; cursor:default; }

  main { flex:1; overflow-y:auto; padding:22px 0; }
  .col { max-width:820px; margin:0 auto; padding:0 18px; display:flex;
    flex-direction:column; gap:14px; }
  .msg { max-width:86%; padding:10px 14px; border-radius:12px; white-space:pre-wrap;
    overflow-wrap:break-word; }
  .user { align-self:flex-end; background:var(--panel2); border:1px solid var(--line); }
  .bot  { align-self:flex-start; background:var(--panel); border:1px solid var(--line); }
  .meta { align-self:flex-start; font-size:11.5px; color:var(--faint);
    font-family:ui-monospace,Menlo,monospace; margin:-6px 0 2px 4px; }
  .thinking { color:var(--accent); font-family:ui-monospace,Menlo,monospace; }
  .empty { text-align:center; color:var(--faint); margin-top:8vh; }
  .empty svg { width:52px; height:52px; opacity:.9; }

  footer { border-top:1px solid var(--line); padding:12px 18px 16px; }
  .inputrow { max-width:820px; margin:0 auto; display:flex; gap:10px; align-items:flex-end; }
  textarea { flex:1; resize:none; background:var(--panel2); color:var(--ink);
    border:1px solid var(--line); border-radius:10px; padding:10px 14px;
    font:15px/1.5 inherit; min-height:44px; max-height:180px; }
  textarea:focus, select:focus, button:focus-visible { outline:2px solid var(--accent); }
  .hint { max-width:820px; margin:6px auto 0; font-size:11.5px; color:var(--faint);
    display:flex; justify-content:space-between; flex-wrap:wrap; gap:4px; }
  @media (max-width: 760px) { aside { position:absolute; z-index:5; height:100%; } }
</style>
</head>
<body>
<aside id="side">
  <div class="side-top">
    <button id="newchat" type="button">+ New chat</button>
    <button class="ghost" id="newproj" type="button" title="New project">+ 📁</button>
  </div>
  <div class="side-scroll" id="projects"></div>
  <div class="side-note">History lives in this browser (localStorage) — the
  runtime itself stays stateless.</div>
</aside>

<div class="maincol">
<header>
  <button class="ghost" id="toggleside" type="button" title="Toggle sidebar">☰</button>
  <svg viewBox="0 0 64 64" aria-hidden="true">
    <rect x="3" y="3" width="16" height="16" rx="4" fill="none" stroke="#5B6577" stroke-width="4"/>
    <rect x="45" y="3" width="16" height="16" rx="4" fill="none" stroke="#5B6577" stroke-width="4"/>
    <rect x="3" y="24" width="16" height="16" rx="4" fill="none" stroke="#5B6577" stroke-width="4"/>
    <rect x="24" y="24" width="16" height="16" rx="4" fill="none" stroke="#5B6577" stroke-width="4"/>
    <rect x="45" y="24" width="16" height="16" rx="4" fill="none" stroke="#5B6577" stroke-width="4"/>
    <rect x="24" y="45" width="16" height="16" rx="4" fill="none" stroke="#5B6577" stroke-width="4"/>
    <rect x="24" y="3" width="16" height="16" rx="4" fill="#E8A33D"/>
    <rect x="3" y="45" width="16" height="16" rx="4" fill="#E8A33D"/>
    <rect x="45" y="45" width="16" height="16" rx="4" fill="#E8A33D"/>
  </svg>
  <span class="brand mono">sparsify</span>
  <span class="dot" id="dot" title="server status"></span>
  <span class="spacer"></span>
  <label class="set">model
    <select id="model"><option>loading…</option></select>
  </label>
  <label class="set">max tokens
    <input type="number" id="maxtok" value="512" min="16" max="8192" step="16">
  </label>
</header>

<main><div class="col" id="chat"></div></main>

<footer>
  <div class="inputrow">
    <textarea id="box" placeholder="Message… (Enter to send, Shift+Enter for newline)" rows="1"></textarea>
    <button id="send" type="button">Send</button>
  </div>
  <div class="hint mono">
    <span id="status">·</span>
    <span>same API for your apps: POST /v1/chat/completions</span>
  </div>
</footer>
</div>

<script>
const chat = document.getElementById("chat");
const box = document.getElementById("box");
const send = document.getElementById("send");
const modelSel = document.getElementById("model");
const statusEl = document.getElementById("status");
const FRAMES = ["▖","▘","▝","▗","▚","▞"];
let generating = false;

/* ── conversation store (browser-local) ─────────────────────────── */
const KEY = "sparsify.chats.v1";
const uid = () => Math.random().toString(36).slice(2, 10);

function loadState() {
  try {
    const s = JSON.parse(localStorage.getItem(KEY));
    if (s && Array.isArray(s.projects) && s.projects.length) return s;
  } catch (e) {}
  const c = {id: uid(), title: "New chat", history: [], ts: Date.now()};
  return {projects: [{id: uid(), name: "Chats", chats: [c]}], active: c.id};
}
let state = loadState();
const saveState = () => localStorage.setItem(KEY, JSON.stringify(state));

function findChat(id) {
  for (const p of state.projects)
    for (const c of p.chats) if (c.id === id) return [p, c];
  return [null, null];
}
function activeChat() {
  let [, c] = findChat(state.active);
  if (!c) { c = state.projects[0].chats[0]; state.active = c?.id; }
  return c;
}

/* ── sidebar rendering ───────────────────────────────────────────── */
const projectsEl = document.getElementById("projects");
function renderSidebar() {
  projectsEl.innerHTML = "";
  for (const p of state.projects) {
    const wrap = document.createElement("div");
    wrap.className = "proj";
    const head = document.createElement("div");
    head.className = "proj-head";
    head.innerHTML = `<span class="grow"></span>`;
    head.querySelector(".grow").textContent = p.name;
    const addBtn = mkIcon("+", "new chat here", () => newChat(p.id));
    const renBtn = mkIcon("✎", "rename project", () => {
      const n = prompt("Project name", p.name);
      if (n) { p.name = n.trim().slice(0, 40) || p.name; saveState(); renderSidebar(); }
    });
    const delBtn = mkIcon("✕", "delete project", () => {
      if (state.projects.length === 1) { alert("Keep at least one project."); return; }
      if (!confirm(`Delete project "${p.name}" and its ${p.chats.length} chats?`)) return;
      state.projects = state.projects.filter(x => x.id !== p.id);
      if (!findChat(state.active)[1]) state.active = state.projects[0].chats[0]?.id;
      saveState(); renderSidebar(); renderChat();
    });
    head.append(addBtn, renBtn, delBtn);
    wrap.appendChild(head);
    for (const c of p.chats) {
      const item = document.createElement("div");
      item.className = "chat-item" + (c.id === state.active ? " active" : "");
      item.innerHTML = `<span class="grow"></span>`;
      item.querySelector(".grow").textContent = c.title;
      item.onclick = () => { if (!generating) { state.active = c.id; saveState(); renderSidebar(); renderChat(); } };
      const ren = mkIcon("✎", "rename", (e) => {
        e.stopPropagation();
        const n = prompt("Chat title", c.title);
        if (n) { c.title = n.trim().slice(0, 60) || c.title; saveState(); renderSidebar(); }
      });
      const del = mkIcon("✕", "delete chat", (e) => {
        e.stopPropagation();
        if (!confirm(`Delete "${c.title}"?`)) return;
        p.chats = p.chats.filter(x => x.id !== c.id);
        if (!p.chats.length && state.projects.length === 1 && state.projects[0].chats.length === 0)
          newChat(p.id, true);
        if (state.active === c.id) state.active = (p.chats[0] || activeChat())?.id;
        saveState(); renderSidebar(); renderChat();
      });
      item.append(ren, del);
      wrap.appendChild(item);
    }
    projectsEl.appendChild(wrap);
  }
}
function mkIcon(txt, title, fn) {
  const b = document.createElement("button");
  b.className = "iconbtn"; b.textContent = txt; b.title = title; b.type = "button";
  b.onclick = fn;
  return b;
}
function newChat(projectId, silent) {
  const p = state.projects.find(x => x.id === projectId) || state.projects[0];
  const c = {id: uid(), title: "New chat", history: [], ts: Date.now()};
  p.chats.unshift(c);
  state.active = c.id;
  saveState();
  if (!silent) { renderSidebar(); renderChat(); box.focus(); }
}
document.getElementById("newchat").onclick = () => newChat(state.projects[0].id);
document.getElementById("newproj").onclick = () => {
  const n = prompt("Project name", "New project");
  if (!n) return;
  state.projects.push({id: uid(), name: n.trim().slice(0, 40), chats: []});
  saveState(); renderSidebar();
};
document.getElementById("toggleside").onclick = () =>
  document.getElementById("side").classList.toggle("hidden");

/* ── chat rendering ──────────────────────────────────────────────── */
function emptyHero() {
  const d = document.createElement("div");
  d.className = "empty";
  d.innerHTML = `<svg viewBox="0 0 64 64" aria-hidden="true">
    <rect x="3" y="3" width="16" height="16" rx="4" fill="none" stroke="#5B6577" stroke-width="4"/>
    <rect x="45" y="3" width="16" height="16" rx="4" fill="none" stroke="#5B6577" stroke-width="4"/>
    <rect x="3" y="24" width="16" height="16" rx="4" fill="none" stroke="#5B6577" stroke-width="4"/>
    <rect x="24" y="24" width="16" height="16" rx="4" fill="none" stroke="#5B6577" stroke-width="4"/>
    <rect x="45" y="24" width="16" height="16" rx="4" fill="none" stroke="#5B6577" stroke-width="4"/>
    <rect x="24" y="45" width="16" height="16" rx="4" fill="none" stroke="#5B6577" stroke-width="4"/>
    <rect x="24" y="3" width="16" height="16" rx="4" fill="#E8A33D"/>
    <rect x="3" y="45" width="16" height="16" rx="4" fill="#E8A33D"/>
    <rect x="45" y="45" width="16" height="16" rx="4" fill="#E8A33D"/></svg>
    <p>Pick a model and say something.<br>
    <span class="mono" style="font-size:12px">experts page in from SSD as the router needs them</span></p>`;
  return d;
}
function renderChat() {
  chat.innerHTML = "";
  const c = activeChat();
  if (!c || !c.history.length) { chat.appendChild(emptyHero()); return; }
  for (const m of c.history)
    add(m.role === "user" ? "user" : "bot", m.content, true);
  chat.lastElementChild?.scrollIntoView({block: "end"});
}
function add(cls, text, bulk) {
  chat.querySelector(".empty")?.remove();
  const d = document.createElement("div");
  d.className = "msg " + cls;
  d.textContent = text;
  chat.appendChild(d);
  if (!bulk) d.scrollIntoView({block: "end"});
  return d;
}

/* ── server state ────────────────────────────────────────────────── */
async function refresh() {
  try {
    const h = await (await fetch("/health")).json();
    document.getElementById("dot").className = "dot";
    const models = await (await fetch("/v1/models")).json();
    if (models.data && modelSel.options.length !== models.data.length) {
      const current = modelSel.value;
      modelSel.innerHTML = "";
      for (const m of models.data) {
        const o = document.createElement("option");
        o.value = m.id;
        o.textContent = `${m.id.split("/").pop()} (${m.size_gb.toFixed(1)} GB)`;
        modelSel.appendChild(o);
      }
      if (h.loaded) modelSel.value = h.loaded;
      else if (current) modelSel.value = current;
    }
  } catch (e) {
    document.getElementById("dot").className = "dot err";
  }
}
refresh(); setInterval(refresh, 5000);

/* ── send / stream ───────────────────────────────────────────────── */
async function go() {
  const text = box.value.trim();
  if (!text || generating) return;
  const conv = activeChat();
  box.value = ""; autosize();
  add("user", text);
  conv.history.push({role: "user", content: text});
  if (conv.title === "New chat")
    conv.title = text.slice(0, 48) + (text.length > 48 ? "…" : "");
  saveState(); renderSidebar();
  generating = true; send.disabled = true;

  const bot = add("bot", "");
  bot.classList.add("thinking");
  let frame = 0;
  const anim = setInterval(() => {
    if (bot.classList.contains("thinking"))
      bot.textContent = FRAMES[frame++ % FRAMES.length] + " routing experts…";
  }, 140);

  let full = "", stats = null;
  try {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        model: modelSel.value,
        messages: conv.history,
        max_tokens: parseInt(document.getElementById("maxtok").value) || 512,
        stream: true,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => null);
      throw new Error(err?.error?.message || resp.statusText);
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split("\n\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ") || line.includes("[DONE]")) continue;
        const c = JSON.parse(line.slice(6));
        const delta = c.choices?.[0]?.delta?.content;
        if (delta) {
          if (bot.classList.contains("thinking")) {
            bot.classList.remove("thinking"); bot.textContent = "";
          }
          full += delta;
          bot.textContent = full;
          bot.scrollIntoView({block: "end"});
        }
        if (c.sparsify) stats = c.sparsify;
      }
    }
    conv.history.push({role: "assistant", content: full});
    saveState();
    if (stats) {
      const m = document.createElement("div");
      m.className = "meta";
      let line = `${stats.throughput.toFixed(1)} tok/s · rss ${stats.rss_gb.toFixed(2)} GB`;
      if (stats.paging) line += ` · cache ${(stats.paging.hit_rate * 100).toFixed(0)}% hit`;
      m.textContent = line;
      chat.appendChild(m);
      m.scrollIntoView({block: "end"});
    }
  } catch (e) {
    bot.classList.remove("thinking");
    bot.textContent = "⚠ " + e.message;
  } finally {
    clearInterval(anim);
    generating = false; send.disabled = false;
    statusEl.textContent = "·";
    box.focus();
  }
}

send.onclick = go;
box.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); go(); }
});
function autosize() {
  box.style.height = "auto";
  box.style.height = Math.min(box.scrollHeight, 180) + "px";
}
box.addEventListener("input", autosize);

renderSidebar();
renderChat();
box.focus();
</script>
</body>
</html>
"""
