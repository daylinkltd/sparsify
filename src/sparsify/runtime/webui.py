"""Embedded web chat UI served at http://localhost:7777/.

Single self-contained page (no external assets — works offline) matching
the Sparsify brand: slate ground, amber accent, mono-led type. Talks to
the same OpenAI-compatible endpoints third-party clients use.

Conversations and projects live in the *browser* (localStorage): the
server stays stateless on purpose — it is a runtime, not a database.

Sidebar behavior: on desktop it collapses in-flow (state remembered); on
narrow screens it is an off-canvas overlay with a backdrop, closed by
default. All icons are inline SVG — no emoji, no icon fonts.
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
    --good:#3FB27F; --err:#C0504A; color-scheme: dark;
  }
  @media (prefers-color-scheme: light) {
    :root { --ground:#F2F4F8; --panel:#FFF; --panel2:#E9EDF3; --line:#D8DFEA;
      --ink:#1A2130; --soft:#4C586E; --faint:#8B96A9; --accent:#9A6A14;
      --good:#1D7A52; --err:#A03830; color-scheme: light; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--ground); color:var(--ink); height:100vh;
    height:100dvh; display:flex; overflow:hidden;
    font:15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .mono { font-family: ui-monospace,"SF Mono",Menlo,Consolas,monospace; }
  svg.ic { width:16px; height:16px; display:block; fill:none;
    stroke:currentColor; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }

  /* ── sidebar ── */
  aside { width:260px; min-width:260px; border-right:1px solid var(--line);
    background:var(--panel2); display:flex; flex-direction:column;
    transition:margin-left .18s ease; }
  body.side-closed aside { margin-left:-260px; }
  #backdrop { display:none; }
  @media (max-width: 760px) {
    aside { position:fixed; left:0; top:0; bottom:0; z-index:30;
      margin-left:0; transition:transform .18s ease; }
    body.side-closed aside { margin-left:0; transform:translateX(-100%); }
    body:not(.side-closed) #backdrop { display:block; position:fixed;
      inset:0; background:rgba(0,0,0,.45); z-index:20; }
  }
  .side-top { padding:12px; display:flex; gap:8px; }
  .side-top .primary { flex:1; display:flex; align-items:center; gap:8px; justify-content:center; }
  .side-scroll { flex:1; overflow-y:auto; padding:4px 8px 12px; }
  .proj { margin-top:12px; }
  .proj-head { display:flex; align-items:center; gap:4px; padding:4px 8px;
    font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--faint); }
  .proj-head .grow { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .iconbtn { background:none; border:0; color:var(--faint); cursor:pointer;
    padding:3px; border-radius:5px; opacity:0; display:flex; }
  .iconbtn svg.ic { width:13px; height:13px; }
  .proj-head:hover .iconbtn, .chat-item:hover .iconbtn { opacity:1; }
  .iconbtn:hover { color:var(--accent); background:var(--panel); }
  .chat-item { display:flex; align-items:center; gap:4px; padding:7px 10px;
    border-radius:8px; cursor:pointer; font-size:13.5px; color:var(--soft); }
  .chat-item .grow { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .chat-item:hover { background:var(--panel); }
  .chat-item.active { background:var(--panel); color:var(--ink);
    box-shadow:inset 2px 0 0 var(--accent); }
  .side-note { padding:8px 12px; font-size:10.5px; color:var(--faint);
    border-top:1px solid var(--line); }

  /* ── main column ── */
  .maincol { flex:1; display:flex; flex-direction:column; min-width:0; }
  header { display:flex; align-items:center; gap:10px; padding:10px 16px;
    border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header .logo { width:20px; height:20px; }
  header .brand { font-weight:650; letter-spacing:.02em; }
  header .dot { width:8px; height:8px; border-radius:50%; background:var(--good); }
  header .dot.err { background:var(--err); }
  header .spacer { flex:1; }
  select, input[type=number] {
    background:var(--panel2); color:var(--ink); border:1px solid var(--line);
    border-radius:8px; padding:6px 10px; font-size:13px; max-width:320px;
  }
  label.set { font-size:12px; color:var(--faint); display:flex; gap:6px; align-items:center; }
  button { background:var(--accent); border:0; color:var(--ground);
    font-weight:650; border-radius:8px; padding:8px 14px; cursor:pointer; font-size:14px; }
  button.ghost { background:transparent; color:var(--soft); border:1px solid var(--line); }
  button.ghost:hover { color:var(--ink); border-color:var(--soft); }
  button:disabled { opacity:.45; cursor:default; }
  button.icononly { padding:7px; display:flex; }

  main { flex:1; overflow-y:auto; padding:22px 0; }
  .col { max-width:820px; margin:0 auto; padding:0 18px; display:flex;
    flex-direction:column; gap:14px; }
  .msg { max-width:86%; padding:10px 14px; border-radius:12px;
    overflow-wrap:break-word; }
  .user { align-self:flex-end; background:var(--panel2); border:1px solid var(--line);
    white-space:pre-wrap; }
  .bot  { align-self:flex-start; background:var(--panel); border:1px solid var(--line); }
  .bot > :first-child { margin-top:0; } .bot > :last-child { margin-bottom:0; }
  .bot p { margin:.5em 0; } .bot ul, .bot ol { margin:.4em 0; padding-left:1.4em; }
  .bot h1,.bot h2,.bot h3 { margin:.7em 0 .3em; font-size:1.05em; }
  .bot code { background:var(--panel2); border:1px solid var(--line);
    border-radius:5px; padding:1px 5px; font-size:.9em;
    font-family:ui-monospace,Menlo,monospace; }
  .codeblock { background:var(--ground); border:1px solid var(--line);
    border-radius:10px; margin:.6em 0; overflow:hidden; }
  .codeblock .cb-head { display:flex; justify-content:space-between; align-items:center;
    padding:4px 10px; border-bottom:1px solid var(--line); color:var(--faint);
    font-size:11px; font-family:ui-monospace,Menlo,monospace; }
  .codeblock pre { margin:0; padding:10px 12px; overflow-x:auto;
    font:12.5px/1.55 ui-monospace,Menlo,monospace; }
  .codeblock pre code { background:none; border:0; padding:0; font-size:inherit; }
  .cb-copy { background:none; border:0; color:var(--faint); cursor:pointer;
    display:flex; gap:4px; align-items:center; font-size:11px; padding:2px 4px; }
  .cb-copy:hover { color:var(--accent); }
  .meta { align-self:flex-start; font-size:11.5px; color:var(--faint);
    font-family:ui-monospace,Menlo,monospace; margin:-6px 0 2px 4px; }
  .errmsg { color:var(--err); }
  .thinking { color:var(--accent); font-family:ui-monospace,Menlo,monospace; }
  .empty { text-align:center; color:var(--faint); margin-top:8vh; }
  .empty svg { width:52px; height:52px; opacity:.9; }

  footer { border-top:1px solid var(--line); padding:12px 16px 14px; }
  .inputrow { max-width:820px; margin:0 auto; display:flex; gap:10px; align-items:flex-end; }
  textarea { flex:1; resize:none; background:var(--panel2); color:var(--ink);
    border:1px solid var(--line); border-radius:10px; padding:10px 14px;
    font:15px/1.5 inherit; min-height:44px; max-height:180px; }
  textarea:focus, select:focus, button:focus-visible { outline:2px solid var(--accent); }
  .hint { max-width:820px; margin:6px auto 0; font-size:11.5px; color:var(--faint);
    display:flex; justify-content:space-between; flex-wrap:wrap; gap:4px; }
</style>
</head>
<body>
<div id="backdrop"></div>
<aside id="side">
  <div class="side-top">
    <button id="newchat" type="button" class="primary">
      <svg class="ic" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>
      New chat</button>
    <button class="ghost icononly" id="newproj" type="button" title="New project">
      <svg class="ic" viewBox="0 0 24 24"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M12 11v6M9 14h6"/></svg>
    </button>
  </div>
  <div class="side-scroll" id="projects"></div>
  <div class="side-note">History lives in this browser (localStorage) — the
  runtime itself stays stateless.</div>
</aside>

<div class="maincol">
<header>
  <button class="ghost icononly" id="toggleside" type="button"
          title="Toggle sidebar" aria-label="Toggle sidebar">
    <svg class="ic" viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h16"/></svg>
  </button>
  <svg class="logo" viewBox="0 0 64 64" aria-hidden="true">
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
    <button id="send" type="button" title="Send" aria-label="Send">
      <svg class="ic" viewBox="0 0 24 24" style="width:18px;height:18px">
        <path d="M5 12h13M13 6l6 6-6 6"/></svg>
    </button>
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

/* ── icons ───────────────────────────────────────────────────────── */
const IC = {
  plus:   '<svg class="ic" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
  pencil: '<svg class="ic" viewBox="0 0 24 24"><path d="M17 3l4 4L8 20l-5 1 1-5z"/></svg>',
  trash:  '<svg class="ic" viewBox="0 0 24 24"><path d="M4 7h16M9 7V5h6v2m-8 0 1 13h8l1-13"/></svg>',
  copy:   '<svg class="ic" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a1 1 0 0 1 1-1h9"/></svg>',
  check:  '<svg class="ic" viewBox="0 0 24 24"><path d="M4 13l5 5L20 7"/></svg>',
};

/* ── sidebar open/close ──────────────────────────────────────────── */
const SIDEKEY = "sparsify.side";
const narrow = () => matchMedia("(max-width: 760px)").matches;
function setSide(closed) {
  document.body.classList.toggle("side-closed", closed);
  if (!narrow()) localStorage.setItem(SIDEKEY, closed ? "closed" : "open");
}
setSide(narrow() ? true : localStorage.getItem(SIDEKEY) === "closed");
document.getElementById("toggleside").onclick = () =>
  setSide(!document.body.classList.contains("side-closed"));
document.getElementById("backdrop").onclick = () => setSide(true);
addEventListener("keydown", e => { if (e.key === "Escape" && narrow()) setSide(true); });

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
  if (!c) {
    const p = state.projects.find(x => x.chats.length) || state.projects[0];
    if (!p.chats.length) p.chats.push({id: uid(), title: "New chat", history: [], ts: Date.now()});
    c = p.chats[0]; state.active = c.id;
  }
  return c;
}

/* ── sidebar rendering ───────────────────────────────────────────── */
const projectsEl = document.getElementById("projects");
function mkIcon(svg, title, fn) {
  const b = document.createElement("button");
  b.className = "iconbtn"; b.innerHTML = svg; b.title = title; b.type = "button";
  b.onclick = fn;
  return b;
}
function renderSidebar() {
  projectsEl.innerHTML = "";
  for (const p of state.projects) {
    const wrap = document.createElement("div");
    wrap.className = "proj";
    const head = document.createElement("div");
    head.className = "proj-head";
    head.innerHTML = `<span class="grow"></span>`;
    head.querySelector(".grow").textContent = p.name;
    head.append(
      mkIcon(IC.plus, "new chat here", () => newChat(p.id)),
      mkIcon(IC.pencil, "rename project", () => {
        const n = prompt("Project name", p.name);
        if (n) { p.name = n.trim().slice(0, 40) || p.name; saveState(); renderSidebar(); }
      }),
      mkIcon(IC.trash, "delete project", () => {
        if (state.projects.length === 1) { alert("Keep at least one project."); return; }
        if (!confirm(`Delete project "${p.name}" and its ${p.chats.length} chats?`)) return;
        state.projects = state.projects.filter(x => x.id !== p.id);
        if (!findChat(state.active)[1]) state.active = null;
        activeChat(); saveState(); renderSidebar(); renderChat();
      }),
    );
    wrap.appendChild(head);
    for (const c of p.chats) {
      const item = document.createElement("div");
      item.className = "chat-item" + (c.id === state.active ? " active" : "");
      item.innerHTML = `<span class="grow"></span>`;
      item.querySelector(".grow").textContent = c.title;
      item.onclick = () => {
        if (generating) return;
        state.active = c.id; saveState(); renderSidebar(); renderChat();
        if (narrow()) setSide(true);
      };
      item.append(
        mkIcon(IC.pencil, "rename", (e) => {
          e.stopPropagation();
          const n = prompt("Chat title", c.title);
          if (n) { c.title = n.trim().slice(0, 60) || c.title; saveState(); renderSidebar(); }
        }),
        mkIcon(IC.trash, "delete chat", (e) => {
          e.stopPropagation();
          if (!confirm(`Delete "${c.title}"?`)) return;
          p.chats = p.chats.filter(x => x.id !== c.id);
          if (state.active === c.id) state.active = null;
          activeChat(); saveState(); renderSidebar(); renderChat();
        }),
      );
      wrap.appendChild(item);
    }
    projectsEl.appendChild(wrap);
  }
}
function newChat(projectId) {
  const p = state.projects.find(x => x.id === projectId) || state.projects[0];
  const c = {id: uid(), title: "New chat", history: [], ts: Date.now()};
  p.chats.unshift(c);
  state.active = c.id;
  saveState(); renderSidebar(); renderChat(); box.focus();
}
document.getElementById("newchat").onclick = () => newChat(state.projects[0].id);
document.getElementById("newproj").onclick = () => {
  const n = prompt("Project name", "New project");
  if (!n) return;
  state.projects.push({id: uid(), name: n.trim().slice(0, 40), chats: []});
  saveState(); renderSidebar();
};

/* ── markdown (escape first — no raw HTML ever reaches the DOM) ──── */
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function inline(md) {
  return md
    .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|\W)\*([^*\n]+)\*(?=\W|$)/g, "$1<i>$2</i>")
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,
      '<a href="$2" rel="noopener" target="_blank">$1</a>');
}
function renderMarkdown(el, text) {
  const parts = text.split(/```(\w*)\n?/);
  el.innerHTML = "";
  for (let i = 0; i < parts.length; i += 2) {
    const prose = parts[i];
    if (prose.trim()) {
      const holder = document.createElement("div");
      const lines = esc(prose).split("\n");
      let html = "", list = null;
      const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
      for (const ln of lines) {
        const h = ln.match(/^(#{1,3})\s+(.*)/);
        const ul = ln.match(/^\s*[-*]\s+(.*)/);
        const ol = ln.match(/^\s*\d+[.)]\s+(.*)/);
        if (h) { closeList(); html += `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`; }
        else if (ul) { if (list !== "ul") { closeList(); html += "<ul>"; list = "ul"; } html += `<li>${inline(ul[1])}</li>`; }
        else if (ol) { if (list !== "ol") { closeList(); html += "<ol>"; list = "ol"; } html += `<li>${inline(ol[1])}</li>`; }
        else if (!ln.trim()) { closeList(); }
        else { closeList(); html += `<p>${inline(ln)}</p>`; }
      }
      closeList();
      holder.innerHTML = html;
      while (holder.firstChild) el.appendChild(holder.firstChild);
    }
    if (i + 1 < parts.length) {
      const lang = parts[i + 1] || "code";
      const code = parts[i + 2] ?? "";
      const block = document.createElement("div");
      block.className = "codeblock";
      block.innerHTML = `<div class="cb-head"><span></span>
        <button class="cb-copy" type="button">${IC.copy} copy</button></div>
        <pre><code></code></pre>`;
      block.querySelector(".cb-head span").textContent = lang;
      block.querySelector("code").textContent = code.replace(/\n$/, "");
      block.querySelector(".cb-copy").onclick = (ev) => {
        navigator.clipboard.writeText(code);
        const b = ev.currentTarget;
        b.innerHTML = IC.check + " copied";
        setTimeout(() => { b.innerHTML = IC.copy + " copy"; }, 1200);
      };
      el.appendChild(block);
      i++;  // consumed the code part too
    }
  }
}

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
  for (const m of c.history) add(m.role === "user" ? "user" : "bot", m.content, true);
  chat.lastElementChild?.scrollIntoView({block: "end"});
}
function add(cls, text, bulk) {
  chat.querySelector(".empty")?.remove();
  const d = document.createElement("div");
  d.className = "msg " + cls;
  if (cls === "bot") renderMarkdown(d, text); else d.textContent = text;
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
  statusEl.textContent = "generating…";

  const bot = add("bot", "");
  bot.classList.add("thinking");
  let frame = 0;
  const anim = setInterval(() => {
    if (bot.classList.contains("thinking"))
      bot.textContent = FRAMES[frame++ % FRAMES.length] + " routing experts…";
  }, 140);

  let full = "", stats = null, lastRender = 0;
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
          const now = performance.now();
          if (now - lastRender > 80) {   // markdown re-render, throttled
            renderMarkdown(bot, full); lastRender = now;
            bot.scrollIntoView({block: "end"});
          }
        }
        if (c.sparsify) stats = c.sparsify;
      }
    }
    renderMarkdown(bot, full);
    bot.scrollIntoView({block: "end"});
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
    bot.classList.add("errmsg");
    bot.textContent = "error: " + e.message;
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
