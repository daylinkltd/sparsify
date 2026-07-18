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
  
  /* msg container and alignments */
  .msg-container { display: flex; flex-direction: column; width: 100%; position: relative; margin-bottom: 4px; }
  .user-container { align-items: flex-end; }
  .bot-container { align-items: flex-start; }
  
  .msg { max-width:86%; padding:10px 14px; border-radius:12px;
    overflow-wrap:break-word; }
  .user { background:var(--panel2); border:1px solid var(--line);
    white-space:pre-wrap; }
  .bot  { background:var(--panel); border:1px solid var(--line); }
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
  .toolcard { align-self:flex-start; background:var(--panel2);
    border:1px solid var(--line); border-left:2px solid var(--good);
    border-radius:8px; padding:5px 12px; margin:2px 0; font-size:12.5px;
    color:var(--soft); font-family:ui-monospace,Menlo,monospace; }

  /* ── settings drawer ── */
  .update-pill { background:var(--accent); color:var(--accent-contrast, #0B0E14);
    border:0; border-radius:99px; padding:3px 12px; font-size:12px;
    font-weight:650; cursor:pointer; }
  .update-pill:hover { filter:brightness(1.08); }
  .update-pill.busy { background:var(--panel); color:var(--faint); cursor:default; }

  #settings-backdrop { display:none; position:fixed; inset:0;
    background:rgba(0,0,0,.45); z-index:40; }
  body.settings-open #settings-backdrop { display:block; }
  #settings { position:fixed; top:0; right:0; bottom:0; width:360px;
    max-width:90vw; background:var(--panel2); border-left:1px solid var(--line);
    z-index:41; transform:translateX(100%); transition:transform .18s ease;
    display:flex; flex-direction:column; }
  body.settings-open #settings { transform:translateX(0); }
  .set-head { display:flex; align-items:center; justify-content:space-between;
    padding:14px 16px; border-bottom:1px solid var(--line);
    font-weight:650; letter-spacing:.02em; }
  .set-head .iconbtn { opacity:1; }
  .set-body { padding:16px; overflow-y:auto; display:flex; flex-direction:column; gap:18px; }
  .field { display:flex; flex-direction:column; gap:6px; font-size:13px;
    color:var(--soft); }
  .field .val { color:var(--faint); font-size:11.5px; font-family:ui-monospace,Menlo,monospace; }
  .field textarea, .field input, .field select {
    background:var(--panel); color:var(--ink); border:1px solid var(--line);
    border-radius:8px; padding:8px 10px; font:13px/1.5 inherit; width:100%; }
  .field textarea { resize:vertical; }
  .field input[type=range] { padding:0; accent-color:var(--accent); }
  .set-info { font-size:11.5px; color:var(--faint); line-height:1.7;
    font-family:ui-monospace,Menlo,monospace; border-top:1px solid var(--line);
    padding-top:12px; white-space:pre-wrap; }
  
  /* message actions bar */
  .msg-actions {
    display: flex; gap: 8px; margin: 4px 12px 10px; opacity: 0;
    transition: opacity 0.15s ease, transform 0.15s ease;
    transform: translateY(2px);
  }
  .msg-container:hover .msg-actions { opacity: 1; transform: translateY(0); }
  
  .actionbtn {
    background: none; border: 0; color: var(--faint); cursor: pointer;
    padding: 3px 6px; border-radius: 5px; display: flex; align-items: center; gap: 4px;
    font-size: 11px; transition: color .12s, background .12s;
  }
  .actionbtn:hover { color: var(--accent); background: var(--panel2); }
  .actionbtn svg.ic { width: 12px; height: 12px; }
  
  /* telemetry header */
  .msg-telemetry {
    font-size: 11px; font-family: ui-monospace, SF Mono, Menlo, monospace;
    color: var(--soft); margin: 0 0 6px 12px; opacity: 0.85;
  }
  .msg-telemetry b { color: var(--ink); font-weight: 600; }

  /* table formatting */
  .bot table {
    width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13.5px;
    background: var(--ground); border-radius: 8px; overflow: hidden;
    border: 1px solid var(--line);
  }
  .bot th, .bot td {
    padding: 8px 12px; border: 1px solid var(--line); text-align: left;
  }
  .bot th {
    background: var(--panel2); color: var(--ink); font-weight: 600;
  }
  .bot tr:nth-child(even) { background: rgba(255, 255, 255, 0.02); }
  @media (prefers-color-scheme: light) {
    .bot tr:nth-child(even) { background: rgba(0, 0, 0, 0.015); }
  }
  .bot tr:hover { background: rgba(232, 163, 61, 0.04); }

  .errmsg { color:var(--err); }
  .thinking { color:var(--accent); font-family:ui-monospace,Menlo,monospace; }
  .empty { text-align:center; color:var(--faint); margin-top:8vh; }
  .empty svg { width:52px; height:52px; opacity:.9; }

  footer { border-top:1px solid var(--line); padding:12px 16px 14px; }
  .inputrow { max-width:820px; margin:0 auto; display:flex; gap:10px; align-items:flex-end; }

  /* ── attachments ── */
  .chips { max-width:820px; margin:0 auto 8px; display:none; flex-wrap:wrap; gap:6px; }
  .chips.has { display:flex; }
  .chip { display:flex; align-items:center; gap:6px; background:var(--panel2);
    border:1px solid var(--line); border-radius:8px; padding:4px 8px;
    font-size:12px; color:var(--soft); font-family:ui-monospace,Menlo,monospace; }
  .chip .x { background:none; border:0; color:var(--faint); cursor:pointer;
    padding:0 2px; font-size:13px; line-height:1; }
  .chip .x:hover { color:var(--err); }
  .chip .meta { color:var(--faint); }
  #dropzone { display:none; position:fixed; inset:0; z-index:50;
    background:rgba(11,14,20,.72); border:2px dashed var(--accent);
    align-items:center; justify-content:center; color:var(--accent);
    font:15px ui-monospace,Menlo,monospace; pointer-events:none; }
  body.dragging #dropzone { display:flex; }
  #micbtn.rec { color:var(--err); border-color:var(--err);
    animation:pulse 1.1s ease-in-out infinite; }
  @keyframes pulse { 50% { opacity:.55; } }
  .toast { position:fixed; bottom:86px; left:50%; transform:translateX(-50%);
    background:var(--panel); border:1px solid var(--line); color:var(--soft);
    border-radius:10px; padding:8px 14px; font-size:12.5px; z-index:60;
    max-width:80vw; }
  .oc-snippet { background:var(--panel); border:1px solid var(--line);
    border-radius:8px; padding:8px 10px; font:11px/1.6 ui-monospace,Menlo,monospace;
    white-space:pre; overflow-x:auto; color:var(--soft); }
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
  <button id="updatebtn" type="button" class="update-pill" style="display:none"></button>
  <span class="spacer"></span>
  <label class="set">model
    <select id="model"><option>loading…</option></select>
  </label>
  <button class="ghost" id="toolsbtn" type="button" aria-pressed="false"
          title="Let the model use tools (fetch URLs, search, read/write files, run shell)">
    <svg class="ic" viewBox="0 0 24 24" style="width:14px;height:14px;display:inline-block;vertical-align:-2px">
      <path d="M14 7a4 4 0 0 1-5.3 5.3L4 17v3h3l4.7-4.7A4 4 0 0 1 17 10l3-3-3-3-3 3z"/></svg>
    Tools: off</button>
  <button class="ghost icononly" id="settingsbtn" type="button"
          title="Settings" aria-label="Settings">
    <svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-1-1.5 1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.5-1 1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z"/></svg>
  </button>
</header>

<div id="settings-backdrop"></div>
<aside id="settings">
  <div class="set-head">
    <span>Settings</span>
    <button class="iconbtn" id="settings-close" type="button" aria-label="Close">
      <svg class="ic" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
  </div>
  <div class="set-body">
    <label class="field">System prompt
      <textarea id="set-system" rows="3" placeholder="Optional. Sets the assistant's role/instructions for this browser."></textarea>
    </label>
    <label class="field">Temperature <span class="val" id="set-temp-val">0.0 (greedy)</span>
      <input type="range" id="set-temp" min="0" max="1.5" step="0.05" value="0">
    </label>
    <label class="field">Max tokens per reply
      <input type="number" id="set-maxtok" value="" placeholder="unlimited" min="0" step="128">
    </label>
    <label class="field">Tools
      <select id="set-tools"><option value="on">on</option><option value="off">off</option></select>
    </label>
    <label class="field">Theme
      <select id="set-theme">
        <option value="auto">auto (system)</option>
        <option value="dark">dark</option>
        <option value="light">light</option>
      </select>
    </label>
    <div class="field">Agent &amp; tools
      <div class="set-info" id="set-agent" style="border-top:0;padding-top:2px"></div>
    </div>
    <div class="field">Connect OpenClaw (or any OpenAI-SDK agent)
      <div class="oc-snippet" id="oc-snippet"></div>
      <button class="ghost" id="oc-copy" type="button" style="width:100%;font-size:12px">Copy provider config</button>
      <span class="val">paste into ~/.openclaw/openclaw.json under models.providers — the agent shell runs there, every token runs here</span>
    </div>
    <div class="set-info" id="set-info"></div>
    <button class="ghost" id="set-clearall" type="button" style="width:100%">Delete all chats &amp; projects</button>
  </div>
</aside>

<main><div class="col" id="chat"></div></main>

<div id="dropzone">drop files to attach (text files — content goes into your message)</div>

<footer>
  <div class="chips" id="chips"></div>
  <div class="inputrow">
    <input type="file" id="filein" multiple style="display:none">
    <button class="ghost icononly" id="attachbtn" type="button"
            title="Attach files (or drag &amp; drop) — text files only; vision models are on the roadmap"
            aria-label="Attach files">
      <svg class="ic" viewBox="0 0 24 24"><path d="M21 12.5l-8.5 8.5a5.5 5.5 0 0 1-7.8-7.8L13 5a3.7 3.7 0 0 1 5.2 5.2l-8.2 8.2a1.8 1.8 0 0 1-2.6-2.6L15 8.3"/></svg>
    </button>
    <button class="ghost icononly" id="micbtn" type="button"
            title="Voice input — transcribed locally on this machine (mlx-whisper)"
            aria-label="Voice input">
      <svg class="ic" viewBox="0 0 24 24"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>
    </button>
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

/* ── settings (persisted) ────────────────────────────────────────── */
const SETKEY = "sparsify.settings.v1";
const DEFAULTS = {system: "", temperature: 0, maxTokens: 0, tools: true, theme: "auto"};
let settings = (() => {
  try { return {...DEFAULTS, ...JSON.parse(localStorage.getItem(SETKEY))}; }
  catch (e) { return {...DEFAULTS}; }
})();
const saveSettings = () => localStorage.setItem(SETKEY, JSON.stringify(settings));
let toolsOn = settings.tools;

function applyTheme() {
  const t = settings.theme;
  if (t === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", t);
}
function withSystemPrompt(history) {
  const sys = (settings.system || "").trim();
  return sys ? [{role: "system", content: sys}, ...history] : history;
}

const toolsBtn = document.getElementById("toolsbtn");
function renderToolsBtn() {
  toolsBtn.setAttribute("aria-pressed", toolsOn ? "true" : "false");
  toolsBtn.style.color = toolsOn ? "var(--accent)" : "";
  toolsBtn.style.borderColor = toolsOn ? "var(--accent)" : "";
  toolsBtn.lastChild.textContent = toolsOn ? " Tools: on" : " Tools: off";
}
toolsBtn.onclick = () => {
  toolsOn = !toolsOn; settings.tools = toolsOn; saveSettings(); renderToolsBtn();
  const s = document.getElementById("set-tools"); if (s) s.value = toolsOn ? "on" : "off";
};

/* ── settings drawer ── */
function openSettings(open) { document.body.classList.toggle("settings-open", open); if (open) refreshInfo(); }
document.getElementById("settingsbtn").onclick = () => openSettings(true);
document.getElementById("settings-close").onclick = () => openSettings(false);
document.getElementById("settings-backdrop").onclick = () => openSettings(false);
addEventListener("keydown", e => { if (e.key === "Escape") openSettings(false); });

const $sys = document.getElementById("set-system");
const $temp = document.getElementById("set-temp");
const $tempVal = document.getElementById("set-temp-val");
const $maxtok = document.getElementById("set-maxtok");
const $tools = document.getElementById("set-tools");
const $theme = document.getElementById("set-theme");
function tempLabel(v) { return v <= 0 ? "0.0 (greedy)" : v.toFixed(2); }
function syncSettingsUI() {
  $sys.value = settings.system;
  $temp.value = settings.temperature; $tempVal.textContent = tempLabel(settings.temperature);
  $maxtok.value = settings.maxTokens > 0 ? settings.maxTokens : "";
  $tools.value = settings.tools ? "on" : "off";
  $theme.value = settings.theme;
}
$sys.oninput = () => { settings.system = $sys.value; saveSettings(); };
$temp.oninput = () => { settings.temperature = parseFloat($temp.value) || 0;
  $tempVal.textContent = tempLabel(settings.temperature); saveSettings(); };
$maxtok.oninput = () => { settings.maxTokens = parseInt($maxtok.value) || 0; saveSettings(); };
$tools.onchange = () => { toolsOn = settings.tools = ($tools.value === "on"); saveSettings(); renderToolsBtn(); };
$theme.onchange = () => { settings.theme = $theme.value; saveSettings(); applyTheme(); };
document.getElementById("set-clearall").onclick = () => {
  if (!confirm("Delete ALL chats and projects in this browser?")) return;
  localStorage.removeItem(KEY); location.reload();
};

function ocSnippet(modelId, contextWindow) {
  const note = contextWindow ? "" :
    "  // no model loaded yet — load one, reopen Settings, and re-copy\n";
  return `${note}"sparsify": {
  "baseUrl": "${location.origin}/v1",
  "apiKey": "sparsify-local",
  "api": "openai-completions",
  "models": [{ "id": "${modelId}",
    "name": "Sparsify (paged, local)",
    "input": ["text"], "contextWindow": ${contextWindow || 8192},
    "maxTokens": 8192,
    "cost": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0} }]
}`;
}
async function refreshInfo() {
  const info = document.getElementById("set-info");
  const agent = document.getElementById("set-agent");
  try {
    const [h, t] = await Promise.all([
      (await fetch("/health")).json(),
      (await fetch("/v1/tools")).json(),
    ]);
    document.getElementById("oc-snippet").textContent =
      ocSnippet(modelSel.value || h.loaded || "<model id>", h.safe_context_tokens);
    const tiers = (t.tiers_enabled || []).join(" · ") || "read";
    let a = `tiers enabled: ${tiers}\n`;
    if (t.workspace) a += `workspace: ${t.workspace}\n`;
    a += `tier grants are a server startup decision, not a browser toggle:\n`
       + `  sparsify serve --shell --browser\n`
       + `(deliberate — a webpage reaching localhost must not grant itself shell)`;
    if (h.supports_tools === false)
      a += `\nnote: the loaded model has no tool template — tools are ignored`;
    agent.textContent = a;
    let s = `server: ${h.loaded || "no model loaded"}\n`;
    if (h.context_limit) {
      s += `context: ${h.safe_context_tokens.toLocaleString()} tokens safe on this machine's free RAM right now`;
      if (h.safe_context_tokens < h.context_limit)
        s += ` (model supports ${h.context_limit.toLocaleString()} architecturally — more RAM free, more usable context)`;
      s += "\n";
    }
    if (h.stats) s += `cache: ${(h.stats.hit_rate*100).toFixed(0)}% hit · ${(h.stats.resident_bytes/1e9).toFixed(1)}/${(h.stats.budget_bytes/1e9).toFixed(1)} GB\n`;
    info.textContent = s.trim();
  } catch (e) {
    document.getElementById("oc-snippet").textContent = ocSnippet(modelSel.value || "<model id>");
    info.textContent = "server unreachable"; agent.textContent = "";
  }
}
document.getElementById("oc-copy").onclick = (ev) => {
  navigator.clipboard.writeText(document.getElementById("oc-snippet").textContent);
  const b = ev.currentTarget; const old = b.textContent;
  b.textContent = "Copied"; setTimeout(() => { b.textContent = old; }, 1200);
};

applyTheme(); syncSettingsUI(); renderToolsBtn();

/* ── icons ───────────────────────────────────────────────────────── */
const IC = {
  plus:   '<svg class="ic" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
  pencil: '<svg class="ic" viewBox="0 0 24 24"><path d="M17 3l4 4L8 20l-5 1 1-5z"/></svg>',
  trash:  '<svg class="ic" viewBox="0 0 24 24"><path d="M4 7h16M9 7V5h6v2m-8 0 1 13h8l1-13"/></svg>',
  copy:   '<svg class="ic" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a1 1 0 0 1 1-1h9"/></svg>',
  check:  '<svg class="ic" viewBox="0 0 24 24"><path d="M4 13l5 5L20 7"/></svg>',
  reload: '<svg class="ic" viewBox="0 0 24 24"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l.73-.73"/></svg>',
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
      
      let k = 0;
      while (k < lines.length) {
        const ln = lines[k];
        
        // 1. Table Parser
        if (ln.includes("|") && k + 1 < lines.length) {
          const nextLn = lines[k + 1];
          const isDelimiter = nextLn.includes("|") && /^[|:\-\s]+$/.test(nextLn.trim()) && nextLn.includes("-");
          if (isDelimiter) {
            closeList();
            
            const parseRow = (rowText) => {
              let trimmed = rowText.trim();
              if (trimmed.startsWith("|")) trimmed = trimmed.slice(1);
              if (trimmed.endsWith("|")) trimmed = trimmed.slice(0, -1);
              return trimmed.split("|").map(s => s.trim());
            };
            
            const headers = parseRow(ln);
            const delimCols = parseRow(nextLn);
            const aligns = delimCols.map(col => {
              const left = col.startsWith(":");
              const right = col.endsWith(":");
              if (left && right) return "center";
              if (right) return "right";
              if (left) return "left";
              return "";
            });
            
            let tableHTML = "<table><thead><tr>";
            for (let colIdx = 0; colIdx < headers.length; colIdx++) {
              const alignStyle = aligns[colIdx] ? ` style="text-align:${aligns[colIdx]}"` : "";
              tableHTML += `<th${alignStyle}>${inline(headers[colIdx])}</th>`;
            }
            tableHTML += "</tr></thead><tbody>";
            
            k += 2; // skip header and delimiter lines
            while (k < lines.length && lines[k].includes("|")) {
              const rowLn = lines[k];
              if (/^[|:\-\s]+$/.test(rowLn.trim())) break; // safety
              
              const cells = parseRow(rowLn);
              tableHTML += "<tr>";
              for (let colIdx = 0; colIdx < headers.length; colIdx++) {
                const alignStyle = aligns[colIdx] ? ` style="text-align:${aligns[colIdx]}"` : "";
                const cellVal = cells[colIdx] !== undefined ? cells[colIdx] : "";
                tableHTML += `<td${alignStyle}>${inline(cellVal)}</td>`;
              }
              tableHTML += "</tr>";
              k++;
            }
            
            tableHTML += "</tbody></table>";
            html += tableHTML;
            continue;
          }
        }
        
        const h = ln.match(/^(#{1,3})\s+(.*)/);
        const ul = ln.match(/^\s*[-*]\s+(.*)/);
        const ol = ln.match(/^\s*\d+[.)]\s+(.*)/);
        if (h) { closeList(); html += `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`; }
        else if (ul) { if (list !== "ul") { closeList(); html += "<ul>"; list = "ul"; } html += `<li>${inline(ul[1])}</li>`; }
        else if (ol) { if (list !== "ol") { closeList(); html += "<ol>"; list = "ol"; } html += `<li>${inline(ol[1])}</li>`; }
        else if (!ln.trim()) { closeList(); }
        else { closeList(); html += `<p>${inline(ln)}</p>`; }
        k++;
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

function formatTelemetry(stats) {
  let line = `<b>${stats.n_tokens || 0}</b> tokens · <b>${(stats.throughput || 0).toFixed(1)}</b> tok/s`;
  if (stats.rss_gb) line += ` · rss <b>${stats.rss_gb.toFixed(2)}</b> GB`;
  else if (stats.active_gb) line += ` · active <b>${stats.active_gb.toFixed(2)}</b> GB`;
  
  if (stats.paging) {
    line += ` · cache <b>${(stats.paging.hit_rate * 100).toFixed(0)}%</b> hit`;
  }
  return line;
}

function editMessage(msgContainer, index) {
  if (generating) return;
  const conv = activeChat();
  const originalText = conv.history[index].content;
  const msgBody = msgContainer.querySelector(".msg");
  const oldHTML = msgBody.innerHTML;
  
  // Hide actions and telemetry
  msgContainer.querySelector(".msg-actions").style.display = "none";
  const tel = msgContainer.querySelector(".msg-telemetry");
  if (tel) tel.style.display = "none";
  
  msgBody.innerHTML = "";
  msgBody.className = "msg user editing";
  
  const ta = document.createElement("textarea");
  ta.style.width = "100%";
  ta.style.background = "var(--ground)";
  ta.style.color = "var(--ink)";
  ta.style.border = "1px solid var(--line)";
  ta.style.borderRadius = "8px";
  ta.style.padding = "8px";
  ta.style.font = "inherit";
  ta.style.resize = "vertical";
  ta.style.minHeight = "60px";
  ta.value = originalText;
  msgBody.appendChild(ta);
  
  const btnRow = document.createElement("div");
  btnRow.style.display = "flex";
  btnRow.style.gap = "8px";
  btnRow.style.marginTop = "6px";
  btnRow.style.justifyContent = "flex-end";
  
  const cancelBtn = document.createElement("button");
  cancelBtn.className = "ghost";
  cancelBtn.style.padding = "4px 8px";
  cancelBtn.style.fontSize = "12px";
  cancelBtn.textContent = "Cancel";
  cancelBtn.onclick = (e) => {
    e.stopPropagation();
    msgBody.innerHTML = "";
    msgBody.className = "msg user";
    msgBody.innerHTML = oldHTML;
    msgContainer.querySelector(".msg-actions").style.display = "flex";
    if (tel) tel.style.display = "block";
  };
  
  const saveBtn = document.createElement("button");
  saveBtn.style.padding = "4px 8px";
  saveBtn.style.fontSize = "12px";
  saveBtn.textContent = "Save & Submit";
  saveBtn.onclick = (e) => {
    e.stopPropagation();
    const val = ta.value.trim();
    if (val) {
      submitEdit(index, val);
    }
  };
  
  btnRow.appendChild(cancelBtn);
  btnRow.appendChild(saveBtn);
  msgBody.appendChild(btnRow);
  ta.focus();
}

function submitEdit(index, newText) {
  const conv = activeChat();
  conv.history = conv.history.slice(0, index);
  saveState();
  renderChat();
  box.value = newText;
  go();
}

function regenerateLast() {
  if (generating) return;
  const conv = activeChat();
  if (conv.history.length < 2) return;
  
  const lastMsg = conv.history[conv.history.length - 1];
  if (lastMsg.role !== "assistant") return;
  
  conv.history.pop(); // Remove last assistant response
  
  const userMsg = conv.history[conv.history.length - 1];
  if (userMsg.role !== "user") return;
  
  conv.history.pop(); // Remove last user prompt
  
  saveState();
  renderChat();
  box.value = userMsg.content;
  go();
}

function renderChat() {
  chat.innerHTML = "";
  const c = activeChat();
  if (!c || !c.history.length) { chat.appendChild(emptyHero()); return; }
  c.history.forEach((m, idx) => {
    add(m.role === "user" ? "user" : "bot", m.content, true, idx, m.sparsify);
  });
  chat.lastElementChild?.scrollIntoView({block: "end"});
}

function add(cls, text, bulk, index, stats) {
  chat.querySelector(".empty")?.remove();
  
  const container = document.createElement("div");
  container.className = `msg-container ${cls}-container`;
  if (index !== undefined) {
    container.dataset.index = index;
  }
  
  // Create telemetry header for bot messages
  if (cls === "bot") {
    const telEl = document.createElement("div");
    telEl.className = "msg-telemetry";
    if (stats) {
      telEl.innerHTML = formatTelemetry(stats);
    }
    container.appendChild(telEl);
  }
  
  const d = document.createElement("div");
  d.className = "msg " + cls;
  if (cls === "bot") renderMarkdown(d, text); else d.textContent = text;
  container.appendChild(d);
  
  // Create actions toolbar
  const actions = document.createElement("div");
  actions.className = "msg-actions";
  
  // Copy button
  const copyBtn = document.createElement("button");
  copyBtn.className = "actionbtn";
  copyBtn.innerHTML = `${IC.copy} Copy`;
  copyBtn.onclick = (e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text);
    copyBtn.innerHTML = `${IC.check} Copied`;
    setTimeout(() => { copyBtn.innerHTML = `${IC.copy} Copy`; }, 1200);
  };
  actions.appendChild(copyBtn);
  
  // Edit button for user messages
  if (cls === "user" && index !== undefined) {
    const editBtn = document.createElement("button");
    editBtn.className = "actionbtn";
    editBtn.innerHTML = `${IC.pencil} Edit`;
    editBtn.onclick = (e) => {
      e.stopPropagation();
      editMessage(container, index);
    };
    actions.appendChild(editBtn);
  }
  
  // Regenerate button for the last bot message
  if (cls === "bot" && index !== undefined) {
    const conv = activeChat();
    if (index === conv.history.length - 1) {
      const regBtn = document.createElement("button");
      regBtn.className = "actionbtn";
      regBtn.innerHTML = `${IC.reload} Regenerate`;
      regBtn.onclick = (e) => {
        e.stopPropagation();
        regenerateLast();
      };
      actions.appendChild(regBtn);
    }
  }
  
  container.appendChild(actions);
  chat.appendChild(container);
  
  if (!bulk) container.scrollIntoView({block: "end"});
  return d; // Return actual message bubble element for streaming compatibility
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
        const prefix = m.available ? "" : "☁️ [Pullable] ";
        o.textContent = `${prefix}${m.id.split("/").pop()} (${m.size_gb.toFixed(1)} GB)`;
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

/* ── update check + button ──────────────────────────────────────── */
const updateBtn = document.getElementById("updatebtn");
let updating = false;
async function checkUpdate() {
  if (updating) return;
  try {
    const v = await (await fetch("/version")).json();
    if (v.update_available) {
      updateBtn.style.display = "";
      updateBtn.textContent = "Update available";
      updateBtn.title = `${v.current} → ${v.latest} — click to update`;
    } else {
      updateBtn.style.display = "none";
    }
  } catch (e) { /* offline: leave as-is */ }
}
updateBtn.onclick = async () => {
  if (updating) return;
  if (!confirm("Update Sparsify now? The server will restart and this page "
             + "will reconnect in a few seconds.")) return;
  updating = true;
  updateBtn.classList.add("busy");
  updateBtn.textContent = "Updating…";
  try { await fetch("/admin/update", {method: "POST"}); } catch (e) {}
  // poll until the server comes back on a new version
  const before = Date.now();
  const poll = setInterval(async () => {
    try {
      const v = await (await fetch("/version")).json();
      if (!v.update_available || Date.now() - before > 120000) {
        clearInterval(poll); updating = false;
        updateBtn.classList.remove("busy");
        updateBtn.textContent = "Updated — reload";
        updateBtn.onclick = () => location.reload();
      }
    } catch (e) { /* server mid-restart */ }
  }, 3000);
};
checkUpdate(); setInterval(checkUpdate, 6 * 3600 * 1000);

/* ── send / stream ───────────────────────────────────────────────── */
async function go() {
  let text = box.value.trim();
  if ((!text && !attachments.length) || generating) return;
  text = consumeAttachments(text);
  const conv = activeChat();
  box.value = ""; autosize();
  
  // Add the user message block to UI and history
  const userIndex = conv.history.length;
  add("user", text, false, userIndex);
  conv.history.push({role: "user", content: text});
  
  if (conv.title === "New chat")
    conv.title = text.slice(0, 48) + (text.length > 48 ? "…" : "");
  saveState(); renderSidebar();
  generating = true; send.disabled = true;
  statusEl.textContent = "generating…";

  // Bot bubble index is userIndex + 1
  const botIndex = userIndex + 1;
  const bot = add("bot", "", false, botIndex);
  const botContainer = bot.parentElement;
  const telHeader = botContainer.querySelector(".msg-telemetry");
  
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
        messages: withSystemPrompt(conv.history),
        ...(settings.maxTokens > 0 ? {max_tokens: settings.maxTokens} : {}),
        ...(settings.temperature > 0 ? {temperature: settings.temperature} : {}),
        ...(toolsOn ? {tools: "auto"} : {}),
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
            botContainer.scrollIntoView({block: "end"});
          }
        }
        if (c.sparsify) {
          stats = c.sparsify;
          if (telHeader) {
            telHeader.innerHTML = formatTelemetry(stats);
          }
        }
        if (c.sparsify_tool) {
          if (bot.classList.contains("thinking")) {
            bot.classList.remove("thinking"); bot.textContent = "";
          }
          const t = c.sparsify_tool;
          const args = Object.entries(t.arguments || {})
            .map(([k, v]) => `${k}=${v}`).join(", ");
          const card = document.createElement("div");
          card.className = "toolcard";
          card.innerHTML = `<svg class="ic" viewBox="0 0 24 24" style="width:12px;height:12px;display:inline-block;vertical-align:-2px"><path d="M14 7a4 4 0 0 1-5.3 5.3L4 17v3h3l4.7-4.7A4 4 0 0 1 17 10l3-3-3-3-3 3z"/></svg> <span></span>`;
          card.querySelector("span").textContent = `${t.name}(${args})`;
          botContainer.insertBefore(card, bot);
          botContainer.scrollIntoView({block: "end"});
        }
      }
    }
    renderMarkdown(bot, full);
    botContainer.scrollIntoView({block: "end"});
    
    // Save to history with stats
    conv.history.push({role: "assistant", content: full, sparsify: stats});
    saveState();
    
    // Trigger render sidebar to make sure last messages have correct action triggers (e.g. Regenerate button)
    renderChat();
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

/* ── attachments: files + drag & drop ────────────────────────────── */
const chipsEl = document.getElementById("chips");
const fileIn = document.getElementById("filein");
let attachments = [];               // {name, content, truncated}
const ATTACH_MAX = 48 * 1024;       // per-file cap fed into the prompt
const IMG_EXT = /\.(png|jpe?g|gif|webp|heic|bmp|tiff?|svg)$/i;

function toast(msg, ms = 4200) {
  const t = document.createElement("div");
  t.className = "toast"; t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), ms);
}
function renderChips() {
  chipsEl.innerHTML = "";
  chipsEl.classList.toggle("has", attachments.length > 0);
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    const kb = (a.content.length / 1024).toFixed(1);
    c.innerHTML = `<span class="name"></span><span class="meta"></span>
      <button class="x" type="button" title="remove">×</button>`;
    c.querySelector(".name").textContent = a.name;
    c.querySelector(".meta").textContent = `${kb} KB${a.truncated ? " (truncated)" : ""}`;
    c.querySelector(".x").onclick = () => { attachments.splice(i, 1); renderChips(); };
    chipsEl.appendChild(c);
  });
}
async function addFiles(files) {
  for (const f of files) {
    if (IMG_EXT.test(f.name) || (f.type || "").startsWith("image/")) {
      toast(`${f.name}: images need a vision model — Sparsify runs text ` +
            `models today (mlx-vlm is on the roadmap, honestly not built yet)`);
      continue;
    }
    if ((f.type || "").startsWith("audio/") || (f.type || "").startsWith("video/")) {
      toast(`${f.name}: audio/video attachments aren't supported — use the mic for voice input`);
      continue;
    }
    if (f.size > 4 * 1024 * 1024) { toast(`${f.name}: too large (4 MB max)`); continue; }
    const raw = await f.text();
    if (raw.includes("\u0000")) { toast(`${f.name}: binary file — only text files can go into the prompt`); continue; }
    attachments.push({name: f.name, content: raw.slice(0, ATTACH_MAX),
                      truncated: raw.length > ATTACH_MAX});
  }
  renderChips();
}
document.getElementById("attachbtn").onclick = () => fileIn.click();
fileIn.onchange = () => { addFiles([...fileIn.files]); fileIn.value = ""; };

let dragDepth = 0;
addEventListener("dragenter", e => {
  if (e.dataTransfer?.types?.includes("Files")) {
    e.preventDefault(); dragDepth++;
    document.body.classList.add("dragging");
  }
});
addEventListener("dragover", e => {
  if (e.dataTransfer?.types?.includes("Files")) e.preventDefault();
});
addEventListener("dragleave", () => {
  if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dragging"); }
});
addEventListener("drop", e => {
  e.preventDefault(); dragDepth = 0;
  document.body.classList.remove("dragging");
  if (e.dataTransfer?.files?.length) addFiles([...e.dataTransfer.files]);
});

function consumeAttachments(text) {
  if (!attachments.length) return text;
  const blocks = attachments.map(a =>
    `[attached file: ${a.name}${a.truncated ? " — truncated to 48 KB" : ""}]\n` +
    "```\n" + a.content + "\n```").join("\n\n");
  attachments = []; renderChips();
  return blocks + (text ? "\n\n" + text : "");
}

/* ── voice input: recorded here, transcribed locally by the server ── */
const micBtn = document.getElementById("micbtn");
let rec = null;   // {stream, ctx, node, chunks:[Float32Array], rate}

function encodeWav16k(chunks, rate) {
  let n = 0; for (const c of chunks) n += c.length;
  const all = new Float32Array(n);
  let off = 0; for (const c of chunks) { all.set(c, off); off += c.length; }
  // linear resample to 16 kHz mono
  const ratio = rate / 16000, outN = Math.floor(all.length / ratio);
  const pcm = new Int16Array(outN);
  for (let i = 0; i < outN; i++) {
    const x = i * ratio, i0 = Math.floor(x), t = x - i0;
    const s = all[i0] * (1 - t) + (all[Math.min(i0 + 1, all.length - 1)] || 0) * t;
    pcm[i] = Math.max(-1, Math.min(1, s)) * 32767;
  }
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, 16000, true);
  v.setUint32(28, 32000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, pcm.length * 2, true);
  new Int16Array(buf, 44).set(pcm);
  return new Blob([buf], {type: "audio/wav"});
}
async function micStart() {
  const stream = await navigator.mediaDevices.getUserMedia({audio: true});
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(stream);
  const node = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  node.onaudioprocess = e => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(node); node.connect(ctx.destination);
  rec = {stream, ctx, node, chunks, rate: ctx.sampleRate};
  micBtn.classList.add("rec"); micBtn.title = "Stop recording";
  statusEl.textContent = "recording… click mic to stop";
}
async function micStop() {
  const {stream, ctx, node, chunks, rate} = rec; rec = null;
  node.disconnect(); stream.getTracks().forEach(t => t.stop()); ctx.close();
  micBtn.classList.remove("rec");
  micBtn.title = "Voice input — transcribed locally on this machine (mlx-whisper)";
  if (!chunks.length) { statusEl.textContent = "·"; return; }
  statusEl.textContent = "transcribing locally…";
  try {
    const fd = new FormData();
    fd.append("file", encodeWav16k(chunks, rate), "speech.wav");
    const r = await fetch("/v1/audio/transcriptions", {method: "POST", body: fd});
    const j = await r.json();
    if (!r.ok) throw new Error(j?.error?.message || r.statusText);
    if (j.text) {
      box.value = (box.value ? box.value + " " : "") + j.text;
      autosize(); box.focus();
    } else toast("nothing transcribed — try speaking closer to the mic");
  } catch (e) {
    toast("voice: " + e.message, 6000);
  } finally { statusEl.textContent = "·"; }
}
micBtn.onclick = async () => {
  try { rec ? await micStop() : await micStart(); }
  catch (e) { toast("microphone: " + e.message); micBtn.classList.remove("rec"); rec = null; }
};

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
