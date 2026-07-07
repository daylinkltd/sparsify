"""Embedded web chat UI served at http://localhost:7777/.

Single self-contained page (no external assets — works offline) matching
the Sparsify brand: slate ground, amber accent, mono-led type. Talks to
the same OpenAI-compatible endpoints third-party clients use.
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
    display:flex; flex-direction:column;
    font:15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .mono { font-family: ui-monospace,"SF Mono",Menlo,Consolas,monospace; }

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
</style>
</head>
<body>
<header>
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
  <button class="ghost" id="clear" type="button">clear</button>
</header>

<main><div class="col" id="chat">
  <div class="empty" id="empty">
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
    <p>Pick a model and say something.<br>
    <span class="mono" style="font-size:12px">experts page in from SSD as the router needs them</span></p>
  </div>
</div></main>

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

<script>
const chat = document.getElementById("chat");
const box = document.getElementById("box");
const send = document.getElementById("send");
const modelSel = document.getElementById("model");
const statusEl = document.getElementById("status");
const FRAMES = ["▖","▘","▝","▗","▚","▞"];
let history = [];
let generating = false;

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

function add(cls, text) {
  document.getElementById("empty")?.remove();
  const d = document.createElement("div");
  d.className = "msg " + cls;
  d.textContent = text;
  chat.appendChild(d);
  d.scrollIntoView({block: "end"});
  return d;
}

async function go() {
  const text = box.value.trim();
  if (!text || generating) return;
  box.value = ""; autosize();
  add("user", text);
  history.push({role: "user", content: text});
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
        messages: history,
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
    history.push({role: "assistant", content: full});
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
document.getElementById("clear").onclick = () => {
  history = [];
  chat.innerHTML = "";
  add("meta", "");
  chat.lastChild.remove();
};
box.focus();
</script>
</body>
</html>
"""
