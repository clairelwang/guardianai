// shared/assistant.js
// Auto-injects a simple chatbot widget on every page that includes this script.

(function () {
  // =========================
  // CONFIG (edit these 2)
  // =========================
  const HF_API_BASE = "https://anshvkapadia-guardianai-backend.hf.space"; // <-- change this
  const ASSISTANT_ENDPOINT = `${HF_API_BASE}/api/assistant`;

  // Optional: try to pull "page context" from these containers (first match wins)
  const CONTEXT_SELECTORS = ["#lessonCard", "main", "#content", "body"];

  // How much page text to send as context (keep bounded)
  const MAX_CONTEXT_CHARS = 3500;

  // How many past messages to include in each request
  const MAX_HISTORY_MESSAGES = 12;

  // Storage key (shared across all your pages)
  const STORAGE_KEY = "gs_chat_history_v1";

  // =========================
  // Helpers
  // =========================
  function $(sel, root = document) { return root.querySelector(sel); }

  function escapeHtml(str) {
    return String(str)
      .replaceAll("&","&amp;")
      .replaceAll("<","&lt;")
      .replaceAll(">","&gt;")
      .replaceAll('"',"&quot;")
      .replaceAll("'","&#039;");
  }

  function getPageContext() {
    for (const sel of CONTEXT_SELECTORS) {
      const el = $(sel);
      if (el && el.innerText && el.innerText.trim().length > 40) {
        return el.innerText.trim().slice(0, MAX_CONTEXT_CHARS);
      }
    }
    return "";
  }

  function loadHistory() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      const obj = raw ? JSON.parse(raw) : {};
      // store per-path so each page has its own thread
      return obj[location.pathname] || [];
    } catch {
      return [];
    }
  }

  function saveHistory(history) {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      const obj = raw ? JSON.parse(raw) : {};
      obj[location.pathname] = history;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(obj));
    } catch {}
  }

  async function callAssistant({ userText, history }) {
    // Build a single "text" prompt compatible with your current backend
    // (no backend changes required).
    const context = getPageContext();
    const trimmedHistory = history.slice(-MAX_HISTORY_MESSAGES);

    const convo = trimmedHistory.map(m => `${m.role.toUpperCase()}: ${m.content}`).join("\n");
    const payloadText =
      `PAGE CONTEXT (may be partial):\n${context}\n\n` +
      `CONVERSATION:\n${convo}\n\n` +
      `USER: ${userText}\n\n` +
      `ASSISTANT:`;

    const r = await fetch(ASSISTANT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: payloadText })
    });

    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data?.error || `Assistant error (${r.status})`);
    return data.reply || "";
  }

  // =========================
  // UI injection
  // =========================
  function injectStyles() {
    if ($("#gs-chat-style")) return;

    const style = document.createElement("style");
    style.id = "gs-chat-style";
    style.textContent = `
      .gsChatFab{
        position:fixed; right:18px; bottom:18px; z-index:99999;
        border:none; cursor:pointer;
        padding:12px 14px; border-radius:999px;
        font-weight:900; font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;

        /* purple look */
        background:#7c3aed;
        color:#fff;

        /* glow */
        box-shadow:
          0 14px 36px rgba(124,58,237,.35),
          0 0 0 6px rgba(124,58,237,.14);

        display:flex; align-items:center; gap:10px;
        transition: transform .12s ease, filter .12s ease;
      }
      .gsChatFab:hover{
        transform: translateY(-1px);
        filter: brightness(1.03);
        box-shadow:
          0 18px 46px rgba(124,58,237,.42),
          0 0 0 7px rgba(124,58,237,.16);
      }
      .gsChatFab:active{
        transform: translateY(0px) scale(.99);
      }
      .gsChatFab:focus{
        outline:none;
      }
      .gsChatFab:focus-visible{
        box-shadow:
          0 18px 46px rgba(124,58,237,.42),
          0 0 0 8px rgba(124,58,237,.22);
      }
      .gsChatWrap{
        position:fixed; right:18px; bottom:72px; z-index:99999;
        width:min(380px, calc(100vw - 36px));
        border-radius:18px;
        border:1px solid rgba(15,23,42,.12);
        background:#fff;
        box-shadow:0 24px 70px rgba(15,23,42,.22);
        overflow:hidden;
        display:none;
        font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      }
      .gsChatHead{
        background:#f8fafc;
        border-bottom:1px solid rgba(15,23,42,.10);
        padding:12px 12px;
        display:flex; align-items:center; justify-content:space-between; gap:12px;
      }
      .gsChatTitle{
        font-weight:950; font-size:13px; color:#0f172a;
        display:flex; align-items:center; gap:10px;
      }
      .gsChatTitle .dot{
        width:26px; height:26px; border-radius:10px;
        background:#7c3aed; color:#fff;
        display:grid; place-items:center; font-weight:950;
      }
      .gsChatClose{
        border:none; background:#eef2f7; color:#0f172a;
        padding:8px 10px; border-radius:12px;
        cursor:pointer; font-weight:950; font-size:12px;
      }
      .gsChatBody{
        max-height: min(60vh, 430px);
        overflow:auto;
        padding:12px;
        background:#ffffff;
      }
      .gsMsg{
        margin:8px 0;
        display:flex;
      }
      .gsMsg.user{ justify-content:flex-end; }
      .gsMsg.bot{ justify-content:flex-start; }
      .gsBubble{
        max-width: 86%;
        padding:10px 12px;
        border-radius:14px;
        font-size:13px;
        line-height:1.45;
        font-weight:650;
        white-space:pre-wrap;
      }
      .gsMsg.user .gsBubble{
        background:#0f172a; color:#fff;
        border-top-right-radius:6px;
      }
      .gsMsg.bot .gsBubble{
        background:#f1f5f9; color:#0f172a;
        border-top-left-radius:6px;
        border:1px solid rgba(15,23,42,.08);
      }
      .gsChatFoot{
        border-top:1px solid rgba(15,23,42,.10);
        padding:10px;
        background:#fff;
      }
      .gsRow{
        display:flex; gap:8px; align-items:flex-end;
      }
      .gsInput{
        flex:1;
        resize:none;
        min-height:40px; max-height:110px;
        padding:10px 10px;
        border-radius:12px;
        border:1px solid rgba(15,23,42,.14);
        outline:none;
        font-size:13px;
        font-weight:650;
      }
      .gsSend{
        border:none; cursor:pointer;
        padding:10px 12px;
        border-radius:12px;
        background:#7c3aed; color:#fff;
        font-weight:950; font-size:13px;
      }
      .gsSend:disabled{
        opacity:.55; cursor:not-allowed;
      }
      .gsHint{
        margin-top:6px;
        font-size:11px;
        color:#64748b;
        font-weight:700;
      }
      .gsClear{
        margin-left:auto;
        border:none; background:#f3f4f6; color:#0f172a;
        padding:8px 10px; border-radius:12px;
        cursor:pointer; font-weight:950; font-size:12px;
        border:1px solid #e5e7eb;
      }
    `;
    document.head.appendChild(style);
  }

  function injectWidget() {
    if ($("#gsChatWrap")) return;

    const fab = document.createElement("button");
    fab.className = "gsChatFab";
    fab.id = "gsChatFab";
    fab.type = "button";
    fab.innerHTML = `💬 <span>Ask Guardian AI</span>`;

    const wrap = document.createElement("div");
    wrap.className = "gsChatWrap";
    wrap.id = "gsChatWrap";
    wrap.innerHTML = `
      <div class="gsChatHead">
        <div class="gsChatTitle">
          <div class="dot">✨</div>
          <span>AI Learning Assistant</span>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <button class="gsClear" id="gsClearChat" type="button">Clear</button>
          <button class="gsChatClose" id="gsChatClose" type="button">Close</button>
        </div>
      </div>

      <div class="gsChatBody" id="gsChatBody"></div>

      <div class="gsChatFoot">
        <div class="gsRow">
          <textarea class="gsInput" id="gsChatInput" placeholder="Type a question…" rows="1"></textarea>
          <button class="gsSend" id="gsChatSend" type="button">Send</button>
        </div>
        <div class="gsHint">Tip: Ask “Explain Lesson 2 in simpler terms” or “What are the red flags here?”</div>
      </div>
    `;

    document.body.appendChild(wrap);
    document.body.appendChild(fab);

    return { fab, wrap };
  }

  function renderHistory(bodyEl, history) {
    bodyEl.innerHTML = history.map(m => {
      const cls = m.role === "user" ? "user" : "bot";
      return `
        <div class="gsMsg ${cls}">
          <div class="gsBubble">${escapeHtml(m.content)}</div>
        </div>
      `;
    }).join("");
    bodyEl.scrollTop = bodyEl.scrollHeight;
  }

  // =========================
  // Main init
  // =========================
  function init() {
    injectStyles();
    const ui = injectWidget();
    if (!ui) return;

    const fab = $("#gsChatFab");
    const wrap = $("#gsChatWrap");
    const closeBtn = $("#gsChatClose");
    const clearBtn = $("#gsClearChat");
    const body = $("#gsChatBody");
    const input = $("#gsChatInput");
    const send = $("#gsChatSend");

    let history = loadHistory();
    renderHistory(body, history);

    function open() { wrap.style.display = "block"; input.focus(); }
    function close() { wrap.style.display = "none"; }
    function toggle() { (wrap.style.display === "block") ? close() : open(); }

    fab.addEventListener("click", toggle);
    closeBtn.addEventListener("click", close);

    clearBtn.addEventListener("click", () => {
      history = [];
      saveHistory(history);
      renderHistory(body, history);
    });

    async function handleSend() {
      const text = (input.value || "").trim();
      if (!text) return;

      // push user msg
      history.push({ role: "user", content: text });
      saveHistory(history);
      renderHistory(body, history);

      input.value = "";
      send.disabled = true;

      // show typing bubble
      const typing = { role: "assistant", content: "Thinking..." };
      history.push(typing);
      renderHistory(body, history);

      try {
        const reply = await callAssistant({ userText: text, history });
        typing.content = reply || "No response.";
      } catch (e) {
        typing.content = e?.message || "Error contacting assistant.";
      } finally {
        send.disabled = false;
        saveHistory(history);
        renderHistory(body, history);
      }
    }

    send.addEventListener("click", handleSend);

    input.addEventListener("keydown", (e) => {
      // Enter sends, Shift+Enter makes a new line
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });
  }

  // Run on every page
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();