// RAG Knowledge Assistant - plain JavaScript, talks to the FastAPI backend (/api/*)

const $ = (id) => document.getElementById(id);

const EXAMPLES = [
  "What chunk size and overlap does ingestion use?",
  "How does hybrid search merge the two result lists?",
  "What happens when the documents have no answer?",
];

const asked = []; // { question, data } asked in this browser session

// ------------------------------------------------------------------ helpers
function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function sourceLabel(s) {
  const parts = [];
  if (s.section) parts.push(`section “${s.section}”`);
  if (s.page) parts.push(`page ${s.page}`);
  return parts.join(" · ");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    ...options,
  });
  let body = null;
  try { body = await response.json(); } catch { /* not JSON */ }
  if (response.status === 401 && path !== "/api/me") {
    showSignIn("Your session has ended. Please sign in again.");  // expired while using the app
  }
  if (!response.ok) {
    const error = new Error(readableError(body, response.status));
    error.body = body;
    error.status = response.status;
    throw error;
  }
  return body;
}

function readableError(body, status) {
  if (!body) return `The server answered with status ${status}.`;
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) {  // FastAPI validation errors
    return body.detail.map((d) => `${d.loc?.slice(-1)[0] ?? "input"}: ${d.msg}`).join("; ");
  }
  return `The server answered with status ${status}.`;
}

// ------------------------------------------------------------------ sign-in
const LOGIN_ERRORS = {
  not_allowed: (user) => `@${user} signed in with GitHub, but has no access to this app. Ask an admin to add your username.`,
  cancelled: () => "Sign-in was cancelled on GitHub.",
  expired: () => "That sign-in link expired. Please try again.",
  failed: () => "GitHub sign-in failed. Please try again.",
};

async function start() {
  // Errors sent back by /auth/callback arrive as ?login_error=...
  const params = new URLSearchParams(location.search);
  const loginError = params.get("login_error");
  if (loginError) window.history.replaceState(null, "", "/");

  let me;
  try {
    me = await api("/api/me");
  } catch (error) {
    if (error.status === 503) {
      showSignIn(error.message, { configError: true });
    } else if (error.status === 401 || error.status === 403) {
      const message = loginError ? (LOGIN_ERRORS[loginError] || LOGIN_ERRORS.failed)(params.get("user") || "")
                                 : (error.status === 403 ? error.message : "");
      showSignIn(message);
    } else {
      showSignIn("The server is not reachable. Is `python serve.py` running?", { configError: true });
    }
    return;
  }
  showApp(me);
}

function showSignIn(message = "", { configError = false } = {}) {
  $("app").hidden = true;
  $("user-menu").hidden = true;
  $("status").hidden = true;
  $("signin").hidden = false;
  $("signin-button").hidden = configError;
  const box = $("signin-error");
  box.hidden = !message;
  box.innerHTML = configError
    ? `<strong>Sign-in is not set up yet.</strong> ${escapeHtml(message)}<br>See “Set up GitHub sign-in” in the README.`
    : escapeHtml(message);
}

function showApp(me) {
  $("signin").hidden = true;
  $("app").hidden = false;
  $("status").hidden = false;
  $("user-menu").hidden = false;
  $("user-login").textContent = me.auth === "off" ? "Sign-in off" : `@${me.login}`;
  $("user-role").textContent = me.role;
  $("user-avatar").hidden = !me.avatar_url;
  if (me.avatar_url) $("user-avatar").src = me.avatar_url;
  $("sign-out").hidden = me.auth === "off";
  loadStatus();
}

$("sign-out").addEventListener("click", async () => {
  await fetch("/auth/logout", { method: "POST", credentials: "same-origin" });
  asked.length = 0;
  $("history").innerHTML = '<li class="empty">No questions yet</li>';
  $("result").hidden = true;
  $("placeholder").hidden = false;
  showSignIn("You are signed out.");
});

// ------------------------------------------------------------------ status + filters
async function loadStatus() {
  const status = $("status");
  try {
    const health = await api("/api/health");
    status.className = "status status--ready";
    status.textContent = `Ready · ${health.chunks} chunks · ${health.llm.replace("openai:", "")}` +
      (health.fake_mode ? " · offline test mode" : "");
    await loadFiles();
  } catch (error) {
    status.className = "status status--error";
    status.textContent = error.body?.error || error.message || "Backend not reachable";
  }
}

async function loadFiles() {
  const files = await api("/api/files");
  const select = $("f-file");
  const sections = new Set();
  for (const f of files) {
    const option = document.createElement("option");
    option.value = f.file_name;
    option.textContent = `${f.file_name} (${f.chunks})`;
    select.append(option);
    f.sections.forEach((s) => sections.add(s));
  }
  $("section-list").innerHTML = [...sections].sort()
    .map((s) => `<option value="${escapeHtml(s)}"></option>`).join("");
}

function readFilters() {
  const page = parseInt($("f-page").value, 10);
  return {
    file_name: $("f-file").value || null,
    file_type: $("f-type").value || null,
    section: $("f-section").value.trim() || null,
    page: Number.isFinite(page) && page > 0 ? page : null,
    tables_only: $("f-tables").checked,
    exclude_flagged: $("f-flagged").checked,
  };
}

function resetFilters() {
  ["f-file", "f-type", "f-section", "f-page"].forEach((id) => { $(id).value = ""; });
  $("f-tables").checked = false;
  $("f-flagged").checked = false;
  $("f-topk").value = 3;
  $("topk-value").textContent = "3";
}

// ------------------------------------------------------------------ asking
async function ask(question) {
  question = question.trim();
  if (question.length < 3) {
    showError("Please type a question of at least 3 characters.");
    return;
  }
  setBusy(true);
  hideError();
  try {
    const data = await api("/api/ask", {
      method: "POST",
      body: JSON.stringify({
        question,
        filters: readFilters(),
        top_k: parseInt($("f-topk").value, 10),
        include_search: true,
      }),
    });
    showResult(data);
    remember(question, data);
  } catch (error) {
    showError(error.message, error.body?.passages);
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  const button = $("ask-button");
  button.disabled = busy;
  button.querySelector(".button-label").innerHTML = busy ? '<span class="spinner"></span>' : "Ask";
}

// ------------------------------------------------------------------ rendering
function renderAnswerText(text, sources) {
  const valid = new Set(sources.map((s) => s.number));
  return escapeHtml(text).replace(/\[(\d+)\]/g, (match, n) =>
    valid.has(Number(n))
      ? `<button type="button" class="cite" data-source="${n}" title="Show source ${n}">${n}</button>`
      : match);
}

function showResult(data) {
  $("placeholder").hidden = true;
  $("result").hidden = false;
  $("result-question").textContent = data.question;
  $("result-meta").textContent = `${data.seconds}s` + (data.filters !== "none" ? ` · filters: ${data.filters}` : "");

  const answer = $("answer");
  answer.className = data.found ? "answer" : "answer answer--none";
  answer.innerHTML = data.found ? renderAnswerText(data.answer, data.sources) : escapeHtml(data.answer);

  $("warnings").innerHTML = (data.warnings || []).map((w) => `<li>${escapeHtml(w)}</li>`).join("");

  $("sources-section").hidden = !data.sources.length;
  $("sources").innerHTML = data.sources.map((s) => `
    <div class="source" id="source-${s.number}">
      <div class="source-head">
        <span class="badge">${s.number}</span>
        <span class="source-title">${escapeHtml(s.file_name)}</span>
        <span class="source-where">${escapeHtml(sourceLabel(s))}</span>
        <span class="tag ${s.cited ? "tag--cited" : "tag--unused"}">${s.cited ? "cited" : "not cited"}</span>
      </div>
      <div class="score">
        <span>similarity ${s.similarity.toFixed(2)}</span>
        <span class="bar"><span style="width:${Math.max(0, Math.min(1, s.similarity)) * 100}%"></span></span>
        ${s.keyword_score ? `<span>keyword ${s.keyword_score.toFixed(1)}</span>` : ""}
      </div>
      <details><summary>Show the text sent to the model</summary><pre>${escapeHtml(s.text)}</pre></details>
    </div>`).join("");

  $("search-rows").innerHTML = (data.search || []).map((h) => `
    <tr class="${h.relevant ? "passed" : "dropped"}">
      <td>${h.relevant ? "✓" : "·"}</td>
      <td>${escapeHtml(h.file_name)}<br><span class="hint">${escapeHtml(h.section || "—")}${h.page ? ` · p.${h.page}` : ""}</span></td>
      <td class="num">${h.similarity.toFixed(3)}</td>
      <td class="num">${h.keyword_score.toFixed(2)}</td>
      <td class="num">sem #${h.semantic_rank ?? "–"} · kw #${h.keyword_rank ?? "–"}</td>
    </tr>`).join("") || `<tr><td colspan="5" class="hint">No chunks matched the filters.</td></tr>`;
}

function flashSource(number) {
  const card = $(`source-${number}`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.add("source--flash");
  setTimeout(() => card.classList.remove("source--flash"), 1600);
}

function showError(message, passages) {
  const box = $("error");
  let html = `<strong>Something went wrong.</strong> ${escapeHtml(message)}`;
  if (passages?.length) {
    html += "<p>The search still found these passages:</p><ul>" + passages.map((p) =>
      `<li><strong>${escapeHtml(p.file_name)}</strong>: ${escapeHtml(p.preview)}</li>`).join("") + "</ul>";
  }
  box.innerHTML = html;
  box.hidden = false;
}

function hideError() { $("error").hidden = true; }

function remember(question, data) {
  asked.unshift({ question, data });
  $("history").innerHTML = asked.slice(0, 12).map((h, i) => `
    <li><button type="button" data-index="${i}">
      <span class="dot ${h.data.found ? "" : "dot--none"}"></span>${escapeHtml(h.question)}
    </button></li>`).join("");
}

// ------------------------------------------------------------------ events
$("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  ask($("question").value);
});

$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    ask($("question").value);
  }
});

$("answer").addEventListener("click", (event) => {
  const cite = event.target.closest(".cite");
  if (cite) flashSource(cite.dataset.source);
});

$("history").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-index]");
  if (!button) return;
  const item = asked[Number(button.dataset.index)];
  $("question").value = item.question;
  showResult(item.data);
});

$("f-topk").addEventListener("input", (event) => { $("topk-value").textContent = event.target.value; });
$("reset-filters").addEventListener("click", resetFilters);

$("examples").innerHTML = EXAMPLES.map((q) =>
  `<button type="button" class="chip">${escapeHtml(q)}</button>`).join("");
$("examples").addEventListener("click", (event) => {
  const chip = event.target.closest(".chip");
  if (!chip) return;
  $("question").value = chip.textContent;
  ask(chip.textContent);
});

start();
