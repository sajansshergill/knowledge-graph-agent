const apiStatus = document.querySelector("#apiStatus");
const queryInput = document.querySelector("#queryInput");
const topK = document.querySelector("#topK");
const askButton = document.querySelector("#askButton");
const clearButton = document.querySelector("#clearButton");
const suggestions = document.querySelector("#suggestions");
const emptyState = document.querySelector("#emptyState");
const result = document.querySelector("#result");
const answerText = document.querySelector("#answerText");
const routeBadge = document.querySelector("#routeBadge");
const retrievalPath = document.querySelector("#retrievalPath");
const latency = document.querySelector("#latency");
const faithfulness = document.querySelector("#faithfulness");
const sourceCount = document.querySelector("#sourceCount");
const citations = document.querySelector("#citations");
const timeline = document.querySelector("#timeline");

const citationTemplate = document.querySelector("#citationTemplate");
const timelineTemplate = document.querySelector("#timelineTemplate");

const sampleQueries = [
  "Why was the auth service migrated off PostgreSQL?",
  "What does onboarding for a backend engineer look like?",
  "Which team owns the rate-limiting module?",
  "What value does this project create for a 600 engineer enterprise?",
];

init();

function init() {
  renderSuggestions();
  checkHealth();

  askButton.addEventListener("click", ask);
  clearButton.addEventListener("click", clear);
  queryInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      ask();
    }
  });
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("API health check failed");
    const body = await response.json();
    apiStatus.classList.add("online");
    apiStatus.classList.remove("offline");
    apiStatus.innerHTML = `<span class="pulse"></span>${body.service} online`;
  } catch {
    apiStatus.classList.add("offline");
    apiStatus.classList.remove("online");
    apiStatus.innerHTML = `<span class="pulse"></span>API offline`;
  }
}

function renderSuggestions() {
  suggestions.innerHTML = "";
  sampleQueries.forEach((query) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = query;
    button.addEventListener("click", () => {
      queryInput.value = query;
      queryInput.focus();
    });
    suggestions.appendChild(button);
  });
}

async function ask() {
  const query = queryInput.value.trim();
  if (!query) {
    showToast("Ask a question first.");
    queryInput.focus();
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, top_k: Number(topK.value) }),
    });

    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || body.error || "Query failed");
    }
    renderResult(body);
  } catch (error) {
    showToast(error.message || "Something went wrong.");
  } finally {
    setLoading(false);
  }
}

function renderResult(body) {
  emptyState.classList.add("hidden");
  result.classList.remove("hidden");

  routeBadge.textContent = body.query_type || "unknown";
  answerText.textContent = body.answer || "No answer returned.";
  retrievalPath.textContent = body.retrieval_path || "none";
  latency.textContent = `${body.latency_ms || 0} ms`;
  faithfulness.textContent = formatScore(body.eval?.faithfulness);
  sourceCount.textContent = String((body.citations || []).length);

  renderCitations(body.citations || []);
  renderTimeline(body.hops || []);
}

function renderCitations(items) {
  citations.innerHTML = "";
  if (!items.length) {
    citations.appendChild(emptyInline("No citations returned yet."));
    return;
  }

  items.forEach((item) => {
    const node = citationTemplate.content.cloneNode(true);
    node.querySelector("strong").textContent = item.title || item.chunk_id || "Untitled source";
    node.querySelector("p").textContent = `${item.source_type || "source"} · ${item.chunk_id || "unknown chunk"}`;
    const link = node.querySelector("a");
    if (item.url) {
      link.href = item.url;
    } else {
      link.remove();
    }
    citations.appendChild(node);
  });
}

function renderTimeline(hops) {
  timeline.innerHTML = "";
  if (!hops.length) {
    timeline.appendChild(emptyInline("No agent hops returned."));
    return;
  }

  hops.forEach((hop) => {
    const node = timelineTemplate.content.cloneNode(true);
    node.querySelector("strong").textContent = titleCase(hop.agent_name || "agent");
    node.querySelector("p").textContent = `${hop.latency_ms || 0} ms · ${summarizeHop(hop)}`;
    timeline.appendChild(node);
  });
}

function summarizeHop(hop) {
  const metadata = hop.metadata || {};
  if (hop.agent_name === "router") {
    return `classified as ${metadata.query_type || "unknown"}`;
  }
  if (hop.agent_name === "retrieval") {
    return `${metadata.retrieval_path || "none"} retrieval, ${(metadata.chunks || []).length} cited evidence chunks`;
  }
  if (hop.agent_name === "synthesis") {
    return `grounded executive answer with ${(metadata.citations || []).length} citations`;
  }
  if (hop.agent_name === "eval") {
    return `grounded: ${metadata.grounded ? "yes" : "needs data"}`;
  }
  return "completed";
}

function emptyInline(text) {
  const div = document.createElement("div");
  div.className = "citation";
  div.innerHTML = `<div><strong>${text}</strong><p>Run ingestion or connect live stores for richer results.</p></div>`;
  return div;
}

function setLoading(isLoading) {
  askButton.disabled = isLoading;
  askButton.querySelector("span").textContent = isLoading ? "Thinking..." : "Ask EKGA";
  routeBadge.textContent = isLoading ? "running" : routeBadge.textContent;
}

function clear() {
  queryInput.value = "";
  queryInput.focus();
  result.classList.add("hidden");
  emptyState.classList.remove("hidden");
  routeBadge.textContent = "idle";
}

function showToast(message) {
  const existing = document.querySelector(".toast");
  existing?.remove();

  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4200);
}

function formatScore(value) {
  if (value === undefined || value === null) return "-";
  return `${Math.round(Number(value) * 100)}%`;
}

function titleCase(value) {
  return value.replace(/(^|_)([a-z])/g, (_, spacer, char) => `${spacer ? " " : ""}${char.toUpperCase()}`);
}
