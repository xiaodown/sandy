const stageOrder = [
  { key: "message_received", label: "Intake", desc: "Message enters pipeline" },
  { key: "vision", label: "Vision", desc: "Image inspection if needed" },
  { key: "bouncer", label: "Bouncer", desc: "Respond or ignore" },
  { key: "tool_started", label: "Tool", desc: "Lookup before talking" },
  { key: "retrieval", label: "Memory", desc: "Vector/RAG recall" },
  { key: "brain", label: "Brain", desc: "Main generation" },
  { key: "reply_send", label: "Send", desc: "Reply delivery" }
];
const MAX_SERVER_NAMES = 4;
const MAX_SERVER_NAME_CHARS = 22;

function el(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function truncateText(value, maxChars) {
  const text = String(value ?? "");
  return text.length > maxChars ? `${text.slice(0, maxChars - 1)}…` : text;
}

function formatJson(value) {
  return escapeHtml(JSON.stringify(value ?? {}, null, 2));
}

function formatTime(isoOrEpoch) {
  if (!isoOrEpoch) return "-";
  const date = typeof isoOrEpoch === "number"
    ? new Date(isoOrEpoch * 1000)
    : new Date(isoOrEpoch);
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleTimeString();
}

function formatAgo(epochSeconds) {
  if (!epochSeconds) return "-";
  const delta = Math.max(0, Math.round(Date.now() / 1000 - epochSeconds));
  if (delta < 5) return "just now";
  if (delta < 60) return `${delta}s ago`;
  if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
  return `${Math.round(delta / 3600)}h ago`;
}

function renderStatus(status) {
  const discord = status.discord || {};
  const mode = String(status.mode || "prod").toUpperCase();
  const envBanner = el("env-banner");
  envBanner.textContent = `${mode} MODE`;
  envBanner.classList.toggle("is-test", mode === "TEST");

  el("discord-state").textContent = discord.connected ? "Connected" : "Offline";
  el("discord-user").textContent = discord.connected
    ? `Logged in as ${discord.user_name || "?"}`
    : "Discord session down";
  const serverNames = (discord.server_names || []).slice(0, MAX_SERVER_NAMES).map((name) => truncateText(name, MAX_SERVER_NAME_CHARS));
  const overflow = Math.max(0, (discord.server_count || 0) - serverNames.length);
  const serverSummary = discord.server_count
    ? `${discord.server_count} server${discord.server_count === 1 ? "" : "s"}`
    : "No attached servers";
  const serverLabel = serverNames.length
    ? `${serverSummary} · ${serverNames.join(", ")}${overflow > 0 ? ` +${overflow} more` : ""}`
    : serverSummary;
  el("discord-servers").textContent = serverLabel;

  el("llm-state").textContent = status.llm?.busy ? "Busy" : "Idle";
  el("llm-note").textContent = status.llm?.busy
    ? "Ollama lock is currently held"
    : "No active inference";

  const memory = status.memory_worker || {};
  el("memory-state").textContent = memory.busy ? "Running" : (memory.queue_depth > 0 ? "Queued" : "Idle");
  el("memory-note").textContent = memory.busy
    ? `Processing ${memory.processing_message_id || "message"}`
    : `Queue depth ${memory.queue_depth ?? 0}`;

  const bouncer = status.last_bouncer_decision;
  if (bouncer) {
    el("bouncer-state").textContent = bouncer.should_respond ? "Respond" : "Ignore";
    const toolText = bouncer.use_tool ? `Tool ${bouncer.tool_name || "?"}` : "No tool";
    el("bouncer-note").textContent = `${toolText} · ${formatAgo(bouncer.recorded_at)}`;
  } else {
    el("bouncer-state").textContent = "None";
    el("bouncer-note").textContent = "No decision yet";
  }

  const current = status.current_turn;
  if (!current) {
    el("current-turn-note").textContent = "Idle";
    el("current-author").textContent = "Nobody";
    el("current-location").textContent = "No active turn";
    el("current-trace").textContent = "-";
    el("current-started").textContent = "Waiting";
  } else {
    el("current-turn-note").textContent = `${current.stage} · ${current.status}`;
    el("current-author").textContent = current.author_name;
    el("current-location").textContent = `${current.guild_name} / #${current.channel_name}`;
    el("current-trace").textContent = current.trace_id;
    el("current-started").textContent = `Started ${formatAgo(current.started_at)}`;
  }

  renderTrack(current);
}

function renderTrack(current) {
  const currentStage = current?.stage || null;
  const container = el("racetrack");
  container.innerHTML = stageOrder.map((stage, index) => {
    const active = currentStage === stage.key;
    const pending = currentStage === null;
    return `
      <div class="stage-card ${active ? "is-active" : ""}">
        <div class="stage-step">T${index + 1}</div>
        <div class="stage-name">${stage.label}</div>
        <div class="stage-meta">${stage.desc}<br>${pending ? "Waiting" : (active ? "Live now" : "Idle")}</div>
      </div>
    `;
  }).join("");
}

function renderGpu(payload) {
  el("gpu-backend").textContent = payload.available
    ? `Source: ${payload.backend}`
    : (payload.error || "No GPU data");

  const grid = el("gpu-grid");
  if (!payload.available || !payload.devices?.length) {
    grid.innerHTML = `<div class="detail-empty">${escapeHtml(payload.error || "GPU telemetry unavailable.")}</div>`;
    return;
  }

  grid.innerHTML = payload.devices.map((gpu) => `
    <div class="gpu-card">
      <div class="gpu-head">
        <div>
          <div class="gpu-index">GPU ${gpu.index}</div>
          <div class="gpu-name">${escapeHtml(gpu.name)}</div>
        </div>
        <div class="badge ${gpu.utilization_gpu_percent > 70 ? "status-warn" : "status-ok"}">${gpu.utilization_gpu_percent}% util</div>
      </div>
      <div class="meter">
        <div class="meter-row"><span>Compute</span><strong>${gpu.utilization_gpu_percent}%</strong></div>
        <div class="meter-bar"><div class="meter-fill" style="width:${Math.max(0, Math.min(100, gpu.utilization_gpu_percent))}%"></div></div>
      </div>
      <div class="meter">
        <div class="meter-row"><span>VRAM</span><strong>${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB</strong></div>
        <div class="meter-bar"><div class="meter-fill" style="width:${Math.max(0, Math.min(100, gpu.memory_utilization_percent))}%"></div></div>
      </div>
      <div class="gpu-facts">
        <div class="fact">Temperature<strong>${gpu.temperature_c ?? "?"}C</strong></div>
        <div class="fact">Power<strong>${gpu.power_draw_w ?? "?"}W</strong></div>
        <div class="fact">Power Limit<strong>${gpu.power_limit_w ?? "?"}W</strong></div>
        <div class="fact">VRAM Use<strong>${gpu.memory_utilization_percent}%</strong></div>
      </div>
    </div>
  `).join("");
}

function renderRecent(payload) {
  const tbody = el("recent-turns");
  const turns = payload.turns || [];
  if (!turns.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="detail-empty">No recent turns yet.</td></tr>`;
    return;
  }
  tbody.innerHTML = turns.map((turn) => `
    <tr data-trace-id="${turn.trace_id}">
      <td class="mono">${formatTime(turn.created_at)}</td>
      <td>${escapeHtml(turn.author_name)}</td>
      <td>${escapeHtml(turn.guild_name)} / #${escapeHtml(turn.channel_name)}</td>
      <td>${turn.replied ? "yes" : "no"}</td>
      <td>${escapeHtml(turn.tool_name || "-")}</td>
      <td class="mono">${turn.duration_ms ?? "-"}ms</td>
      <td class="truncate"><div class="recent-content">${escapeHtml(turn.content || "")}</div></td>
    </tr>
  `).join("");

  tbody.querySelectorAll("tr[data-trace-id]").forEach((row) => {
    row.addEventListener("click", () => loadTrace(row.dataset.traceId));
  });
}

function renderDetail(detail) {
  openTraceModal();
  el("detail-trace-note").textContent = `Trace ${detail.trace_id}`;
  const input = detail.turn_input || {};
  const artifacts = detail.artifacts || {};
  const timeline = detail.timeline || [];
  el("detail-pane").className = "detail-stack";
  el("detail-pane").innerHTML = `
    <div class="detail-card">
      <h3>Turn Input</h3>
      <pre>${escapeHtml((input.guild_name || "?") + " / #" + (input.channel_name || "?") + " <" + (input.author_name || "?") + ">\\n\\n" + (input.resolved_content || input.raw_content || "(empty)"))}</pre>
    </div>
    <div class="detail-card">
      <h3>Timeline</h3>
      <div class="timeline">
        ${timeline.map((event) => `
          <div class="timeline-item">
            <div class="timeline-stage">${escapeHtml(event.stage || "?")}</div>
            <div class="timeline-body">
              <pre>${formatJson(event)}</pre>
              <div class="timeline-meta">status=${escapeHtml(event.status || "ok")} · duration=${escapeHtml(event.duration_ms ?? "-")}ms</div>
            </div>
          </div>
        `).join("")}
      </div>
    </div>
    <div class="detail-card">
      <h3>Bouncer</h3>
      <pre>${formatJson(artifacts.bouncer_decision)}</pre>
    </div>
    <div class="detail-card">
      <h3>Retrieval</h3>
      <pre>${formatJson(artifacts.retrieval)}</pre>
    </div>
    <div class="detail-card">
      <h3>Tool</h3>
      <pre>${formatJson(artifacts.tool_call)}</pre>
    </div>
    <div class="detail-card">
      <h3>Reply</h3>
      <pre>${formatJson(artifacts.reply_output)}</pre>
    </div>
  `;
}

function openTraceModal() {
  el("trace-modal-backdrop").classList.remove("is-hidden");
  el("trace-modal-shell").classList.remove("is-hidden");
  el("trace-modal-shell").setAttribute("aria-hidden", "false");
}

function closeTraceModal() {
  el("trace-modal-backdrop").classList.add("is-hidden");
  el("trace-modal-shell").classList.add("is-hidden");
  el("trace-modal-shell").setAttribute("aria-hidden", "true");
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function refreshStatus() {
  try {
    const [status, gpu] = await Promise.all([
      fetchJson("/api/status"),
      fetchJson("/api/gpu")
    ]);
    renderStatus(status);
    renderGpu(gpu);
  } catch (error) {
    el("discord-state").textContent = "Error";
    el("discord-user").textContent = error.message;
  }
}

async function refreshRecent() {
  try {
    const recent = await fetchJson("/api/turns/recent?limit=12");
    renderRecent(recent);
  } catch (error) {
    el("recent-turns").innerHTML = `<tr><td colspan="7" class="detail-empty">${escapeHtml(error.message)}</td></tr>`;
  }
}

async function loadTrace(traceId) {
  try {
    const detail = await fetchJson(`/api/turns/${traceId}`);
    renderDetail(detail);
  } catch (error) {
    openTraceModal();
    el("detail-trace-note").textContent = "Trace detail failed";
    el("detail-pane").className = "detail-empty";
    el("detail-pane").textContent = error.message;
  }
}

el("trace-modal-close").addEventListener("click", closeTraceModal);
el("trace-modal-backdrop").addEventListener("click", closeTraceModal);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeTraceModal();
  }
});

refreshStatus();
refreshRecent();
setInterval(refreshStatus, 1200);
setInterval(refreshRecent, 3500);
