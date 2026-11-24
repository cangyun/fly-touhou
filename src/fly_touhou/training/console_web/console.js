"use strict";

const labels = {
  zh: {
    title: "训练控制台", lead: "启动原作作业，查看采样进度、参数更新和评估结果。",
    activeJobs: "活跃作业", activeDecisions: "活跃作业决策", activeUpdates: "活跃作业参数更新",
    shownDecisions: "所列作业决策", shownUpdates: "所列作业参数更新",
    newJob: "新建作业", operation: "作业类型", train: "训练", evaluate: "冻结评估", "fit-attack": "符卡微调",
    game: "原作", gameTh14: "TH14 · 辉针城", gameTh08: "TH08 · 永夜抄", objective: "通关目标",
    difficulty: "难度", scope: "训练范围", joint: "全网络", encoderHeads: "固定脑内参数",
    exe: "原作 EXE 路径", guide: "已验证的符卡轨迹目录", workers: "并行实例", decisions: "决策预算",
    fitSteps: "Adam 更新次数", seed: "种子", record: "记录输入特征", advanced: "权重、输出与续训",
    checkpoint: "模型权重", resume: "续训 Checkpoint", output: "输出目录", connectome: "连接组目录",
    spec: "策略配置", course: "TH08 课程清单", segment: "每次采样决策", published: "使用发布权重",
    preset: "使用作业配置", autoOutput: "自动分配", preview: "预览计划", start: "启动作业",
    queue: "作业队列", history: "显示已结束作业", job: "作业", status: "状态", stage: "关卡",
    decisionsShort: "决策", updates: "更新", loading: "正在读取作业", details: "作业详情",
    choose: "选择一项作业查看运行详情。", open: "打开输出目录", stop: "边界停训",
    logs: "最近日志", refreshLog: "刷新日志", close: "关闭", cancel: "取消", confirmStop: "请求停止",
    stopQuestion: "为所选作业提交停止请求，在后端可保存的边界结束？",
    ready: "本地已连接", connecting: "正在连接", offline: "本地服务离线", lastUpdate: "更新于",
    noJobs: "当前没有活跃作业。", noHistory: "该作业目录尚无记录。", noLog: "暂无日志。",
    phase: "运行阶段", elapsed: "用时", miss: "失误", bomb: "Bomb", contact: "原作碰撞", matched: "路线匹配",
    originalOutcome: "原作结果", yes: "有", no: "无", started: "已提交作业", stopSent: "已请求边界停训",
    folderOpened: "已打开输出目录", requestFailed: "请求失败", inProgress: "正在处理",
    states: {starting: "启动中", running: "运行中", stopping: "待停", stopped: "已停止",
      finished: "已结束", failed: "失败", exited: "已退出", "launcher-error": "启动失败"},
    phases: {complete: "完成", frozen: "冻结推理", sampling: "采样", update: "参数更新",
      loading: "加载模型", paused: "已暂停", "original-prefix-replay": "回放已验证前缀"},
    outcomes: {"story-clear": "通关", "nbnm-failure": "首次碰撞", stopped: "已停止", "game-over": "Game Over"}
  },
  en: {
    title: "Training console", lead: "Launch original-game jobs and inspect rollouts, updates, and evaluation results.",
    activeJobs: "Active jobs", activeDecisions: "Active job decisions", activeUpdates: "Active optimizer updates",
    shownDecisions: "Listed job decisions", shownUpdates: "Listed optimizer updates",
    newJob: "New job", operation: "Job type", train: "Train", evaluate: "Frozen eval", "fit-attack": "Fit attack",
    game: "Original game", gameTh14: "TH14 · Double Dealing Character", gameTh08: "TH08 · Imperishable Night",
    objective: "Clear objective", difficulty: "Difficulty", scope: "Training scope", joint: "Joint training",
    encoderHeads: "Freeze brain parameters", exe: "Original EXE path", guide: "Verified attack guide folder",
    workers: "Parallel instances", decisions: "Decision budget", fitSteps: "Adam updates", seed: "Seed",
    record: "Record input features", advanced: "Weights, outputs, and resumption", checkpoint: "Model weights",
    resume: "Resume checkpoint", output: "Output directory", connectome: "Connectome folder", spec: "Policy spec",
    course: "TH08 course manifest", segment: "Decisions per segment", published: "Use published weights",
    preset: "Use job configuration", autoOutput: "Choose automatically", preview: "Preview plan", start: "Start job",
    queue: "Job queue", history: "Show completed jobs", job: "Job", status: "Status", stage: "Stage",
    decisionsShort: "Decisions", updates: "Updates", loading: "Loading jobs", details: "Job details",
    choose: "Select a job to see its details.", open: "Open output folder", stop: "Stop at boundary",
    logs: "Recent logs", refreshLog: "Refresh log", close: "Close", cancel: "Cancel", confirmStop: "Request stop",
    stopQuestion: "Ask the selected runner to stop at its next supported save boundary?",
    ready: "Local service connected", connecting: "Connecting", offline: "Local service offline", lastUpdate: "Updated",
    noJobs: "There are no active jobs.", noHistory: "This job directory has no records.", noLog: "No log entries yet.",
    phase: "Phase", elapsed: "Elapsed", miss: "Misses", bomb: "Bombs", contact: "Original contact", matched: "Guide match",
    originalOutcome: "Game outcome", yes: "Yes", no: "No", started: "Job submitted", stopSent: "Stop requested",
    folderOpened: "Output folder opened", requestFailed: "Request failed", inProgress: "Working",
    states: {starting: "Starting", running: "Running", stopping: "Stopping", stopped: "Stopped",
      finished: "Finished", failed: "Failed", exited: "Exited", "launcher-error": "Launch error"},
    phases: {complete: "Complete", frozen: "Frozen inference", sampling: "Sampling", update: "Optimizer update",
      loading: "Loading model", paused: "Paused", "original-prefix-replay": "Verified prefix replay"},
    outcomes: {"story-clear": "Clear", "nbnm-failure": "First contact", stopped: "Stopped", "game-over": "Game over"}
  }
};

const $ = id => document.getElementById(id);
const form = $("job-form");
const activeStates = new Set(["starting", "running", "stopping"]);
const numberOptions = new Set(["workers", "decisions", "seed", "fit_steps", "segment"]);
const history = $("history");
history.checked = new URLSearchParams(window.location.search).get("history") === "1";

let language = "zh";
try { if (localStorage.getItem("fly-touhou.language") === "en") language = "en"; } catch {}
let token = "";
let connection = "connecting";
let busy = false;
let snapshot = null;
let selected = null;
let refreshTimer;
let refreshing = false;
let previousList = "";
let stopTarget = null;
let notice = null;

const t = key => labels[language][key] ?? key;
const translate = (group, key) => key == null ? "—" : labels[language][group][key] ?? key;
const n = value => Number.isFinite(value) ? new Intl.NumberFormat(language).format(value) : "—";
const sum = (items, key) => items.reduce((total, row) => total + (Number.isFinite(row[key]) ? row[key] : 0), 0);
const show = (id, value) => { $(id).textContent = value; };
const selectedJob = () => snapshot?.jobs.find(row => row.id === selected);

function applyLanguage() {
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  for (const node of document.querySelectorAll("[data-i18n]")) node.textContent = t(node.dataset.i18n);
  for (const node of document.querySelectorAll("[data-placeholder]")) node.placeholder = t(node.dataset.placeholder);
  show("language", language === "zh" ? "English" : "简体中文");
  updateConnection(connection);
  renderSnapshot();
  renderNotice();
}

function updateConnection(state) {
  connection = state;
  $("connection").dataset.state = state;
  $("connection").lastElementChild.textContent = t(state);
  updateButtons();
}

function updateButtons() {
  const disabled = busy || connection !== "ready";
  $("preview").disabled = disabled;
  $("start").disabled = disabled;
  $("open-output").disabled = disabled || !selectedJob();
  $("stop").disabled = disabled || !["starting", "running"].includes(selectedJob()?.status);
}

function updateFields() {
  const fields = form.elements;
  const operation = fields.operation.value;
  const fitting = operation === "fit-attack";
  if (fitting) fields.game.value = "th08";
  const th08 = fields.game.value === "th08";
  fields.game.disabled = fitting;
  fields.objective.options[1].disabled = th08;
  if (th08) fields.objective.value = "nbnm";
  for (const label of form.querySelectorAll("[data-modes]")) {
    let visible = label.dataset.modes.split(" ").includes(operation);
    const field = label.querySelector("input, select");
    if (field.name === "record_observations" || field.name === "course") visible &&= th08;
    label.hidden = !visible;
    field.disabled = !visible;
  }
  fields.executable.placeholder = th08 ? "D:\\Games\\TH08\\th08.exe" : "D:\\Games\\TH14\\th14.exe";
  fields.workers.max = fields.objective.value === "clear" ? "30" : "100";
  fields.seed.max = th08 ? "65535" : "4294967295";
}

async function request(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json", "X-Console-Token": token},
    body: JSON.stringify(body)
  };
  const response = await fetch(path, {...options, cache: "no-store"});
  const data = await response.json();
  if (!response.ok) {
    if (response.status === 403 && body !== undefined) token = "";
    throw new Error(data.error || t("requestFailed"));
  }
  return data;
}

async function refresh() {
  clearTimeout(refreshTimer);
  if (refreshing) return;
  refreshing = true;
  try {
    if (!token || connection === "offline") token = (await request("/api/session")).token;
    snapshot = await request("/api/jobs" + (history.checked ? "?history=1" : ""));
    if (!selectedJob()) selected = null;
    updateConnection("ready");
    renderSnapshot();
  } catch {
    updateConnection("offline");
  } finally {
    refreshing = false;
    refreshTimer = setTimeout(refresh, document.hidden ? 10000 : 2000);
  }
}

function cell(content, className) {
  const td = document.createElement("td");
  if (className) td.className = className;
  if (typeof content === "string") td.textContent = content;
  else td.append(content);
  return td;
}

function renderSnapshot() {
  if (!snapshot) return;
  const jobs = snapshot.jobs;
  show("active-count", n(jobs.filter(row => activeStates.has(row.status)).length));
  show("decision-label", t(history.checked ? "shownDecisions" : "activeDecisions"));
  show("updates-label", t(history.checked ? "shownUpdates" : "activeUpdates"));
  show("decision-count", n(sum(jobs, "decisions")));
  show("update-count", n(sum(jobs, "updates")));
  $("refreshed").dateTime = snapshot.observed_at;
  show("refreshed", t("lastUpdate") + " " + new Date(snapshot.observed_at).toLocaleTimeString(language));
  $("empty").hidden = jobs.length > 0;
  show("empty", t(history.checked ? "noHistory" : "noJobs"));
  const fingerprint = JSON.stringify([language, jobs.map(row =>
    [row.id, row.status, row.phase, row.stage, row.frame, row.decisions, row.updates])]);
  if (fingerprint !== previousList) {
    previousList = fingerprint;
    const fragment = document.createDocumentFragment();
    for (const job of jobs) {
      const row = document.createElement("tr");
      row.className = "job-row";
      row.dataset.id = job.id;
      const select = document.createElement("button");
      select.type = "button";
      select.className = "job-name";
      select.textContent = t(job.operation);
      const subline = document.createElement("span");
      subline.className = "job-subtitle";
      subline.textContent = job.game.toUpperCase() + " · " + (job.objective === "clear" ? "1CC" : "NBNM");
      const name = cell(select);
      name.append(subline);
      const badge = document.createElement("span");
      badge.className = "pill";
      if (activeStates.has(job.status)) badge.classList.add("running");
      if (["failed", "launcher-error", "exited"].includes(job.status)) badge.classList.add("failed");
      badge.textContent = translate("states", job.status);
      const stage = cell(n(job.stage));
      if (Number.isFinite(job.frame)) {
        const frame = document.createElement("span");
        frame.className = "job-subtitle";
        frame.textContent = n(job.frame) + " F";
        stage.append(frame);
      }
      row.append(name, cell(badge), stage, cell(n(job.decisions), "numeric"), cell(n(job.updates), "numeric"));
      row.addEventListener("click", () => selectJob(job.id));
      fragment.append(row);
    }
    $("jobs").replaceChildren(fragment);
  }
  renderDetail();
}

function selectJob(id) {
  if (id !== selected) {
    selected = id;
    $("logs").open = false;
    show("log-text", "");
  }
  renderDetail();
}

function renderDetail() {
  const job = selectedJob();
  for (const row of $("jobs").children) {
    const chosen = row.dataset.id === selected;
    row.classList.toggle("selected", chosen);
    row.querySelector("button").setAttribute("aria-pressed", chosen);
  }
  $("job-detail").hidden = !job;
  $("no-selection").hidden = !!job;
  show("detail-job", job ? job.game.toUpperCase() + " · " + t(job.operation) : "");
  updateButtons();
  if (!job) return;
  const elapsed = Number.isFinite(job.elapsed_seconds) ?
    new Date(Math.round(job.elapsed_seconds) * 1000).toISOString().slice(11, 19) : "—";
  const facts = [
    ["phase", translate("phases", job.phase)], ["elapsed", elapsed],
    ["decisions", Number.isFinite(job.budget) ? n(job.budget) : "—"]
  ];
  if (job.outcome) facts.push(["originalOutcome", translate("outcomes", job.outcome)]);
  if (job.miss != null) facts.push(["miss", n(job.miss)]);
  if (job.bomb != null) facts.push(["bomb", n(job.bomb)]);
  if (job.contact != null) facts.push(["contact", t(job.contact ? "yes" : "no")]);
  if (job.matched != null) facts.push(["matched", n(job.matched) + " / " + n(job.route_decisions)]);
  const nodes = facts.map(([key, value]) => {
    const group = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = t(key);
    dd.textContent = value;
    group.append(dt, dd);
    return group;
  });
  $("job-facts").replaceChildren(...nodes);
  const progress = $("progress");
  progress.hidden = !(job.budget > 0 && Number.isFinite(job.decisions));
  if (!progress.hidden) {
    progress.max = job.budget;
    progress.value = Math.min(job.decisions, job.budget);
    progress.setAttribute("aria-label", t("decisions"));
  }
  show("output-path", job.output);
  show("job-error", job.error || "");
  $("job-error").hidden = !job.error;
}

function renderNotice() {
  const output = $("form-note");
  output.classList.toggle("error", !!notice?.error);
  output.textContent = notice ? (notice.key ? t(notice.key) : "") + (notice.detail ? " · " + notice.detail : "") : "";
}

async function command(path, body, success) {
  if (busy || connection !== "ready") return;
  busy = true;
  notice = {key: "inProgress"};
  renderNotice();
  updateButtons();
  try {
    await success(await request(path, body));
  } catch (error) {
    notice = {key: "requestFailed", detail: error.message, error: true};
  } finally {
    busy = false;
    renderNotice();
    updateButtons();
  }
}

function options() {
  const setup = {};
  for (const [key, value] of new FormData(form)) {
    const clean = value.trim();
    if (clean) setup[key] = numberOptions.has(key) ? Number(clean) : key === "record_observations" ? true : clean;
  }
  if (setup.operation === "fit-attack") setup.game = "th08";
  return setup;
}

async function loadLog() {
  const id = selected;
  if (!id) return;
  const channel = $("log-channel").value;
  try {
    const params = new URLSearchParams({id, channel});
    const result = await request("/api/log?" + params);
    if (id === selected && channel === $("log-channel").value) show("log-text", result.text || t("noLog"));
  } catch (error) {
    if (id === selected) show("log-text", error.message);
  }
}

$("language").addEventListener("click", () => {
  language = language === "zh" ? "en" : "zh";
  try { localStorage.setItem("fly-touhou.language", language); } catch {}
  applyLanguage();
});
form.addEventListener("change", updateFields);
history.addEventListener("change", refresh);
form.addEventListener("submit", event => {
  event.preventDefault();
  void command("/api/start", options(), async result => {
    selected = result.id;
    notice = {key: "started", detail: result.id};
    await refresh();
  });
});
$("preview").addEventListener("click", () => {
  if (!form.reportValidity()) return;
  void command("/api/plan", options(), result => {
    notice = null;
    show("plan-text", JSON.stringify(result.plan, null, 2));
    $("plan-dialog").showModal();
  });
});
$("open-output").addEventListener("click", () => {
  if (selectedJob()) void command("/api/open", {id: selected}, () => { notice = {key: "folderOpened"}; });
});
$("stop").addEventListener("click", () => {
  if (!selectedJob()) return;
  stopTarget = selected;
  $("stop-dialog").returnValue = "";
  $("stop-dialog").showModal();
});
$("stop-dialog").addEventListener("close", () => {
  if ($("stop-dialog").returnValue !== "confirm") return;
  void command("/api/stop", {id: stopTarget}, async () => {
    notice = {key: "stopSent"};
    await refresh();
  });
});
$("logs").addEventListener("toggle", () => { if ($("logs").open) void loadLog(); });
$("log-channel").addEventListener("change", loadLog);
$("refresh-log").addEventListener("click", loadLog);

updateFields();
applyLanguage();
void refresh();
