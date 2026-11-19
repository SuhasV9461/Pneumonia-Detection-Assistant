// app.js — the web page: upload, draw the X-ray with its findings, show the report, chat.
// Talks to app.py: POST /api/analyze, GET /api/image/<id>/<kind>.png, POST /api/chat, GET /api/report/<id>.pdf

const $ = (id) => document.getElementById(id);
const state = { file: null, id: null, record: null, xray: null, heat: null };
const AMBER = "#E39B2D";

// ------------------------------------------------------------------ choosing a file
const stage = $("stage");
$("file").addEventListener("change", (e) => pick(e.target.files[0]));
["dragenter", "dragover"].forEach((t) => stage.addEventListener(t, (e) => { e.preventDefault(); stage.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => stage.addEventListener(t, (e) => { e.preventDefault(); stage.classList.remove("over"); }));
stage.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));

function pick(file) {
  if (!file) return;
  state.file = file;
  $("file-name").textContent = `${file.name} (${(file.size / 1048576).toFixed(1)} MB)`;
  $("go").disabled = false;
  hideError();
}

// ------------------------------------------------------------------ analyse
$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.file) return;
  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("age", $("age").value);
  fd.append("sex", $("sex").value);
  fd.append("view", $("view").value);
  busy(true, "Analysing the X-ray. The first one also loads the language model (about 30 s).");
  try {
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "The analysis failed.");
    state.id = data.id; state.record = data.record;
    [state.xray, state.heat] = await Promise.all([loadImg(`/api/image/${data.id}/xray.png`), loadImg(`/api/image/${data.id}/heat.png`)]);
    fillFromHeader(data.header);
    show(data);
  } catch (err) {
    showError(err.message);
  } finally {
    busy(false);
  }
});

function loadImg(src) {
  return new Promise((ok, fail) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => fail(new Error("Could not load the image.")); i.src = src; });
}

function fillFromHeader(h) {
  if (!h) return;
  if (h.age != null && !$("age").value) $("age").value = h.age;
  if (h.sex && !$("sex").value) $("sex").value = h.sex;
  if (h.view && !$("view").value && ["AP", "PA"].includes(h.view)) $("view").value = h.view;
}

// ------------------------------------------------------------------ drawing
const cv = $("canvas"), ctx = cv.getContext("2d");
$("show-boxes").addEventListener("change", draw);
$("show-heat").addEventListener("change", draw);

function draw() {
  if (!state.xray) return;
  ctx.drawImage($("show-heat").checked ? state.heat : state.xray, 0, 0, 512, 512);
  if (!$("show-boxes").checked) return;
  state.record.findings.forEach((f, i) => {
    const [x1, y1, x2, y2] = f.box.map((v) => v * 512);
    ctx.save();
    ctx.strokeStyle = AMBER; ctx.lineWidth = 2.5;
    ctx.setLineDash(f.box_confidence === "clearly marked" ? [] : [9, 6]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    ctx.fillStyle = AMBER; ctx.fillRect(x1, y1, 22, 22);
    ctx.fillStyle = "#1C1408"; ctx.font = "600 13px Public Sans, sans-serif"; ctx.textAlign = "center";
    ctx.fillText(String(i + 1), x1 + 11, y1 + 16);
    ctx.restore();
  });
  // orientation labels: the patient's right is on the image left
  ctx.save(); ctx.fillStyle = "rgba(255,255,255,.8)"; ctx.font = "600 13px Public Sans, sans-serif";
  ctx.fillText("R", 10, 500); ctx.textAlign = "right"; ctx.fillText("L", 502, 500); ctx.restore();
}

// ------------------------------------------------------------------ the report sheet
const BAND_POS = { "very unlikely": [0, 10], "unlikely": [10, 30], "uncertain": [30, 70], "likely": [70, 90], "very likely": [90, 100] };

function show(data) {
  const rec = data.record, c = rec.classifier, p = rec.patient;
  stage.classList.remove("empty"); $("drop").hidden = true; cv.hidden = false; $("layers").hidden = false;
  draw();

  $("intro").hidden = true; $("result").hidden = false;
  const borderline = rec.notes.some((n) => n.startsWith("Borderline"));
  const pos = c.result === "pneumonia";
  $("verdict").className = "verdict " + (borderline ? "borderline" : pos ? "positive" : "negative");
  $("headline").textContent = borderline
    ? (pos ? "Possible pneumonia, borderline result" : "No clear pneumonia, borderline result")
    : (pos ? `Pneumonia ${c.band}` : "No pneumonia-type opacity detected");
  const sex = { F: "Female", M: "Male" }[p.sex] || "Sex not given";
  $("who").textContent = `${sex}, ${p.age != null ? p.age + " years" : "age not given"}, ${p.view === "unknown" ? "view unknown" : p.view + " view"}`;
  $("marker").style.left = `${Math.min(99, Math.max(1, c.probability * 100))}%`;
  $("prob").textContent = `Calibrated probability ${c.probability.toFixed(2)} (${c.band}). On held-out X-rays, results in each band were pneumonia about as often as the band says.`;

  const tb = $("findings").querySelector("tbody"); tb.innerHTML = "";
  rec.findings.forEach((f, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="${f.box_confidence === "clearly marked" ? "" : "dashed"}"><b>${i + 1}</b></td>
      <td>${cap(f.side)}</td><td>${cap(f.zones.join(" and "))}</td><td>${cap(f.extent)}</td><td>${cap(f.box_confidence)}</td>`;
    tb.appendChild(tr);
  });
  $("findings").hidden = rec.findings.length === 0;
  const nf = $("no-findings");
  nf.hidden = rec.findings.length > 0;
  nf.textContent = pos ? "No localised region was found; the change may be diffuse or subtle." : "No region is marked.";
  if (rec.localisation === "possible_location" && !rec.notes.some((n) => n.startsWith("The location is approximate")))
    rec.notes.unshift("The location is approximate: no region was marked clearly.");
  const notes = rec.notes.filter((n) => !n.startsWith("No localised region"));   // already shown above
  $("notes").innerHTML = notes.map((n) => `<li>${esc(n)}</li>`).join("");

  $("report").textContent = data.report;
  $("source").textContent = data.report_source === "llm"
    ? `Written by the fine-tuned GPT-Neo from the findings, then checked against them. Models ${data.seconds.models} s, report ${data.seconds.report} s.`
    : "Template report: the language model's text did not pass the check against the findings, so it is not shown.";
  $("pdf").href = `/api/report/${data.id}.pdf`;
  $("chat").innerHTML = "";
  $("result").scrollIntoView ? window.scrollTo({ top: 0 }) : null;
}

const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const esc = (s) => s.replace(/[&<>"]/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[m]));

// ------------------------------------------------------------------ chat
$("chips").addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") ask(e.target.textContent); });
$("ask").addEventListener("submit", (e) => { e.preventDefault(); ask($("q").value); $("q").value = ""; });

async function ask(question) {
  question = question.trim();
  if (!question || !state.id) return;
  const box = $("chat");
  box.insertAdjacentHTML("beforeend", `<div class="q">${esc(question)}</div>`);
  const a = document.createElement("div"); a.className = "a wait"; a.textContent = "Writing an answer"; box.appendChild(a);
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: state.id, question }) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "No answer.");
    a.className = "a"; a.textContent = d.answer;
    if (d.source === "fallback") a.insertAdjacentHTML("beforeend", "<small>The language model's answer did not pass the check against the findings, so the checked standard answer is shown.</small>");
  } catch (err) {
    a.className = "a"; a.textContent = err.message;
  }
}

// ------------------------------------------------------------------ reset, busy, errors
$("again").addEventListener("click", () => {
  Object.assign(state, { file: null, id: null, record: null, xray: null, heat: null });
  $("file").value = ""; $("age").value = ""; $("sex").value = ""; $("view").value = "";
  $("file-name").textContent = "No file chosen"; $("go").disabled = true;
  stage.classList.add("empty"); $("drop").hidden = false; cv.hidden = true; $("layers").hidden = true;
  $("result").hidden = true; $("intro").hidden = false; hideError();
});

function busy(on, text) {
  $("busy").hidden = !on; $("go").disabled = on || !state.file;
  if (text) $("busy-text").textContent = text;
}
function showError(msg) { $("error").textContent = msg; $("error").hidden = false; }
function hideError() { $("error").hidden = true; }
