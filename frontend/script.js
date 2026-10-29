const API_URL = "http://127.0.0.1:8000/predict/sleep-stress";

const INT_FIELDS = [
  "age", "quality_of_sleep", "physical_activity_level",
  "systolic", "diastolic", "heart_rate", "daily_steps",
];
const FLOAT_FIELDS = ["sleep_duration"];

const form = document.getElementById("form");
const submitBtn = document.getElementById("submit");
const errorBox = document.getElementById("error");
const panel = document.getElementById("result-panel");
const gaugeFg = document.getElementById("gauge-fg");
const scoreEl = document.getElementById("score");
const badgeEl = document.getElementById("category");
const notesEl = document.getElementById("notes");
const disclaimerEl = document.getElementById("disclaimer");

const ARC_LENGTH = Math.PI * 80;

function setState(state) {
  panel.dataset.state = state;
}

function syncRanges() {
  form.querySelectorAll('input[type="range"]').forEach((input) => {
    const out = form.querySelector(`output[data-for="${input.name}"]`);
    if (out) out.textContent = input.value + (out.dataset.unit || "");
  });
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
  setState("empty");
}

function buildPayload() {
  const data = Object.fromEntries(new FormData(form).entries());
  for (const key of INT_FIELDS) data[key] = parseInt(data[key], 10);
  for (const key of FLOAT_FIELDS) data[key] = parseFloat(data[key]);
  data.occupation = data.occupation.trim();
  return data;
}

function describeApiError(body) {
  if (Array.isArray(body.detail)) {
    return body.detail
      .map((d) => {
        const field = String(d.loc[d.loc.length - 1]).replaceAll("_", " ");
        return `${field}: ${d.msg}`;
      })
      .join(" | ");
  }
  return body.detail || "Something went wrong.";
}

function buildNotes(p, category) {
  const notes = [];
  if (p.sleep_duration < 7) {
    notes.push("Sleep duration is under 7 hours, below the range commonly recommended for adults.");
  }
  if (p.quality_of_sleep <= 5) {
    notes.push("Self-rated sleep quality is on the low side.");
  }
  if (p.daily_steps < 5000) {
    notes.push("Daily steps are fairly low. More movement is generally linked with better sleep and mood.");
  }
  if (p.sleep_disorder !== "None") {
    notes.push(`A sleep disorder (${p.sleep_disorder}) is worth discussing with a healthcare professional.`);
  }
  if (category === "High") {
    notes.push("If stress feels overwhelming, consider talking to someone you trust or a health professional.");
  }
  if (notes.length === 0) {
    notes.push("None of the main sleep or activity inputs stand out as unusual.");
  }
  return notes.slice(0, 3);
}

function renderResult(body, payload) {
  const score = body.predicted_stress_level;
  const category = body.category;

  scoreEl.textContent = score.toFixed(1);
  gaugeFg.style.strokeDashoffset = ARC_LENGTH * (1 - score / 10);
  gaugeFg.setAttribute("class", `gauge-fg show ${category.toLowerCase()}`);

  badgeEl.textContent = `${category} stress`;
  badgeEl.className = `badge ${category.toLowerCase()}`;

  notesEl.replaceChildren(
    ...buildNotes(payload, category).map((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      return li;
    })
  );
  disclaimerEl.textContent = body.disclaimer;
  setState("done");
}

function resetGauge() {
  scoreEl.textContent = "–";
  gaugeFg.style.strokeDashoffset = ARC_LENGTH;
  gaugeFg.setAttribute("class", "gauge-fg");
}

form.addEventListener("input", syncRanges);

form.addEventListener("reset", () => {
  setTimeout(syncRanges, 0);
  errorBox.hidden = true;
  resetGauge();
  setState("empty");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;

  const payload = buildPayload();
  const hasEmpty = Object.values(payload).some(
    (v) => v === "" || (typeof v === "number" && Number.isNaN(v))
  );
  if (hasEmpty) {
    showError("Please fill in every field.");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Estimating...";
  resetGauge();
  setState("loading");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();

    if (!response.ok) {
      showError(describeApiError(body));
      return;
    }
    renderResult(body, payload);
  } catch (err) {
    showError("Cannot reach the API. Make sure uvicorn is running on port 8000.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Estimate stress level";
  }
});

syncRanges();
resetGauge();