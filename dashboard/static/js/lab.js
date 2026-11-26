"use strict";

// Test, Experiments and Run history tabs.
(() => {
  const { $, el, control, toast, fmtSigned, fmtTime, title, hooks } = App;

  const COND = {
    fresh: { label: "Fresh brain", color: "#94a3c4" },
    pretrained_none: { label: "Experienced (no fix)", color: "#f472b6" },
    pretrained_reset_heads: { label: "Experienced + reset heads", color: "#34d399" },
    pretrained_shrink_perturb: { label: "Experienced + shrink & perturb", color: "#fbbf24" },
    pretrained_reset_optimizer: { label: "Experienced + reset optimizer", color: "#60a5fa" },
  };
  const condLabel = (c) => (COND[c] || { label: c }).label;
  const condColor = (c) => (COND[c] || { color: "#a78bfa" }).color;

  const L = {
    lab: null, checkpoints: [], experiments: [], selectedExp: null, expCharts: [], expPoll: null,
    testCharts: [], replay: null, jobsSig: "",
  };

  const getJSON = async (url) => {
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok || data.ok === false) throw new Error(data.error || "Request failed");
    return data;
  };
  const pct = (v) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);
  const ciText = (s, fmt = (v) => Charts.fixed(v, 2)) =>
    !s ? "—" : s.n > 1 ? `${fmt(s.mean)} <span class="ci">[${fmt(s.lo)}, ${fmt(s.hi)}]</span>` : fmt(s.mean);
  const kSteps = (v) => Charts.fmtSteps(v);

  // =====================================================================
  // TEST
  // =====================================================================
  function fillBrainSelects(items) {
    L.checkpoints = items;
    for (const sel of [$("test-brain"), $("replay-brain")]) {
      const prev = sel.value;
      sel.innerHTML = "";
      sel.append(el("option", { value: "__live__", text: "Live brain (current training)" }));
      const groups = {};
      for (const it of items) (groups[it.folder || "results"] ||= []).push(it);
      for (const [folder, list] of Object.entries(groups)) {
        const og = el("optgroup", { label: folder || "results" });
        for (const it of list) og.append(el("option", { value: it.name, text: `${it.name.split("/").pop()} · ${it.modified}` }));
        sel.append(og);
      }
      if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
    }
  }

  $("test-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    const btn = $("test-run");
    btn.disabled = true;
    $("test-status").textContent = "Playing test games…";
    const t0 = performance.now();
    const res = await control({ cmd: "evaluate", checkpoint: f.get("checkpoint"), episodes: Number(f.get("episodes")),
      seed: Number(f.get("seed")), mode: f.get("mode") });
    btn.disabled = false;
    if (!res.ok) { $("test-status").textContent = ""; return; }
    $("test-status").textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
    renderTest(res.result);
  });

  function renderTest(r) {
    $("test-label").textContent = `${r.label.split("/").pop()} · ${r.episodes} games each · ${r.mode === "greedy" ? "best move" : "sampled"}`;
    const box = $("test-results");
    box.innerHTML = "";
    const table = el("table", { class: "data" });
    table.innerHTML = `<thead><tr><th scope="col">Game</th><th scope="col">Brain score</th><th scope="col">Win rate</th>
      <th scope="col">Avg length</th><th scope="col">Random</th><th scope="col">Perfect</th><th scope="col">Skill</th></tr></thead>`;
    const body = el("tbody");
    const verdicts = [];
    for (const [g, d] of Object.entries(r.games)) {
      const m = d.model, rnd = d.random;
      const skill = d.optimal !== null ? (m.mean - rnd.mean) / (d.optimal - rnd.mean) : null;
      const tr = el("tr");
      const untrained = d.trained === false;
      tr.innerHTML = `<td><span class="swatch" style="background:${Games.color(g)}"></span>${title(g)}${untrained ? ' <span class="ci">(never trained on)</span>' : ""}</td>
        <td>${fmtSigned(m.mean)} <span class="ci">± ${m.std.toFixed(2)}</span></td>
        <td>${pct(m.success)}</td><td>${m.length.toFixed(1)} moves</td>
        <td>${fmtSigned(rnd.mean)} <span class="ci">(${pct(rnd.success)} wins)</span></td>
        <td>${fmtSigned(d.optimal)}</td>
        <td><span class="skill">${pct(skill)}</span><span class="skill-bar"><i style="width:${Math.max(0, Math.min(1, skill ?? 0)) * 100}%"></i></span></td>`;
      body.append(tr);
      if (untrained) { verdicts.push(`${title(g)}: this brain never trained on it.`); continue; }
      verdicts.push(skill === null ? `${title(g)}: no perfect score to compare with.`
        : skill >= 0.95 ? `${title(g)}: plays essentially perfectly.`
        : skill >= 0.6 ? `${title(g)}: clearly skilled, with room to improve.`
        : skill >= 0.15 ? `${title(g)}: learning, but far from perfect.`
        : `${title(g)}: about as good as random play.`);
    }
    table.append(body);
    box.append(el("div", { class: "table-wrap" }, [table]));
    box.append(el("p", { class: "muted", text: verdicts.join(" ") }));
    const grid = el("div", { class: "hist-grid" });
    L.testCharts = [];
    for (const [g, d] of Object.entries(r.games)) {
      const canvas = el("canvas", { role: "img", "aria-label": `${title(g)} score distribution` });
      grid.append(el("div", { class: "hist-card", style: `--game-color:${Games.color(g)}` }, [
        el("h3", { text: `${title(g)}: score of each game` }), canvas]));
      L.testCharts.push(() => Charts.bars(canvas, { range: r.hist_range, series: [
        { counts: d.model.hist, color: Games.color(g) }, { counts: d.random.hist, color: "#56607c" }] }));
    }
    box.append(grid);
    requestAnimationFrame(() => L.testCharts.forEach((fn) => fn()));
  }

  // ---------- replay ----------
  $("replay-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const game = $("replay-game").value;
    const res = await control({ cmd: "record", checkpoint: $("replay-brain").value, game,
      seed: Number($("replay-seed").value), mode: $("replay-mode").value });
    if (!res.ok) return;
    stopReplay();
    const init = App.getInit();
    const data = res.result;
    const brain = new Brain.BrainView($("replay-brain-canvas"), $("replay-tooltip"), init.actions, init.channels);
    brain.setBrain(data.brain);
    L.replay = { data, i: 0, timer: null, renderer: new Games.Renderer($("replay-canvas"), game), brain };
    $("replay-slider").max = String(data.frames.length - 1);
    showStep(0);
    toast(`Recorded ${data.frames.length - 1} moves · ${data.success ? "won" : "did not win"} · score ${fmtSigned(data.total)}`);
  });

  function showStep(i) {
    const R = L.replay;
    if (!R) return;
    const frames = R.data.frames;
    i = Math.max(0, Math.min(frames.length - 1, i));
    R.i = i;
    const f = frames[i];
    R.renderer.push(f, 300);
    const thinking = f.acts ? f : frames[Math.min(1, frames.length - 1)];
    R.brain.setFrame(thinking, Games.color(R.data.game));
    $("replay-slider").value = String(i);
    const init = App.getInit();
    if (f.reset) {
      $("replay-info").innerHTML = `<b>Start</b> · the board before the first move. Press ▶ to play.`;
      return;
    }
    const probs = f.valid.map((a) => `${init.actions[a]} ${(f.probs[a] * 100).toFixed(0)}%`).join(" · ");
    $("replay-info").innerHTML =
      `<b>Move ${f.step}</b> of ${frames.length - 1} · chose <b>${f.action_name}</b><br>` +
      `Probabilities: ${probs}<br>Value guess: ${fmtSigned(f.value, 3)} · reward ${fmtSigned(f.reward)} · total ${fmtSigned(f.ep_return)}` +
      (f.event ? `<br>Result: <b>${f.event}</b>` : "");
  }

  function stopReplay() {
    if (L.replay?.timer) clearInterval(L.replay.timer);
    if (L.replay) L.replay.timer = null;
    $("replay-play").textContent = "▶";
  }

  document.querySelectorAll("[data-step]").forEach((b) => b.addEventListener("click", () => {
    const R = L.replay;
    if (!R) return;
    const n = R.data.frames.length - 1;
    const action = b.dataset.step;
    if (action === "play") {
      if (R.timer) return stopReplay();
      if (R.i >= n) showStep(0);
      $("replay-play").textContent = "❚❚";
      R.timer = setInterval(() => { if (R.i >= n) stopReplay(); else showStep(R.i + 1); }, 450);
      return;
    }
    stopReplay();
    showStep({ first: 0, prev: R.i - 1, next: R.i + 1, last: n }[action]);
  }));
  $("replay-slider").addEventListener("input", (e) => { stopReplay(); showStep(Number(e.target.value)); });

  // =====================================================================
  // EXPERIMENTS
  // =====================================================================
  function buildExperimentForm(lab) {
    const chip = (name, value, label, checked, color, desc) => {
      const input = el("input", { type: "checkbox", name, value });
      input.checked = checked;
      const children = [input, desc ? el("span", { class: "chip-col" }, [el("span", { text: label }), el("span", { class: "chip-desc", text: desc })]) : label];
      return el("label", { class: "chip", style: color ? `--chip-color:${color}` : "" }, children);
    };
    $("exp-phase-a").innerHTML = "";
    $("exp-phase-b").innerHTML = "";
    $("exp-interventions").innerHTML = "";
    for (const g of lab.games) {
      $("exp-phase-a").append(chip("phase_a", g, title(g), false, Games.color(g)));
      $("exp-phase-b").append(chip("phase_b", g, title(g), false, Games.color(g)));
    }
    const IV_LABELS = { none: "No fix (baseline)", reset_heads: "Reset heads", shrink_perturb: "Shrink & perturb", reset_optimizer: "Reset optimizer" };
    for (const [iv, desc] of Object.entries(lab.interventions)) {
      $("exp-interventions").append(chip("interventions", iv, IV_LABELS[iv] || iv, false, condColor(`pretrained_${iv}`), desc));
    }
    const form = $("exp-form");
    for (const [key, [lo, hi]] of Object.entries(lab.spec_limits)) {
      const input = form.elements[key];
      if (input) { input.min = lo; input.max = hi; }
    }
    applyPreset(lab.quick_spec);
  }

  function applyPreset(spec) {
    const form = $("exp-form");
    for (const [k, v] of Object.entries(spec)) {
      if (Array.isArray(v)) {
        for (const box of form.querySelectorAll(`input[name="${k}"]`)) box.checked = v.includes(box.value);
      } else if (form.elements[k]) form.elements[k].value = v;
    }
    updateEstimate();
  }

  function readSpec() {
    const form = $("exp-form");
    const checked = (name) => [...form.querySelectorAll(`input[name="${name}"]:checked`)].map((x) => x.value);
    const num = (name) => Number(form.elements[name].value);
    return {
      name: form.elements.name.value.trim(), phase_a: checked("phase_a"), phase_b: checked("phase_b"),
      interventions: checked("interventions"), steps_a: num("steps_a"), steps_b: num("steps_b"),
      seeds: num("seeds"), eval_every: num("eval_every"), eval_episodes: num("eval_episodes"),
      envs_per_game: num("envs_per_game"), lr: num("lr"), ent_coef: num("ent_coef"),
    };
  }

  function updateEstimate() {
    if (!L.lab) return;
    const s = readSpec();
    const runs = s.seeds * (s.interventions.length + 1);
    const moves = s.seeds * (s.steps_a + s.steps_b * (s.interventions.length + 1));
    const sps = L.lab.last_sps || 1000;
    const minutes = (moves / sps) * 1.15 / 60; // +15% for the test games
    const problems = [];
    if (!s.phase_a.length) problems.push("pick a phase A game");
    if (!s.phase_b.length) problems.push("pick a phase B game");
    if (!s.interventions.length) problems.push("pick at least one fix (\"No fix\" is the baseline)");
    $("exp-estimate").textContent = problems.length ? `To start: ${problems.join(", ")}.`
      : `${runs} training runs · ${Charts.fmtSteps(moves)} moves · roughly ${minutes < 1 ? "<1" : Math.round(minutes)} min ` +
        `(based on ${Math.round(sps).toLocaleString()} moves/s${L.lab.last_sps ? ", measured" : ", estimated"}).`;
  }

  $("exp-form").addEventListener("input", updateEstimate);
  document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
    if (L.lab) applyPreset(b.dataset.preset === "quick" ? L.lab.quick_spec : L.lab.default_spec);
  }));
  $("exp-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const res = await control({ cmd: "experiment", spec: readSpec() });
    if (!res.ok) return;
    toast("Experiment added to the queue.");
    await refreshExperiments();
    selectExperiment(res.id);
  });

  // ---------- queue ----------
  function renderJobs(jobs) {
    const active = jobs.find((j) => j.status === "running") || jobs.find((j) => j.status === "waiting");
    const pill = $("job-pill");
    if (active) {
      pill.hidden = false;
      pill.textContent = active.status === "running" ? `⚗ ${active.name} · ${Math.round(active.progress * 100)}%` : `⚗ ${active.name} · waiting`;
    } else pill.hidden = true;

    // refresh results when a job changes state
    const sig = jobs.map((j) => `${j.id}:${j.status}`).join("|");
    if (sig !== L.jobsSig) {
      L.jobsSig = sig;
      refreshExperiments();
      if (L.selectedExp) loadExperiment(L.selectedExp);
    }
    if ($("tab-experiments").hidden) return;

    // Update rows in place. Rebuilding them 12×/s would swap a button out from under the
    // mouse between press and release, so clicks would never register.
    const list = $("job-list");
    const empty = list.querySelector(".job-empty");
    if (!jobs.length) {
      if (!empty) { list.innerHTML = ""; list.append(el("li", { class: "muted job-empty", text: "No experiments queued." })); }
      return;
    }
    if (empty) empty.remove();
    const ordered = [...jobs].reverse();
    const wanted = new Set(ordered.map((j) => j.id));
    for (const row of [...list.children]) if (!wanted.has(row.dataset.id)) row.remove();
    ordered.forEach((j, index) => {
      let row = list.querySelector(`li[data-id="${CSS.escape(j.id)}"]`);
      if (!row) {
        row = el("li", { class: "job", "data-id": j.id }, [
          el("div", { class: "job-top" }, [el("b", { class: "job-name" }),
            el("span", { class: "run-actions" }, [
              el("span", { class: "badge" }),
              el("button", { type: "button", class: "btn small", "data-action": "view", text: "View" }),
              el("button", { type: "button", class: "btn small", "data-action": "cancel", text: "Cancel" })])]),
          el("div", { class: "bar" }, [el("div", { class: "bar-fill" })]),
          el("div", { class: "job-detail" }),
        ]);
      }
      if (list.children[index] !== row) list.insertBefore(row, list.children[index] || null);
      const set = (sel, prop, value) => { const node = row.querySelector(sel); if (node[prop] !== value) node[prop] = value; };
      set(".job-name", "textContent", j.name);
      set(".badge", "className", `badge ${j.status}`);
      set(".badge", "textContent", j.status);
      set('[data-action="cancel"]', "hidden", !["queued", "waiting", "running"].includes(j.status));
      row.querySelector(".bar-fill").style.width = `${j.progress * 100}%`;
      const eta = j.status === "running" && j.eta ? ` · ~${fmtTime(j.eta)} left` : "";
      set(".job-detail", "textContent",
        `${Math.round(j.progress * 100)}% · ${j.detail}${j.sps ? ` · ${j.sps.toLocaleString()} moves/s` : ""}${eta}`);
    });
  }

  $("job-list").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const id = btn.closest("li").dataset.id;
    const job = (L.lastJobs || []).find((j) => j.id === id);
    if (btn.dataset.action === "view") {
      selectExperiment(id);
      $("exp-select").scrollIntoView({ behavior: "smooth", block: "center" });
    } else if (btn.dataset.action === "cancel") {
      if (confirm(`Cancel "${job ? job.name : id}"? Runs that already finished stay saved.`)) {
        control({ cmd: "cancel_job", id }).then((r) => { if (r.ok) toast("Cancelling. It stops after the current update."); });
      }
    }
  });

  // ---------- results ----------
  async function refreshExperiments() {
    try {
      L.experiments = await getJSON("/api/experiments");
    } catch { return; }
    const sel = $("exp-select");
    const prev = L.selectedExp || sel.value;
    sel.innerHTML = "";
    sel.append(el("option", { value: "", text: L.experiments.length ? "Choose…" : "No experiments yet" }));
    for (const m of L.experiments) sel.append(el("option", { value: m.id, text: `${m.name} · ${m.created} · ${m.status}` }));
    if (prev && L.experiments.some((m) => m.id === prev)) sel.value = prev;
  }

  $("exp-select").addEventListener("change", (e) => selectExperiment(e.target.value));

  function selectExperiment(id) {
    L.selectedExp = id || null;
    $("exp-select").value = id || "";
    clearInterval(L.expPoll);
    if (!id) return;
    App.showTab("experiments");
    history.replaceState(null, "", `#experiments/${encodeURIComponent(id)}`); // shareable link
    loadExperiment(id);
    // keep updating while it runs
    L.expPoll = setInterval(() => {
      const meta = L.experiments.find((m) => m.id === id);
      const job = L.lastJobs?.find((j) => j.id === id);
      const live = (job && ["running", "waiting", "queued"].includes(job.status)) || (meta && ["running", "queued", "waiting"].includes(meta.status));
      if (live && !$("tab-experiments").hidden) loadExperiment(id);
    }, 4000);
  }

  async function loadExperiment(id) {
    let data;
    try {
      data = await getJSON(`/api/experiment?id=${encodeURIComponent(id)}`);
    } catch (err) { toast(String(err.message || err), true); return; }
    if (id !== L.selectedExp) return;
    renderExperiment(data.meta, data.summary);
  }

  function renderExperiment(meta, sum) {
    const spec = meta.spec || sum?.spec;
    $("exp-export").hidden = false;
    $("exp-export").href = `/api/export?kind=experiment&id=${encodeURIComponent(meta.id)}`;
    $("exp-notes-box").hidden = false;
    if (document.activeElement !== $("exp-notes")) $("exp-notes").value = meta.notes || "";
    $("exp-header").innerHTML = "";
    $("exp-header").append(
      el("span", { class: `badge ${meta.status}`, text: meta.status }),
      el("b", { text: meta.name }),
      el("span", { text: spec ? `Phase A: ${spec.phase_a.map(title).join(" + ")} (${kSteps(spec.steps_a)}) → Phase B: ${spec.phase_b.map(title).join(" + ")} (${kSteps(spec.steps_b)}) · ${spec.seeds} seed${spec.seeds > 1 ? "s" : ""}` : "" }),
      el("span", { text: sum ? `${sum.runs_done}/${sum.runs_total} runs complete` : "" }),
      el("span", { text: meta.created || "" }));
    if (!sum) {
      $("exp-verdict").hidden = true;
      $("exp-table").hidden = true;
      $("exp-charts").innerHTML = "";
      return;
    }
    renderVerdict(sum);
    renderTable(sum);
    renderCharts(sum);
  }

  function renderVerdict(sum) {
    const box = $("exp-verdict");
    const items = [];
    const base = sum.metrics.pretrained_none;
    const n = base?.seeds.length || 0;
    for (const g of sum.games_b) {
      const r = base?.games[g]?.auc_ratio;
      if (!r) {
        if (n) items.push(`<b>Can't compare learning speed on ${title(g)} yet:</b> the fresh brain barely learned it in phase B, so there's nothing to compare against. Make phase B longer.`);
        continue;
      }
      const f2 = (v) => Charts.fixed(v, 2);
      const range = r.n > 1 ? ` (ratio ${f2(r.mean)}, 95% CI ${f2(r.lo)} to ${f2(r.hi)})` : ` (ratio ${f2(r.mean)}, one seed)`;
      const fresh = sum.metrics.fresh?.games[g];
      const exp = base.games[g];
      const reachedText = `fresh reached 90% in ${fresh.reached90}/${fresh.steps90?.n ?? 0} seeds, experienced in ${exp.reached90}/${exp.steps90?.n ?? 0}`;
      if (r.n > 1 && r.hi < 1) {
        const how = r.mean < 0.25 ? "barely learned it at all in phase B, while the fresh brain did"
          : `learned it about ${Math.round((1 - r.mean) * 100)}% slower than a fresh brain`;
        items.push(`<b>Plasticity loss on ${title(g)}:</b> the experienced brain ${how}${range}; ${reachedText}.`);
      } else if (r.n > 1 && r.lo > 1) {
        items.push(`<b>Transfer on ${title(g)}:</b> the experienced brain learned it about ${Math.round((r.mean - 1) * 100)}% faster than a fresh brain${range}; ${reachedText}.`);
      } else {
        items.push(`<b>No clear difference on ${title(g)}:</b> the interval includes 1${range}. More seeds or longer phases may tell.`);
      }
      for (const c of sum.conditions) {
        if (c === "fresh" || c === "pretrained_none") continue;
        const rc = sum.metrics[c]?.games[g]?.auc_ratio;
        if (!rc) continue;
        const better = rc.mean > r.mean;
        items.push(`${condLabel(c)}: ratio ${Charts.fixed(rc.mean, 2)}, ${better ? "faster" : "slower"} than no fix (${Charts.fixed(r.mean, 2)}).`);
      }
    }
    for (const [g, f] of Object.entries(base?.forgetting || {})) {
      if (!f) continue;
      items.push(f.mean > 0.2 ? `<b>Forgetting:</b> after phase B, the ${title(g)} score dropped by ${Charts.fixed(f.mean, 2)} on average.`
        : `<b>${title(g)} was mostly remembered</b> (score change ${fmtSigned(-f.mean)}).`);
    }
    if (!items.length) {
      box.hidden = false;
      box.innerHTML = `<h3>Waiting for results</h3><p>Conclusions appear once the fresh and experienced runs of at least one seed are complete. Curves below update live.</p>`;
      return;
    }
    if (n < 3) items.push(`<span class="warn">Only ${n} seed${n === 1 ? "" : "s"} so far: treat this as a first look, not a result. Use 5+ seeds for your paper.</span>`);
    box.hidden = false;
    box.innerHTML = `<h3>What the data says</h3><ul>${items.map((t) => `<li>${t}</li>`).join("")}</ul>`;
  }

  function renderTable(sum) {
    const table = $("exp-table");
    const headB = sum.games_b.map((g) => `<th scope="col">${title(g)}: speed vs fresh</th><th scope="col">${title(g)}: moves to 90%</th><th scope="col">${title(g)}: final</th>`).join("");
    const forgetGames = sum.games_a.filter((g) => !sum.games_b.includes(g));
    const headF = forgetGames.map((g) => `<th scope="col">${title(g)} forgotten</th>`).join("");
    let html = `<thead><tr><th scope="col">Condition</th>${headB}${headF}<th scope="col">Exploration at switch</th><th scope="col">Dormant at switch</th><th scope="col">Feature rank at switch</th><th scope="col">Seeds</th></tr></thead><tbody>`;
    for (const c of sum.conditions) {
      const m = sum.metrics[c];
      html += `<tr><td><span class="swatch" style="background:${condColor(c)}"></span>${condLabel(c)}</td>`;
      for (const g of sum.games_b) {
        const e = m.games[g];
        html += `<td>${c === "fresh" ? "1.00 (reference)" : e.auc_ratio ? ciText(e.auc_ratio) : '<span class="ci" title="The fresh brain barely learned this game, so a ratio would be meaningless">n/a</span>'}</td>`;
        html += `<td>${ciText(e.steps90, kSteps)} <span class="ci">${e.steps90 ? `${e.reached90}/${e.steps90.n} reached` : ""}</span></td>`;
        html += `<td>${ciText(e.final, (v) => fmtSigned(v))}</td>`;
      }
      for (const g of forgetGames) html += `<td>${c === "fresh" ? "—" : ciText(m.forgetting[g], (v) => fmtSigned(v))}</td>`;
      html += `<td>${ciText(m.switch.entropy)}</td>`;
      html += `<td>${ciText(m.switch.dormant_fc, (v) => `${(v * 100).toFixed(1)}%`)}</td>`;
      html += `<td>${ciText(m.switch.eff_rank, (v) => v.toFixed(0))}</td><td>${m.seeds.length}</td></tr>`;
    }
    table.innerHTML = html + "</tbody>";
    table.hidden = false;
  }

  /** Turn {cond: {x, mean, lo, hi}} into chart rows keyed by condition. */
  function curveRows(byCond) {
    const rows = new Map();
    for (const [c, cur] of Object.entries(byCond)) {
      cur.x.forEach((x, i) => {
        const row = rows.get(x) || { step: x };
        row[c] = cur.mean[i];
        row[`${c}_lo`] = cur.lo[i];
        row[`${c}_hi`] = cur.hi[i];
        rows.set(x, row);
      });
    }
    return [...rows.values()].sort((a, b) => a.step - b.step);
  }

  function renderCharts(sum) {
    const box = $("exp-charts");
    const series = (conds) => conds.map((c) => ({ key: c, lo: `${c}_lo`, hi: `${c}_hi`, label: condLabel(c), color: condColor(c) }));
    const defs = [];
    const switchMarker = sum.switch_step ? [{ x: sum.switch_step, label: "switch to phase B" }] : [];
    for (const g of sum.games_b) {
      const b = sum.baselines[g];
      defs.push({ id: `learn-${g}`, title: `Learning ${title(g)} in phase B`, rows: curveRows(sum.curves.learn[g]), series: series(sum.conditions),
        refs: [{ value: b.random, color: "#94a3c4" }, { value: b.optimal, color: "#94a3c4" }],
        explain: `Test score on ${title(g)} from the start of phase B. If the experienced curves rise slower than the fresh one, that's plasticity loss. Dashed: random and perfect play.` });
    }
    for (const g of Object.keys(sum.curves.remember)) {
      defs.push({ id: `remember-${g}`, title: `Remembering ${title(g)}`, rows: curveRows(sum.curves.remember[g]), series: series(sum.conditions), markers: switchMarker,
        explain: `Test score on ${title(g)} across the whole run. Experienced brains learn it in phase A; a drop after the switch is forgetting. The fresh brain never trained on it.` });
    }
    defs.push({ id: "dormant", title: "Dormant hidden neurons", rows: curveRows(sum.curves.timeline.dormant_fc), series: series(sum.conditions), markers: switchMarker, yMin: 0,
      format: (v) => `${(v * 100).toFixed(0)}%`, explain: "Share of the 128 hidden neurons that are almost silent on a fixed probe set of boards. A possible mechanism behind plasticity loss." });
    defs.push({ id: "entropy", title: "Exploration (entropy)", rows: curveRows(sum.curves.timeline.entropy), series: series(sum.conditions), markers: switchMarker, yMin: 0,
      explain: "How spread out the move probabilities are. A brain that became confident in phase A starts phase B exploring much less than a fresh one, which can slow down learning the new game." });
    defs.push({ id: "rank", title: "Feature rank", rows: curveRows(sum.curves.timeline.eff_rank), series: series(sum.conditions), markers: switchMarker, yMin: 0,
      format: (v) => v.toFixed(0), explain: "How many independent features the hidden layer uses. Lower means less capacity left to learn something new." });

    const sig = defs.map((d) => d.id).join("|") + sum.conditions.join(",");
    if (L.chartSig !== sig) {
      L.chartSig = sig;
      box.innerHTML = "";
      L.expCharts = defs.map((d) => {
        const canvas = el("canvas", { role: "img", "aria-label": `${d.title} chart` });
        const legend = el("div", { class: "legend" }, d.series.map((s) => el("span", {}, [el("i", { style: `background:${s.color}` }), s.label])));
        box.append(el("div", { class: "panel chart-card" }, [el("div", { class: "panel-head" }, [el("h2", { text: d.title })]), legend,
          el("p", { class: "explain", text: d.explain }), canvas]));
        return new Charts.LineChart(canvas, { series: d.series, refs: d.refs || [], markers: d.markers || [], yMin: d.yMin, format: d.format,
          emptyText: "No data yet" });
      });
    }
    defs.forEach((d, i) => { L.expCharts[i].markers = d.markers || []; L.expCharts[i].setRows(d.rows); });
  }

  $("exp-notes-save").addEventListener("click", async () => {
    if (!L.selectedExp) return;
    const res = await control({ cmd: "notes", kind: "experiment", id: L.selectedExp, text: $("exp-notes").value });
    if (res.ok) toast("Notes saved.");
  });

  // =====================================================================
  // RUN HISTORY
  // =====================================================================
  async function refreshRuns() {
    let data;
    try { data = await getJSON("/api/runs"); } catch { return; }
    const runsBox = $("runs-list");
    runsBox.innerHTML = "";
    if (!data.runs.length) runsBox.append(el("p", { class: "muted", text: "No training runs yet. Start one on the Train tab." }));
    for (const r of data.runs) runsBox.append(runCard("run", r));
    const expBox = $("exps-list");
    expBox.innerHTML = "";
    if (!data.experiments.length) expBox.append(el("p", { class: "muted", text: "No experiments yet. Queue one on the Experiments tab." }));
    for (const m of data.experiments) expBox.append(runCard("experiment", m));
  }

  function runCard(kind, m) {
    const meta = [];
    if (kind === "run") {
      meta.push(`created ${m.created}`, `update ${m.update ?? 0}/${m.num_updates ?? "?"}`, `${Charts.fmtSteps(m.env_steps || 0)} moves`, `${m.minutes ?? 0} min`);
      for (const [g, v] of Object.entries(m.scores || {})) meta.push(`${title(g)} ${fmtSigned(v)}`);
      if (m.config) meta.push(`seed ${m.config.seed}, lr ${m.config.lr}`);
    } else {
      const s = m.spec || {};
      meta.push(`created ${m.created}`, `${(s.phase_a || []).map(title).join("+")} → ${(s.phase_b || []).map(title).join("+")}`,
        `${s.seeds} seeds`, `fixes: ${(s.interventions || []).join(", ")}`, `${Math.round((m.progress || 0) * 100)}%`);
    }
    const actions = [];
    const exportLink = el("a", { class: "btn small", href: `/api/export?kind=${kind}&id=${encodeURIComponent(m.id)}`, download: "", text: "⤓ CSV" });
    actions.push(exportLink);
    if (kind === "run" && m.checkpoints?.length) {
      const test = el("button", { type: "button", class: "btn small", text: "Test latest checkpoint" });
      test.addEventListener("click", () => {
        App.showTab("test");
        const latest = m.checkpoints[m.checkpoints.length - 1];
        $("test-brain").value = latest;
        $("replay-brain").value = latest;
        toast(`Selected ${latest}. Press "Run test".`);
      });
      actions.push(test);
    }
    if (kind === "experiment") {
      const open = el("button", { type: "button", class: "btn small", text: "Open results" });
      open.addEventListener("click", () => selectExperiment(m.id));
      actions.push(open);
    }
    const notes = el("textarea", { rows: "3", placeholder: "Notes: what you tried, what you saw, what's next" });
    notes.value = m.notes || "";
    const save = el("button", { type: "button", class: "btn small", text: "Save notes" });
    save.addEventListener("click", async () => {
      const res = await control({ cmd: "notes", kind, id: m.id, text: notes.value });
      if (res.ok) toast("Notes saved.");
    });
    const details = el("details", {}, [el("summary", { text: m.notes ? "Notes ✎" : "Add notes" }), notes, save]);
    if (m.notes) details.open = true;
    return el("div", { class: "run-card" }, [
      el("div", { class: "run-top" }, [
        el("span", {}, [el("span", { class: `badge ${m.status || "idle"}`, text: m.status || "idle" }), " ",
          el("span", { class: "name", text: kind === "experiment" ? `${m.name} (${m.id})` : m.id })]),
        el("div", { class: "run-actions" }, actions)]),
      el("div", { class: "run-meta" }, meta.map((t) => el("span", { text: t }))),
      details,
    ]);
  }
  $("runs-refresh").addEventListener("click", refreshRuns);

  // =====================================================================
  // wiring
  // =====================================================================
  hooks.checkpoints.push(fillBrainSelects);
  hooks.jobs.push((jobs) => { L.lastJobs = jobs; renderJobs(jobs); });
  hooks.init.push(async () => {
    if (L.lab) return;
    try { L.lab = await getJSON("/api/lab"); } catch { return; }
    $("replay-game").innerHTML = "";
    for (const g of L.lab.games) $("replay-game").append(el("option", { value: g, text: title(g) }));
    buildExperimentForm(L.lab);
    refreshExperiments();
  });
  hooks.tab.push(async (name) => {
    if (name === "runs") refreshRuns();
    if (name === "experiments") {
      if (L.lab) {
        try { L.lab.last_sps = (await getJSON("/api/lab")).last_sps; } catch { /* keep the old estimate */ }
        updateEstimate();
      }
      refreshExperiments();
      for (const c of L.expCharts) c.dirty = true;
      if (L.lastJobs) renderJobs(L.lastJobs);
    }
    if (name === "test") {
      App.refreshCheckpoints();
      requestAnimationFrame(() => L.testCharts.forEach((fn) => fn()));
    }
  });
  hooks.tick.push((now, tab) => {
    if (tab === "test" && L.replay) { L.replay.renderer.draw(now); L.replay.brain.draw(now); }
    if (tab === "experiments") for (const c of L.expCharts) c.draw();
  });
  window.addEventListener("resize", () => {
    for (const c of L.expCharts) c.dirty = true;
    L.testCharts.forEach((fn) => fn());
  });

  // the page may have opened directly on a lab tab (e.g. #runs or #experiments/<id>)
  const [current, linkedId] = location.hash.slice(1).split("/");
  if (current && current !== "train") for (const fn of hooks.tab) fn(current);
  if (current === "experiments" && linkedId) selectExperiment(decodeURIComponent(linkedId));
})();
