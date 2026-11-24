"use strict";

// Connects to the Python server, receives live updates, and wires up the page.
(() => {
  const $ = (id) => document.getElementById(id);

  const CONFIG_HELP = {
    lr: "Learning rate: size of each gradient step",
    ent_coef: "Entropy bonus: reward for exploring",
    clip: "PPO clip ε: max policy change per update",
    gamma: "Discount γ: how much future reward counts",
    lam: "GAE λ: smooths the advantage estimate",
    total_steps: "Total moves to train for",
    envs_per_game: "Parallel copies of each game",
    rollout_steps: "Moves per copy before each update",
    epochs: "Passes over each batch",
    minibatches: "Splits of the batch per pass",
    seed: "Random seed (same seed = same run)",
  };

  const S = {
    init: null, run: null, history: [], frames: {}, renderers: {}, focus: null,
    brain: null, status: "offline", demoFps: 6, charts: [], arena: [], lastUpdate: 0, validBuilt: {},
  };

  const brainView = new Brain.BrainView($("brain-canvas"), $("brain-tooltip"), [], []);

  // Other scripts (lab.js) subscribe to these.
  const hooks = { init: [], jobs: [], checkpoints: [], tab: [], tick: [] };
  S.tab = "train";

  function showTab(name) {
    name = String(name).split("/")[0]; // "#experiments/<id>" opens the Experiments tab
    if (!document.getElementById(`tab-${name}`)) name = "train";
    S.tab = name;
    for (const b of document.querySelectorAll("[data-tab]")) {
      const on = b.dataset.tab === name;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
    }
    for (const p of document.querySelectorAll(".tab-panel")) p.hidden = p.id !== `tab-${name}`;
    if (location.hash.slice(1).split("/")[0] !== name) history.replaceState(null, "", `#${name}`);
    for (const c of S.charts) c.dirty = true;
    for (const fn of hooks.tab) fn(name);
  }

  // ---------- helpers ----------
  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "style") node.style.cssText = v;
      else node.setAttribute(k, v);
    }
    for (const child of [].concat(children)) node.append(child);
    return node;
  }

  const fmtInt = (n) => Math.round(n).toLocaleString();
  const fmtSigned = (v, d = 2) => {
    if (v === null || v === undefined || !Number.isFinite(v)) return "—";
    const s = Charts.fixed(v, d);
    return s.startsWith("-") ? s : `+${s}`;
  };
  function fmtTime(sec) {
    sec = Math.floor(sec);
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
  }
  const title = (g) => Games.TITLES[g] || g;

  let toastTimer = null;
  function toast(text, error = false) {
    const t = $("toast");
    t.textContent = text;
    t.className = "toast" + (error ? " error" : "");
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, 3500);
  }

  async function control(body) {
    try {
      const res = await fetch("/api/control", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!data.ok) toast(data.error || "Request failed", true);
      return data;
    } catch {
      toast("Can't reach the server. Is it still running?", true);
      return { ok: false };
    }
  }

  // ---------- page setup (runs again after every reset) ----------
  function setup(init) {
    S.init = init;
    S.history = [];
    S.frames = {};
    S.validBuilt = {};
    if (!S.focus || !init.games.includes(S.focus)) S.focus = init.games[0];
    brainView.actions = init.actions;
    brainView.channels = init.channels;
    $("kpi-params").textContent = fmtInt(init.params);
    buildGames(init);
    buildKpiGames(init);
    buildFocus(init);
    buildArena(init);
    buildCharts(init);
    buildConfig(init);
    refreshCheckpoints();
  }

  function buildGames(init) {
    const box = $("games");
    box.innerHTML = "";
    S.renderers = {};
    for (const g of init.games) {
      const color = Games.color(g);
      const canvas = el("canvas", { class: "game-canvas", role: "img", "aria-label": `${title(g)}, played live by the agent` });
      const card = el("article", { class: "game-card", style: `--game-color:${color}` }, [
        el("h3", {}, [title(g), el("span", { class: "sub", id: `g-${g}-ep`, text: "episode —" })]),
        canvas,
        el("p", { class: "explain", text: Games.DESCRIPTIONS[g] || "" }),
        el("div", { class: "game-stats" }, [
          el("div", { class: "stat" }, [el("span", { text: "This episode" }), el("b", { id: `g-${g}-ret`, text: "—" })]),
          el("div", { class: "stat" }, [el("span", { text: "Avg last 50" }), el("b", { id: `g-${g}-avg`, text: "—" })]),
          el("div", { class: "stat" }, [el("span", { text: "Win rate" }), el("b", { id: `g-${g}-win`, text: "—" })]),
        ]),
        el("ul", { class: "probs", id: `g-${g}-probs`, "aria-label": `${title(g)} move probabilities` }),
        el("div", { class: "value-gauge" }, [
          "Value guess ", el("b", { id: `g-${g}-value`, text: "—" }),
          el("div", { class: "track" }, [el("div", { class: "needle", id: `g-${g}-needle` })]),
        ]),
      ]);
      box.append(card);
      S.renderers[g] = new Games.Renderer(canvas, g);
    }
  }

  function buildKpiGames(init) {
    const box = $("kpi-games");
    box.innerHTML = "";
    for (const g of init.games) {
      const b = init.baselines[g];
      box.append(el("div", { class: "kpi-game", style: `--game-color:${Games.color(g)}` }, [
        el("span", { class: "kpi-label", text: `${title(g)} score` }),
        el("span", { class: "kpi-value", id: `kpi-score-${g}`, text: "—" }),
        el("span", { class: "kpi-sub", text: `random ${fmtSigned(b.random)} · perfect ${fmtSigned(b.optimal)}` }),
        el("div", { class: "mini-progress", title: "Progress from random to perfect" }, [el("div", { id: `kpi-prog-${g}` })]),
      ]));
    }
  }

  function buildFocus(init) {
    const box = $("focus-buttons");
    box.innerHTML = "";
    for (const g of init.games) {
      const b = el("button", { type: "button", class: g === S.focus ? "on" : "", text: title(g), "aria-pressed": String(g === S.focus) });
      b.addEventListener("click", () => {
        S.focus = g;
        for (const x of box.children) { x.classList.toggle("on", x === b); x.setAttribute("aria-pressed", String(x === b)); }
        if (S.frames[g]) brainView.setFrame(S.frames[g], Games.color(g));
      });
      box.append(b);
    }
  }

  function buildArena(init) {
    const box = $("arena");
    box.innerHTML = "";
    S.arena = [];
    const names = init.games.flatMap((g) => Array(init.config.envs_per_game).fill(g));
    names.forEach((g, i) => {
      const canvas = el("canvas", { role: "img", "aria-label": `Training copy ${i + 1}: ${title(g)}` });
      const info = el("span", { text: "—" });
      box.append(el("div", { class: "arena-cell", style: `--game-color:${Games.color(g)}` }, [
        canvas, el("div", {}, [el("span", { text: `${title(g)} #${i + 1}` }), info]),
      ]));
      S.arena.push({ canvas, info });
    });
    $("arena-count").textContent = `${names.length} games in parallel`;
  }

  function buildCharts(init) {
    const box = $("charts");
    box.innerHTML = "";
    const refs = [];
    for (const g of init.games) {
      const b = init.baselines[g], color = Games.color(g);
      refs.push({ value: b.random, color }, { value: b.optimal, color });
    }
    const pct = (v) => (v * 100).toFixed(0) + "%";
    const defs = [
      { title: "Score per game", series: init.games.map((g) => ({ key: `score_${g}`, label: title(g), color: Games.color(g) })), refs,
        explain: "Average return of the last 100 training episodes in each game. Dashed lines in the same color: random play (lower) and perfect play (upper)." },
      { title: "Losses", series: [{ key: "policy_loss", label: "policy", color: "#a78bfa" }, { key: "value_loss", label: "value", color: "#f472b6" }],
        explain: "Policy loss is the clipped PPO objective and hovers near 0, which is normal. Value loss is how wrong the critic's guesses are." },
      { title: "Exploration (entropy)", series: [{ key: "entropy", label: "entropy", color: "#22d3ee" }], yMin: 0,
        explain: "How spread out the move probabilities are. It should fall slowly as the agent gets confident. Dropping to ~0 early means it stopped exploring." },
      { title: "Policy change per update", series: [{ key: "approx_kl", label: "approx KL", color: "#fbbf24" }, { key: "clip_frac", label: "clip fraction", color: "#fb7185" }], yMin: 0,
        explain: "How far the policy moved in one update, and the share of moves where PPO's clip kicked in. Both should stay small (KL below about 0.02)." },
      { title: "Critic accuracy", series: [{ key: "explained_var", label: "explained variance", color: "#34d399" }], yMax: 1,
        explain: "How well the value head predicts real returns: 1 = perfect, 0 = no better than guessing the average." },
      { title: "Plasticity: dormant units", series: [
          { key: "dormant_conv1", label: "conv1", color: "#a78bfa" }, { key: "dormant_conv2", label: "conv2", color: "#818cf8" },
          { key: "dormant_fc", label: "hidden", color: "#22d3ee" }], yMin: 0, format: pct,
        explain: "Share of units that are almost never active (Sokar et al., 2023). Rising dormancy is one sign of plasticity loss, your research topic." },
      { title: "Plasticity: feature rank", series: [{ key: "eff_rank", label: "effective rank", color: "#f472b6" }], yMin: 0, format: (v) => v.toFixed(0),
        explain: "How many independent features the 128 hidden neurons really use (Kumar et al., 2021). Lower = less capacity left for new games." },
      { title: "Learning rate", series: [{ key: "lr", label: "learning rate", color: "#94a3c4" }], yMin: 0,
        explain: "Shrinks linearly to 0 by the end of training, so late updates are gentle." },
    ];
    S.charts = defs.map((d) => {
      const canvas = el("canvas", { role: "img", "aria-label": `${d.title} chart` });
      const legend = el("div", { class: "legend" }, d.series.map((s) =>
        el("span", {}, [el("i", { style: `background:${s.color}` }), s.label])));
      if (d.refs) legend.append(el("span", {}, [el("i", { class: "dash", style: "color:#94a3c4" }), "random / perfect"]));
      box.append(el("div", { class: "panel chart-card" }, [
        el("div", { class: "panel-head" }, [el("h2", { text: d.title })]), legend,
        el("p", { class: "explain", text: d.explain }), canvas,
      ]));
      return new Charts.LineChart(canvas, d);
    });
  }

  function buildConfig(init) {
    const form = $("config-form");
    form.innerHTML = "";
    for (const [key, [lo, hi]] of Object.entries(init.limits)) {
      const isInt = Number.isInteger(lo) && Number.isInteger(hi) && Number.isInteger(init.config[key]);
      form.append(el("label", {}, [
        key,
        el("input", { type: "number", name: key, value: init.config[key], min: lo, max: hi, step: isInt ? "1" : "any", required: "" }),
        el("small", { text: CONFIG_HELP[key] || "" }),
      ]));
    }
    form.append(el("button", { type: "submit", class: "btn primary full", text: "Apply and rebuild network" }));
  }

  // ---------- live updates ----------
  function setStatus(status) {
    S.status = status;
    const pill = $("status-pill");
    const labels = { idle: "Ready", running: "Training", paused: "Paused", finished: "Finished", error: "Error", offline: "Disconnected" };
    pill.textContent = labels[status] || status;
    pill.className = `status-pill ${status}`;
    $("btn-start").disabled = !(status === "idle" || status === "paused");
    $("btn-start").textContent = status === "paused" ? "▶ Resume" : "▶ Start";
    $("btn-pause").disabled = status !== "running";
    $("btn-save").disabled = status === "offline";
    $("btn-reset").disabled = status === "offline";
  }

  function updateLive(msg) {
    const init = S.init;
    if (!init) return;
    const cfg = init.config;
    $("kpi-update").textContent = fmtInt(msg.update);
    $("kpi-update-sub").textContent = `of ${fmtInt(msg.num_updates)}`;
    $("kpi-steps").textContent = Charts.fmtSteps(msg.env_steps);
    $("kpi-sps").textContent = `${fmtInt(msg.sps)} moves/s`;
    $("kpi-time").textContent = fmtTime(msg.elapsed);
    $("kpi-lr").textContent = `lr ${msg.lr.toExponential(2)}`;
    $("overall-progress").style.width = `${(100 * msg.update) / msg.num_updates}%`;
    $("loop-update").textContent = `update ${msg.update} / ${msg.num_updates}`;

    // which PPO step is running
    if (msg.update !== S.lastUpdate) {
      S.lastUpdate = msg.update;
      const rep = document.querySelector('.stage[data-phase="repeat"]');
      rep.classList.add("active");
      setTimeout(() => rep.classList.remove("active"), 350);
    }
    for (const st of document.querySelectorAll(".stage")) {
      if (st.dataset.phase !== "repeat") st.classList.toggle("active", st.dataset.phase === msg.phase && msg.status !== "idle");
    }
    const p = msg.phase_progress;
    const batch = init.num_envs * cfg.rollout_steps;
    $("bar-collect").style.width = msg.phase === "collect" ? `${p * 100}%` : msg.phase === "idle" ? "0%" : "100%";
    $("bar-update").style.width = msg.phase === "update" ? `${p * 100}%` : "0%";
    $("stage-collect-text").textContent = msg.phase === "collect"
      ? `Step ${Math.round(p * cfg.rollout_steps)} of ${cfg.rollout_steps} in ${init.num_envs} games → ${fmtInt(batch)} moves per update.`
      : `Play ${init.num_envs} games for ${cfg.rollout_steps} moves each (${fmtInt(batch)} moves).`;
    $("stage-adv-text").textContent = `Advantage = how much better a move turned out than expected (γ = ${cfg.gamma}, λ = ${cfg.lam}).`;
    $("stage-update-text").textContent = msg.phase === "update"
      ? `Epoch ${Math.min(cfg.epochs, Math.ceil(p * cfg.epochs) || 1)} of ${cfg.epochs} · ${cfg.minibatches} minibatches of ${fmtInt(batch / cfg.minibatches)} · clip ε = ${cfg.clip}`
      : `${cfg.epochs} passes × ${cfg.minibatches} minibatches over the batch, clip ε = ${cfg.clip}.`;

    if (document.activeElement !== $("train-delay")) {
      $("train-delay").value = msg.delay_ms;
      $("train-delay-val").textContent = msg.delay_ms ? `${msg.delay_ms} ms/move` : "max speed";
    }
    if (document.activeElement !== $("demo-fps")) {
      $("demo-fps").value = msg.demo_fps;
      $("demo-fps-val").textContent = `${Math.round(msg.demo_fps)} fps`;
    }
    S.demoFps = msg.demo_fps;
    for (const b of document.querySelectorAll("[data-mode]")) b.classList.toggle("on", b.dataset.mode === msg.demo_mode);
  }

  function updateFromHistory() {
    for (const c of S.charts) c.setRows(S.history);
    const last = S.history[S.history.length - 1];
    if (!last) return;
    for (const g of S.init.games) {
      const v = last[`score_${g}`], b = S.init.baselines[g];
      $(`kpi-score-${g}`).textContent = fmtSigned(v);
      if (v !== null && v !== undefined && b.optimal !== null) {
        const prog = Math.max(0, Math.min(1, (v - b.random) / (b.optimal - b.random)));
        $(`kpi-prog-${g}`).style.width = `${prog * 100}%`;
      }
    }
    $("kpi-entropy").textContent = last.entropy.toFixed(3);
    $("kpi-dormant").textContent = `${(last.dormant_fc * 100).toFixed(1)}%`;
  }

  function onFrame(g, f) {
    S.frames[g] = f;
    const r = S.renderers[g];
    if (r) r.push(f, 1000 / S.demoFps);
    if (g === S.focus) brainView.setFrame(f, Games.color(g));

    $(`g-${g}-ep`).textContent = `episode ${f.episode} · step ${f.step}`;
    $(`g-${g}-ret`).textContent = fmtSigned(f.ep_return);
    $(`g-${g}-avg`).textContent = fmtSigned(f.recent_return);
    $(`g-${g}-win`).textContent = f.success_rate === null ? "—" : `${Math.round(f.success_rate * 100)}%`;

    const list = $(`g-${g}-probs`);
    if (!S.validBuilt[g]) {
      list.innerHTML = "";
      for (const a of f.valid) {
        list.append(el("li", { "data-a": a }, [
          el("span", { text: S.init.actions[a] }), el("div", { class: "pbar" }, [el("div")]), el("span", { class: "pv", text: "—" }),
        ]));
      }
      S.validBuilt[g] = true;
    }
    if (f.probs) {
      for (const li of list.children) {
        const a = Number(li.dataset.a), p = f.probs[a];
        li.classList.toggle("chosen", a === f.action);
        li.querySelector(".pbar div").style.width = `${p * 100}%`;
        li.querySelector(".pv").textContent = `${(p * 100).toFixed(1)}%`;
      }
    }
    if (f.value !== null) {
      $(`g-${g}-value`).textContent = fmtSigned(f.value, 3);
      $(`g-${g}-needle`).style.left = `${50 + 50 * Math.max(-1, Math.min(1, f.value))}%`;
    }
  }

  function updateArena(arena) {
    if (!arena) return;
    const now = performance.now();
    arena.forEach((entry, i) => {
      const cell = S.arena[i];
      if (!cell) return;
      Games.drawMini(cell.canvas, entry, now);
      cell.info.textContent = entry.prob === null ? "—" : `${S.init.actions[entry.action]} ${(entry.prob * 100).toFixed(0)}%`;
    });
  }

  function updateBrainPanels(brain) {
    Brain.drawFilters($("filters-canvas"), brain);
    Brain.drawHistograms($("hist-canvas"), brain, S.init.layers);
    const body = $("health-body");
    body.innerHTML = "";
    const grads = brain.grad_norm || {}, changes = brain.weight_change || {};
    const gmax = Math.max(...Object.values(grads), 1e-9);
    const cmax = Math.max(...Object.values(changes), 1e-9);
    for (const name of S.init.layers) {
      const g = grads[name], c = changes[name], n = brain.hist[name].norm;
      const bar = (v, max, text) => (v === undefined || v === null)
        ? "—" : el("div", { class: "hbar" }, [el("i", { style: `width:${Math.max(2, (v / max) * 70)}px` }), text]);
      body.append(el("tr", {}, [
        el("td", { text: name }),
        el("td", {}, [bar(g, gmax, g?.toFixed(3))]),
        el("td", {}, [bar(c, cmax, c === undefined ? "" : `${(c * 100).toFixed(2)}%`)]),
        el("td", { text: n.toFixed(2) }),
      ]));
    }
  }

  function addEvents(events) {
    const list = $("events");
    for (const e of events) {
      list.prepend(el("li", { class: e.kind }, [el("time", { text: e.time }), e.text]));
    }
    while (list.children.length > 200) list.lastChild.remove();
  }

  async function refreshCheckpoints() {
    const list = $("checkpoints");
    try {
      const items = await (await fetch("/api/checkpoints")).json();
      for (const fn of hooks.checkpoints) fn(items);
      list.innerHTML = "";
      if (!items.length) list.append(el("li", { class: "meta", text: "No checkpoints yet. Press Save." }));
      for (const item of items) {
        const btn = el("button", { type: "button", class: "btn small", text: "Load" });
        btn.addEventListener("click", async () => {
          const res = await control({ cmd: "load", name: item.name });
          if (res.ok) toast(`Loaded ${item.name}`);
        });
        list.append(el("li", {}, [
          el("div", {}, [el("div", { class: "name", text: item.name }), el("div", { class: "meta", text: `${item.modified} · ${item.size_kb} KB` })]),
          btn,
        ]));
      }
    } catch {
      list.innerHTML = "";
      list.append(el("li", { class: "meta", text: "Couldn't load the list." }));
    }
  }

  // ---------- message handling ----------
  function handle(msg) {
    if (msg.init) {
      setup(msg.init);
      for (const fn of hooks.init) fn(msg.init);
    }
    if (!S.init) return;
    setStatus(msg.status);
    if (msg.jobs) for (const fn of hooks.jobs) fn(msg.jobs);
    if (msg.history) { S.history.push(...msg.history); updateFromHistory(); }
    if (msg.events) addEvents(msg.events);
    if (msg.brain) { S.brain = msg.brain; brainView.setBrain(msg.brain); updateBrainPanels(msg.brain); }
    if (msg.demo) for (const [g, f] of Object.entries(msg.demo)) onFrame(g, f);
    updateLive(msg);
    updateArena(msg.arena);
  }

  function connect() {
    const source = new EventSource("/api/stream");
    source.onmessage = (e) => handle(JSON.parse(e.data));
    source.onerror = () => setStatus("offline"); // the browser reconnects on its own
  }

  // ---------- controls ----------
  $("btn-start").addEventListener("click", () => control({ cmd: "start" }));
  $("btn-pause").addEventListener("click", () => control({ cmd: "pause" }));
  $("btn-reset").addEventListener("click", () => {
    if (confirm("Throw away the current network and build a fresh one? Save first if you want to keep it.")) control({ cmd: "reset" });
  });
  $("btn-save").addEventListener("click", async () => {
    const res = await control({ cmd: "save" });
    if (res.ok) { toast(`Saved ${res.file}`); refreshCheckpoints(); }
  });
  $("btn-refresh").addEventListener("click", refreshCheckpoints);
  $("btn-explain").addEventListener("click", (e) => {
    const on = document.body.classList.toggle("explain-on");
    e.currentTarget.setAttribute("aria-pressed", String(on));
    for (const c of S.charts) c.dirty = true;
  });

  $("train-delay").addEventListener("input", (e) => {
    const v = Number(e.target.value);
    $("train-delay-val").textContent = v ? `${v} ms/move` : "max speed";
  });
  $("train-delay").addEventListener("change", (e) => control({ cmd: "speed", delay_ms: Number(e.target.value) }));
  $("demo-fps").addEventListener("input", (e) => { $("demo-fps-val").textContent = `${e.target.value} fps`; });
  $("demo-fps").addEventListener("change", (e) => control({ cmd: "demo", fps: Number(e.target.value) }));
  for (const b of document.querySelectorAll("[data-mode]")) {
    b.addEventListener("click", () => control({ cmd: "demo", mode: b.dataset.mode }));
  }

  $("config-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (!confirm("Build a fresh network with these settings? The current one is thrown away.")) return;
    const config = {};
    for (const input of e.target.querySelectorAll("input")) config[input.name] = Number(input.value);
    control({ cmd: "reset", config }).then((r) => { if (r.ok) toast("New network built with your settings."); });
  });

  // Space = start / pause (when not typing in a field)
  document.addEventListener("keydown", (e) => {
    if (S.tab !== "train" || e.code !== "Space" ||
        ["INPUT", "TEXTAREA", "BUTTON", "SELECT", "SUMMARY"].includes(document.activeElement.tagName)) return;
    e.preventDefault();
    control({ cmd: S.status === "running" ? "pause" : "start" });
  });

  window.addEventListener("resize", () => {
    for (const c of S.charts) c.dirty = true;
    if (S.brain) updateBrainPanels(S.brain);
  });

  // ---------- animation loop ----------
  function tick(now) {
    if (S.tab === "train") {
      for (const r of Object.values(S.renderers)) r.draw(now);
      brainView.draw(now);
      for (const c of S.charts) c.draw();
    }
    for (const fn of hooks.tick) fn(now, S.tab);
    requestAnimationFrame(tick);
  }

  // Tabs: click, arrow keys, and the address bar (#test, #experiments…)
  const tabButtons = [...document.querySelectorAll("[data-tab]")];
  tabButtons.forEach((b, i) => {
    b.addEventListener("click", () => showTab(b.dataset.tab));
    b.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const next = tabButtons[(i + (e.key === "ArrowRight" ? 1 : tabButtons.length - 1)) % tabButtons.length];
      next.focus();
      showTab(next.dataset.tab);
    });
  });
  $("job-pill").addEventListener("click", () => showTab("experiments"));

  window.App = {
    $, el, control, toast, fmtSigned, fmtTime, fmtInt, title, hooks, showTab, refreshCheckpoints,
    getInit: () => S.init,
  };

  setStatus("offline");
  connect();
  showTab(location.hash.slice(1) || "train");
  requestAnimationFrame(tick);
})();
