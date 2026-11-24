"use strict";

// Live neural-network view: input -> conv1 -> conv2 -> hidden -> outputs.
const Brain = (() => {
  const CHANNEL_COLORS = ["#22d3ee", "#94a3b8", "#facc15", "#f87171", "#fb923c", "#a78bfa", "#34d399", "#f472b6"];
  const HEAT = [[0, [10, 14, 34]], [0.25, [62, 32, 128]], [0.5, [168, 52, 140]], [0.75, [247, 114, 82]], [1, [255, 240, 170]]];

  function heat(v) {
    v = Math.max(0, Math.min(1, v));
    for (let i = 1; i < HEAT.length; i++) {
      if (v <= HEAT[i][0]) {
        const [p0, c0] = HEAT[i - 1], [p1, c1] = HEAT[i];
        const t = (v - p0) / (p1 - p0);
        return `rgb(${c0.map((x, k) => Math.round(x + (c1[k] - x) * t)).join(",")})`;
      }
    }
    return "rgb(255,240,170)";
  }

  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  const maxOf = (arr) => arr.reduce((m, v) => (v > m ? v : m), 0) || 1;

  class BrainView {
    constructor(canvas, tooltip, actions, channels) {
      this.canvas = canvas;
      this.tooltip = tooltip;
      this.actions = actions;
      this.channels = channels;
      this.frame = null;
      this.brain = null;
      this.gameColor = "#a78bfa";
      this.base = document.createElement("canvas");
      this.dirty = true;
      this.size = "";
      this.hits = [];
      this.links = [];
      this.mouse = null;
      this.hover = null;
      canvas.addEventListener("mousemove", (e) => {
        const r = canvas.getBoundingClientRect();
        this.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
      });
      canvas.addEventListener("mouseleave", () => { this.mouse = null; this.tooltip.hidden = true; });
    }

    setFrame(frame, color) {
      if (!frame || !frame.acts) return;
      this.frame = frame;
      this.gameColor = color;
      this.dirty = true;
    }

    setBrain(brain) { this.brain = brain; this.dirty = true; }

    // ---------- layout ----------
    _layout(w, h) {
      const top = 46, bottom = h - 16, H = bottom - top, gap = 4, label = 13;
      let tIn = (H - 3 * gap - 4 * label) / 4;
      let t1 = (H - 3 * gap) / 4;
      let t2 = (H - 7 * gap) / 8;
      let sp = H / 16;
      const outW = Math.min(230, w * 0.2);
      const widths = () => [2 * tIn + gap, 4 * t1 + 3 * gap, 4 * t2 + 3 * gap, 8 * sp];
      const free = w - 32 - outW - 4 * 50;
      const total = widths().reduce((a, b) => a + b, 0);
      if (total > free) {
        const k = free / total;
        tIn *= k; t1 *= k; t2 *= k; sp *= k;
      }
      const ws = widths();
      const gapX = (w - 32 - outW - ws.reduce((a, b) => a + b, 0)) / 4;
      const xs = [];
      let x = 16;
      for (const cw of ws) { xs.push(x); x += cw + gapX; }
      xs.push(x);
      const colH = [4 * tIn + 3 * gap + 4 * label, 4 * t1 + 3 * gap, 8 * t2 + 7 * gap, 16 * sp];
      const ys = colH.map((ch) => top + (H - ch) / 2);
      return { top, bottom, H, gap, label, tIn, t1, t2, sp, xs, ws: [...ws, outW], ys, colH, outW };
    }

    // ---------- static picture, redrawn only when data changes ----------
    _rebuild(w, h) {
      const dpr = window.devicePixelRatio || 1;
      this.base.width = Math.round(w * dpr);
      this.base.height = Math.round(h * dpr);
      const ctx = this.base.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const L = (this.L = this._layout(w, h));
      this.hits = [];
      this.links = [];

      const titles = ["Input · 8×10×10", "Conv 1 · 16 maps", "Conv 2 · 32 maps", "Hidden · 128 neurons", "Output"];
      ctx.font = "600 12px Inter, Segoe UI, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      titles.forEach((t, i) => { ctx.fillStyle = "#c7d2fe"; ctx.fillText(t, L.xs[i], 14); });
      ctx.font = "11px Inter, Segoe UI, sans-serif";
      ctx.fillStyle = "#6b7796";
      const subs = ["what it sees", "local patterns", "bigger patterns", "combined features", "decision"];
      subs.forEach((t, i) => ctx.fillText(t, L.xs[i], 29));

      const f = this.frame;
      if (!f) {
        ctx.fillStyle = "#6b7796";
        ctx.textAlign = "center";
        ctx.font = "13px Inter, sans-serif";
        ctx.fillText("Waiting for the live player…", w / 2, h / 2);
        return;
      }

      // input layers
      const obs = f.obs;
      for (let ch = 0; ch < 8; ch++) {
        const col = ch % 2, row = Math.floor(ch / 2);
        const x = L.xs[0] + col * (L.tIn + L.gap), y = L.ys[0] + row * (L.tIn + L.gap + L.label);
        const cell = L.tIn / 10;
        ctx.fillStyle = "#0a1022";
        ctx.fillRect(x, y, L.tIn, L.tIn);
        let any = false;
        for (let i = 0; i < 100; i++) {
          if (obs[ch * 100 + i]) {
            any = true;
            ctx.fillStyle = CHANNEL_COLORS[ch];
            ctx.fillRect(x + (i % 10) * cell, y + Math.floor(i / 10) * cell, cell, cell);
          }
        }
        ctx.strokeStyle = any ? hexA(CHANNEL_COLORS[ch], 0.7) : "rgba(148,163,196,0.18)";
        ctx.strokeRect(x + 0.5, y + 0.5, L.tIn - 1, L.tIn - 1);
        ctx.fillStyle = any ? CHANNEL_COLORS[ch] : "#56607c";
        ctx.font = "10px Inter, sans-serif";
        ctx.fillText(this.channels[ch], x, y + L.tIn + 1);
        this.hits.push({ x, y, w: L.tIn, h: L.tIn, html:
          `<b>Input: ${this.channels[ch]}</b><br>${any ? "Lit cells show where this object is." : "Empty: this game has no " + this.channels[ch] + " objects."}` });
      }

      // conv maps
      const drawMaps = (data, count, side, cols, tile, x0, y0, layerName, maxv) => {
        const cell = tile / side;
        for (let m = 0; m < count; m++) {
          const x = x0 + (m % cols) * (tile + L.gap), y = y0 + Math.floor(m / cols) * (tile + L.gap);
          let mmax = 0;
          for (let i = 0; i < side * side; i++) {
            const v = data[m * side * side + i];
            if (v > mmax) mmax = v;
            ctx.fillStyle = heat(Math.sqrt(v / maxv)); // sqrt keeps weak activations visible
            ctx.fillRect(x + (i % side) * cell, y + Math.floor(i / side) * cell, cell + 0.3, cell + 0.3);
          }
          this.hits.push({ x, y, w: tile, h: tile, html:
            `<b>${layerName} · map ${m}</b><br>Strongest response: ${mmax.toFixed(2)}<br>` +
            (mmax === 0 ? "Silent for this board (its pattern isn't present, or it's dormant)." : "Bright spots = where this filter's pattern appears.") });
        }
      };
      drawMaps(f.acts.conv1, 16, 10, 4, L.t1, L.xs[1], L.ys[1], "Conv 1", maxOf(f.acts.conv1));
      drawMaps(f.acts.conv2, 32, 5, 4, L.t2, L.xs[2], L.ys[2], "Conv 2", maxOf(f.acts.conv2));

      // hidden neurons
      const fc = f.acts.fc, fcMax = maxOf(fc);
      this.neurons = [];
      for (let j = 0; j < 128; j++) {
        const cx = L.xs[3] + (j % 8 + 0.5) * L.sp, cy = L.ys[3] + (Math.floor(j / 8) + 0.5) * L.sp;
        const v = fc[j] / fcMax;
        const r = L.sp * 0.36;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = fc[j] > 0 ? heat(0.15 + 0.85 * v) : "#141b30";
        ctx.fill();
        ctx.strokeStyle = fc[j] > 0 ? "rgba(255,255,255,0.15)" : "rgba(148,163,196,0.12)";
        ctx.stroke();
        this.neurons.push({ cx, cy, r });
      }

      // outputs
      const ox = L.xs[4], ow = L.outW;
      const rowH = Math.min(40, L.H / 8.2);
      const oy0 = L.top + (L.H - rowH * 7.6) / 2;
      this.outputs = [];
      const probs = f.probs || [];
      const valid = new Set(f.valid);
      this.actions.forEach((name, a) => {
        const y = oy0 + a * rowH;
        const isValid = valid.has(a), chosen = a === f.action, p = probs[a] || 0;
        const nx = ox + 8, ny = y + rowH / 2;
        ctx.beginPath();
        ctx.arc(nx, ny, 7, 0, Math.PI * 2);
        ctx.fillStyle = !isValid ? "#1a2138" : chosen ? this.gameColor : hexA(this.gameColor, 0.25 + 0.75 * p);
        ctx.fill();
        if (chosen) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1; }
        ctx.font = `${chosen ? 700 : 500} 12px Inter, sans-serif`;
        ctx.textBaseline = "middle";
        ctx.fillStyle = isValid ? (chosen ? "#fff" : "#c7d2fe") : "#4a5370";
        ctx.fillText(name.toUpperCase(), nx + 14, ny - 6);
        const bx = nx + 14, bw = ow - 70;
        ctx.fillStyle = "rgba(148,163,196,0.12)";
        ctx.fillRect(bx, ny + 3, bw, 5);
        if (isValid) {
          ctx.fillStyle = this.gameColor;
          ctx.fillRect(bx, ny + 3, bw * p, 5);
        }
        ctx.font = "11px Consolas, monospace";
        ctx.textAlign = "right";
        ctx.fillStyle = isValid ? "#e6ecff" : "#4a5370";
        ctx.fillText(isValid ? (p * 100).toFixed(1) + "%" : "unused", ox + ow, ny - 2);
        ctx.textAlign = "left";
        this.outputs.push({ x: nx, y: ny });
        this.hits.push({ x: ox, y, w: ow, h: rowH, html: isValid
          ? `<b>${name.toUpperCase()}</b> · ${(p * 100).toFixed(1)}%${chosen ? " · <b>chosen</b>" : ""}<br>Probability from the policy head (softmax of 128 weighted hidden neurons).`
          : `<b>${name.toUpperCase()}</b><br>Not used in this game, so it's masked out (probability 0).` });
      });

      // value
      const vy = oy0 + 6.5 * rowH;
      const value = f.value ?? 0;
      ctx.textBaseline = "middle";
      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = "#c7d2fe";
      ctx.fillText("VALUE", ox + 22, vy);
      ctx.font = "12px Consolas, monospace";
      ctx.textAlign = "right";
      ctx.fillStyle = value >= 0 ? "#34d399" : "#f87171";
      ctx.fillText((value >= 0 ? "+" : "") + value.toFixed(3), ox + ow, vy);
      ctx.textAlign = "left";
      const gx = ox + 22, gw = ow - 22, gy = vy + 11;
      ctx.fillStyle = "rgba(148,163,196,0.12)";
      ctx.fillRect(gx, gy, gw, 6);
      const vClamped = Math.max(-1, Math.min(1, value));
      ctx.fillStyle = value >= 0 ? "#34d399" : "#f87171";
      const mid = gx + gw / 2;
      ctx.fillRect(Math.min(mid, mid + (gw / 2) * vClamped), gy, Math.abs((gw / 2) * vClamped), 6);
      ctx.beginPath();
      ctx.arc(ox + 8, vy + 3, 7, 0, Math.PI * 2);
      ctx.fillStyle = "#818cf8";
      ctx.fill();
      this.valueNode = { x: ox + 8, y: vy + 3 };
      this.hits.push({ x: ox, y: vy - 12, w: ow, h: 30, html:
        `<b>Value V(s) = ${value.toFixed(3)}</b><br>The critic's guess of future reward from this board. PPO compares real outcomes to this guess to score moves.` });

      // strongest hidden -> output connections ("why it chose this move")
      const W = this.brain && this.brain.policy_w;
      if (W && f.action !== null && f.action !== undefined) {
        for (const a of f.valid) {
          const contrib = [];
          for (let j = 0; j < 128; j++) contrib.push({ j, c: fc[j] * W[a * 128 + j] });
          contrib.sort((p, q) => Math.abs(q.c) - Math.abs(p.c));
          const top = contrib.slice(0, a === f.action ? 26 : 6).filter((x) => x.c !== 0);
          const cmax = Math.abs(top[0]?.c) || 1;
          for (const { j, c } of top) {
            this.links.push({ from: this.neurons[j], to: this.outputs[a], c, strength: Math.abs(c) / cmax,
              main: a === f.action, phase: Math.random() });
          }
        }
      }
      // hover info for neurons
      this.neurons.forEach((n, j) => {
        const wChosen = W && f.action !== null && f.action !== undefined ? W[f.action * 128 + j] : null;
        this.hits.push({ x: n.cx - n.r, y: n.cy - n.r, w: n.r * 2, h: n.r * 2, neuron: j, html:
          `<b>Hidden neuron #${j}</b><br>Activation: ${fc[j].toFixed(3)}${fc[j] === 0 ? " (silent)" : ""}` +
          (wChosen !== null ? `<br>Weight → ${this.actions[f.action].toUpperCase()}: ${wChosen >= 0 ? "+" : ""}${wChosen.toFixed(3)}` +
            `<br>Push for chosen move: ${(fc[j] * wChosen >= 0 ? "+" : "")}${(fc[j] * wChosen).toFixed(3)}` : "") });
      });
    }

    // ---------- animated layer on top ----------
    draw(now) {
      const { ctx, w, h } = Charts.setup(this.canvas);
      const size = `${w}x${h}x${window.devicePixelRatio}`;
      if (this.dirty || size !== this.size) {
        this._rebuild(w, h);
        this.dirty = false;
        this.size = size;
      }
      ctx.clearRect(0, 0, w, h);
      const L = this.L;

      // flowing ribbons between columns
      if (this.frame) {
        for (let i = 0; i < 4; i++) {
          const x1 = L.xs[i] + L.ws[i] + 6, x2 = L.xs[i + 1] - 6;
          const yA = L.ys[i], hA = L.colH[i];
          const yB = i < 3 ? L.ys[i + 1] : L.top + L.H * 0.12, hB = i < 3 ? L.colH[i + 1] : L.H * 0.76;
          for (let k = 0; k < 7; k++) {
            const t = (k + 0.5) / 7;
            const ya = yA + hA * t, yb = yB + hB * t;
            ctx.beginPath();
            ctx.moveTo(x1, ya);
            ctx.bezierCurveTo((x1 + x2) / 2, ya, (x1 + x2) / 2, yb, x2, yb);
            ctx.strokeStyle = "rgba(124,92,255,0.10)";
            ctx.lineWidth = 6;
            ctx.stroke();
            ctx.setLineDash([3, 14]);
            ctx.lineDashOffset = -now * 0.04 - k * 5;
            ctx.strokeStyle = "rgba(167,139,250,0.55)";
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }

      // connections that explain the decision, drawn under the static picture's nodes
      const hoverNeuron = this.hover && this.hover.neuron !== undefined ? this.hover.neuron : null;
      for (const link of this.links) {
        if (hoverNeuron !== null) continue;
        this._link(ctx, link, now);
      }
      if (hoverNeuron !== null && this.brain && this.frame) {
        // show every outgoing weight of the hovered neuron
        const W = this.brain.policy_w, fc = this.frame.acts.fc;
        for (const a of this.frame.valid) {
          const c = fc[hoverNeuron] * W[a * 128 + hoverNeuron], wgt = W[a * 128 + hoverNeuron];
          this._link(ctx, { from: this.neurons[hoverNeuron], to: this.outputs[a], c: wgt,
            strength: Math.min(1, Math.abs(wgt) * 4), main: c !== 0, phase: a / 6 }, now);
        }
      }

      ctx.drawImage(this.base, 0, 0, w, h);

      // caption
      if (this.frame) {
        ctx.font = "11px Inter, sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "top";
        ctx.fillStyle = "#6b7796";
        ctx.fillText(`episode ${this.frame.episode} · step ${this.frame.step} · weights v${this.frame.weights_version}`, w - 16, 14);
      }

      // hover
      this.hover = null;
      if (this.mouse) {
        for (const hit of this.hits) {
          if (this.mouse.x >= hit.x && this.mouse.x <= hit.x + hit.w && this.mouse.y >= hit.y && this.mouse.y <= hit.y + hit.h) {
            this.hover = hit;
          }
        }
      }
      if (this.hover) {
        const hv = this.hover;
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        if (hv.neuron !== undefined) {
          const n = this.neurons[hv.neuron];
          ctx.beginPath(); ctx.arc(n.cx, n.cy, n.r + 3, 0, Math.PI * 2); ctx.stroke();
        } else {
          ctx.strokeRect(hv.x - 1.5, hv.y - 1.5, hv.w + 3, hv.h + 3);
        }
        this.tooltip.innerHTML = hv.html;
        this.tooltip.hidden = false;
        const tx = this.mouse.x + 16 + 260 > w ? this.mouse.x - 270 : this.mouse.x + 16;
        this.tooltip.style.left = `${tx}px`;
        this.tooltip.style.top = `${Math.min(this.mouse.y + 12, h - 90)}px`;
      } else {
        this.tooltip.hidden = true;
      }
    }

    _link(ctx, link, now) {
      const { from, to, c, strength, main } = link;
      if (!from || !to) return;
      const color = c >= 0 ? "96,165,250" : "248,113,113";
      const alpha = (main ? 0.25 : 0.08) + (main ? 0.6 : 0.15) * strength;
      const mx = (from.cx + to.x) / 2;
      ctx.beginPath();
      ctx.moveTo(from.cx, from.cy);
      ctx.bezierCurveTo(mx, from.cy, mx, to.y, to.x, to.y);
      ctx.strokeStyle = `rgba(${color},${alpha})`;
      ctx.lineWidth = 0.6 + (main ? 2.6 : 1) * strength;
      ctx.stroke();
      if (main) {
        // a pulse travelling toward the output = signal flowing forward
        const t = (now * 0.0006 + link.phase) % 1;
        const u = 1 - t;
        const x = u * u * u * from.cx + 3 * u * u * t * mx + 3 * u * t * t * mx + t * t * t * to.x;
        const y = u * u * u * from.cy + 3 * u * u * t * from.cy + 3 * u * t * t * to.y + t * t * t * to.y;
        ctx.fillStyle = `rgba(${color},${0.5 + 0.5 * strength})`;
        ctx.beginPath();
        ctx.arc(x, y, 1.5 + 2 * strength, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  /** First-layer filters for the agent, goal and item channels. */
  function drawFilters(canvas, brain) {
    const { ctx, w, h } = Charts.setup(canvas);
    ctx.clearRect(0, 0, w, h);
    if (!brain) return;
    const groups = brain.conv1_channels, data = brain.conv1; // 16 x 3 x 9
    let lim = 0;
    for (const v of data) lim = Math.max(lim, Math.abs(v));
    lim = lim || 1;
    const groupW = (w - 20) / groups.length;
    const tile = Math.min((groupW - 3 * 6) / 4, (h - 40) / 4 - 6);
    groups.forEach((name, g) => {
      const gx = 10 + g * groupW + (groupW - (4 * tile + 18)) / 2;
      ctx.fillStyle = CHANNEL_COLORS[{ agent: 0, goal: 2, item: 4 }[name]] || "#c7d2fe";
      ctx.font = "600 11px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(`${name} layer`, gx, 4);
      for (let f = 0; f < 16; f++) {
        const x = gx + (f % 4) * (tile + 6), y = 24 + Math.floor(f / 4) * (tile + 6);
        const cell = tile / 3;
        for (let i = 0; i < 9; i++) {
          const v = data[f * 27 + g * 9 + i] / lim;
          ctx.fillStyle = v >= 0 ? `rgba(248,113,113,${Math.abs(v)})` : `rgba(96,165,250,${Math.abs(v)})`;
          ctx.fillRect(x + (i % 3) * cell, y + Math.floor(i / 3) * cell, cell, cell);
        }
        ctx.strokeStyle = "rgba(148,163,196,0.2)";
        ctx.strokeRect(x + 0.5, y + 0.5, tile - 1, tile - 1);
      }
    });
  }

  function drawHistograms(canvas, brain, layers) {
    const { ctx, w, h } = Charts.setup(canvas);
    ctx.clearRect(0, 0, w, h);
    if (!brain) return;
    const colors = ["#a78bfa", "#818cf8", "#22d3ee", "#34d399", "#fbbf24"];
    const rows = Math.ceil(layers.length / 2), cellW = (w - 16) / 2, cellH = h / rows;
    layers.forEach((name, i) => {
      const x = 4 + (i % 2) * (cellW + 8), y = Math.floor(i / 2) * cellH + 2;
      Charts.histogram(ctx, x, y, cellW, cellH - 8, brain.hist[name], colors[i % colors.length], name);
    });
  }

  return { BrainView, drawFilters, drawHistograms };
})();
