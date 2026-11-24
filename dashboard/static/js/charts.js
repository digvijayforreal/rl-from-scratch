"use strict";

// Small canvas chart helpers. No libraries, so everything works offline.
const Charts = (() => {
  const FONT = "11px Inter, Segoe UI, system-ui, sans-serif";

  /** Match the canvas pixel size to its CSS size (sharp on high-DPI screens). */
  function setup(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function fmtSteps(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
    if (n >= 1e3) return Math.round(n / 1e3) + "k";
    return String(Math.round(n));
  }

  /** toFixed without "-0.00". */
  function fixed(v, digits) {
    const s = v.toFixed(digits);
    return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
  }

  function fmtNum(v) {
    const a = Math.abs(v);
    if (a !== 0 && (a < 0.01 || a >= 1e4)) return v.toExponential(1);
    return fixed(v, a < 1 ? 3 : 2);
  }

  /** A round step (1, 2 or 5 × 10^n) close to `raw`, so axis labels are readable. */
  function niceStep(raw) {
    if (!(raw > 0)) return 1;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
  }

  class LineChart {
    /**
     * series: [{key, label, color}]   refs: [{value, label, color}]  (dashed lines)
     */
    constructor(canvas, { series, refs = [], markers = [], yMin, yMax, format = fmtNum, xKey = "step", emptyText }) {
      this.canvas = canvas;
      this.series = series;   // [{key, label, color, lo?, hi?}]  lo/hi = keys of a shaded band
      this.refs = refs;
      this.markers = markers; // [{x, label}] vertical dashed lines
      this.xKey = xKey;
      this.emptyText = emptyText || "Waiting for the first update…";
      this.yMin = yMin;
      this.yMax = yMax;
      this.format = format;
      this.rows = [];
      this.hoverX = null;
      this.dirty = true;
      canvas.addEventListener("mousemove", (e) => {
        this.hoverX = e.clientX - canvas.getBoundingClientRect().left;
        this.dirty = true;
      });
      canvas.addEventListener("mouseleave", () => { this.hoverX = null; this.dirty = true; hideTip(); });
    }

    setRows(rows) { this.rows = rows; this.dirty = true; }

    draw() {
      if (!this.dirty) return;
      this.dirty = false;
      const { ctx, w, h } = setup(this.canvas);
      ctx.clearRect(0, 0, w, h);
      const pad = { l: 50, r: 12, t: 10, b: 22 };
      const x0 = pad.l, x1 = w - pad.r, y0 = pad.t, y1 = h - pad.b;
      const rows = this.rows;

      // y range from data + reference lines
      const xk = this.xKey;
      let lo = Infinity, hi = -Infinity;
      for (const r of rows) for (const s of this.series) {
        for (const key of [s.key, s.lo, s.hi]) {
          const v = key ? r[key] : undefined;
          if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
        }
      }
      for (const ref of this.refs) if (Number.isFinite(ref.value)) { lo = Math.min(lo, ref.value); hi = Math.max(hi, ref.value); }
      if (this.yMin !== undefined) lo = this.yMin;
      if (this.yMax !== undefined) hi = this.yMax;
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = 0; hi = 1; }
      if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
      if (this.yMin === undefined || this.yMax === undefined) {
        const span = hi - lo;
        if (this.yMin === undefined) lo -= span * 0.06;
        if (this.yMax === undefined) hi += span * 0.06;
      }

      const xMin = rows.length ? rows[0][xk] : 0;
      const xMax = rows.length > 1 ? rows[rows.length - 1][xk] : xMin + 1;
      const X = (s) => x0 + ((s - xMin) / (xMax - xMin || 1)) * (x1 - x0);
      const Y = (v) => y1 - ((v - lo) / (hi - lo)) * (y1 - y0);

      // grid + axis labels
      ctx.font = FONT;
      ctx.lineWidth = 1;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      const yStep = niceStep((hi - lo) / 4);
      for (let v = Math.ceil(lo / yStep - 1e-9) * yStep; v <= hi + yStep * 1e-6; v += yStep) {
        const y = Math.round(Y(v)) + 0.5;
        ctx.strokeStyle = "rgba(148,163,196,0.10)";
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
        ctx.fillStyle = "#7f8bab";
        ctx.fillText(this.format(Math.abs(v) < yStep * 1e-6 ? 0 : v), x0 - 6, y);
      }
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      if (rows.length) {
        const xStep = niceStep((xMax - xMin) / 4);
        for (let s = Math.ceil(xMin / xStep - 1e-9) * xStep; s <= xMax + xStep * 1e-6; s += xStep) {
          ctx.fillText(fmtSteps(s), X(s), y1 + 6);
        }
      }

      // reference lines (dashed)
      for (const ref of this.refs) {
        if (!Number.isFinite(ref.value)) continue;
        const y = Y(ref.value);
        ctx.save();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = ref.color;
        ctx.globalAlpha = 0.55;
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
        ctx.restore();
      }

      if (!rows.length) {
        ctx.fillStyle = "#6b7796";
        ctx.textBaseline = "middle";
        ctx.fillText(this.emptyText, (x0 + x1) / 2, (y0 + y1) / 2);
        return;
      }

      // vertical markers (e.g. "game switch")
      for (const m of this.markers) {
        if (!Number.isFinite(m.x) || m.x < xMin || m.x > xMax) continue;
        const x = X(m.x);
        ctx.save();
        ctx.setLineDash([3, 4]);
        ctx.strokeStyle = "rgba(230,236,255,0.45)";
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
        ctx.restore();
        // label in a small pill, on whichever side of the line has room
        ctx.font = FONT;
        const lw = ctx.measureText(m.label).width + 10;
        const lx = x + 4 + lw > x1 ? x - 4 - lw : x + 4;
        ctx.fillStyle = "rgba(10,14,28,0.85)";
        ctx.beginPath(); ctx.roundRect(lx, y0 + 1, lw, 16, 6); ctx.fill();
        ctx.fillStyle = "#c7d2fe";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(m.label, lx + 5, y0 + 9);
      }

      // shaded bands (e.g. confidence intervals across seeds)
      for (const s of this.series) {
        if (!s.lo || !s.hi) continue;
        const pts = rows.filter((r) => Number.isFinite(r[s.lo]) && Number.isFinite(r[s.hi]));
        if (pts.length < 2) continue;
        ctx.beginPath();
        pts.forEach((r, i) => (i ? ctx.lineTo(X(r[xk]), Y(r[s.hi])) : ctx.moveTo(X(r[xk]), Y(r[s.hi]))));
        for (let i = pts.length - 1; i >= 0; i--) ctx.lineTo(X(pts[i][xk]), Y(pts[i][s.lo]));
        ctx.closePath();
        ctx.fillStyle = s.color + "2e";
        ctx.fill();
      }

      // series lines with a soft glow and area fill
      const hasBands = this.series.some((s) => s.lo);
      for (const s of this.series) {
        ctx.beginPath();
        let started = false, firstX = 0, lastX = 0;
        for (const r of rows) {
          const v = r[s.key];
          if (!Number.isFinite(v)) { continue; }
          const x = X(r[xk]), y = Y(v);
          if (!started) { ctx.moveTo(x, y); started = true; firstX = x; } else ctx.lineTo(x, y);
          lastX = x;
        }
        if (!started) continue;
        ctx.save();
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2;
        ctx.shadowColor = s.color;
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.restore();
        if (this.series.length <= 2 && !hasBands) {
          ctx.lineTo(lastX, y1); ctx.lineTo(firstX, y1); ctx.closePath();
          const grad = ctx.createLinearGradient(0, y0, 0, y1);
          grad.addColorStop(0, s.color + "33");
          grad.addColorStop(1, s.color + "00");
          ctx.fillStyle = grad;
          ctx.fill();
        }
      }

      // hover: crosshair and dots on the canvas, details in a floating HTML tooltip
      if (this.hoverX !== null && this.hoverX >= x0 && this.hoverX <= x1) {
        const target = xMin + ((this.hoverX - x0) / (x1 - x0)) * (xMax - xMin);
        let best = rows[0];
        for (const r of rows) if (Math.abs(r[xk] - target) < Math.abs(best[xk] - target)) best = r;
        const x = X(best[xk]);
        ctx.strokeStyle = "rgba(230,236,255,0.35)";
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
        const head = best.update !== undefined ? `update ${best.update} · ${fmtSteps(best[xk])} moves` : `${fmtSteps(best[xk])} moves`;
        const items = [];
        for (const s of this.series) {
          const v = best[s.key];
          if (!Number.isFinite(v)) continue;
          ctx.fillStyle = s.color;
          ctx.beginPath(); ctx.arc(x, Y(v), 3.5, 0, Math.PI * 2); ctx.fill();
          const lo = best[s.lo], hi = best[s.hi];
          const band = s.lo && Number.isFinite(lo) && Math.abs(hi - lo) > 1e-9 ? ` <span class="tip-ci">[${this.format(lo)}, ${this.format(hi)}]</span>` : "";
          items.push(`<div><i style="background:${s.color}"></i>${s.label}: <b>${this.format(v)}</b>${band}</div>`);
        }
        const rect = this.canvas.getBoundingClientRect();
        showTip(`<div class="tip-head">${head}</div>${items.join("")}`, rect.left + x, rect.top + y0);
      } else if (this._tipOwner) {
        hideTip();
      }
      this._tipOwner = this.hoverX !== null;
    }
  }

  // one shared tooltip element, kept inside the window
  let tipEl = null;
  function showTip(html, clientX, clientY) {
    if (!tipEl) {
      tipEl = document.createElement("div");
      tipEl.className = "chart-tip";
      tipEl.setAttribute("role", "tooltip");
      document.body.append(tipEl);
    }
    tipEl.innerHTML = html;
    tipEl.hidden = false;
    const { width, height } = tipEl.getBoundingClientRect();
    let left = clientX + 14;
    if (left + width > window.innerWidth - 8) left = clientX - width - 14;
    left = Math.max(8, left);
    const top = Math.min(Math.max(8, clientY), window.innerHeight - height - 8);
    tipEl.style.left = `${left}px`;
    tipEl.style.top = `${top}px`;
  }
  function hideTip() { if (tipEl) tipEl.hidden = true; }

  /** Histogram of weights for one layer, drawn inside a box. */
  function histogram(ctx, x, y, w, h, hist, color, label) {
    ctx.fillStyle = "#94a3c4";
    ctx.font = FONT;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(label, x, y);
    ctx.textAlign = "right";
    ctx.fillStyle = "#6b7796";
    ctx.fillText(`±${hist.lim.toFixed(2)} · ${hist.count.toLocaleString()} w`, x + w, y);
    const top = y + 16, height = h - 18;
    const counts = hist.counts;
    const max = Math.max(...counts, 1);
    const bw = w / counts.length;
    counts.forEach((c, i) => {
      const bh = Math.sqrt(c / max) * height; // sqrt so small bins stay visible
      const grad = ctx.createLinearGradient(0, top + height - bh, 0, top + height);
      grad.addColorStop(0, color);
      grad.addColorStop(1, color + "33");
      ctx.fillStyle = grad;
      ctx.fillRect(x + i * bw + 0.5, top + height - bh, bw - 1, bh);
    });
    ctx.strokeStyle = "rgba(230,236,255,0.25)";
    ctx.beginPath(); ctx.moveTo(x + w / 2, top); ctx.lineTo(x + w / 2, top + height); ctx.stroke();
  }

  /** Grouped bar chart of histogram counts. series: [{counts, color, label}] */
  function bars(canvas, { range, series }) {
    const { ctx, w, h } = setup(canvas);
    ctx.clearRect(0, 0, w, h);
    if (!series.length) return;
    const pad = { l: 34, r: 8, t: 8, b: 20 };
    const x0 = pad.l, x1 = w - pad.r, y0 = pad.t, y1 = h - pad.b;
    const bins = series[0].counts.length;
    const max = Math.max(1, ...series.flatMap((s) => s.counts));
    const bw = (x1 - x0) / bins;
    ctx.font = FONT;
    ctx.fillStyle = "#7f8bab";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 2; i++) {
      const v = (max * i) / 2, y = y1 - ((y1 - y0) * i) / 2;
      ctx.fillText(String(Math.round(v)), x0 - 5, y);
      ctx.strokeStyle = "rgba(148,163,196,0.10)";
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
    }
    series.forEach((s, k) => {
      const sw = (bw - 2) / series.length;
      s.counts.forEach((c, i) => {
        const bh = ((y1 - y0) * c) / max;
        ctx.fillStyle = s.color;
        ctx.fillRect(x0 + i * bw + 1 + k * sw, y1 - bh, Math.max(1, sw - 1), bh);
      });
    });
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#7f8bab";
    const [lo, hi] = range;
    for (const v of [lo, (lo + hi) / 2, hi]) ctx.fillText(v.toFixed(1), x0 + ((v - lo) / (hi - lo)) * (x1 - x0), y1 + 5);
  }

  return { setup, fmtSteps, fmtNum, fixed, niceStep, LineChart, histogram, bars };
})();
