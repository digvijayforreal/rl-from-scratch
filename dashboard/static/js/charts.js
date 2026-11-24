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

  function fmtNum(v) {
    const a = Math.abs(v);
    if (a !== 0 && (a < 0.01 || a >= 1e4)) return v.toExponential(1);
    return v.toFixed(a < 1 ? 3 : 2);
  }

  class LineChart {
    /**
     * series: [{key, label, color}]   refs: [{value, label, color}]  (dashed lines)
     */
    constructor(canvas, { series, refs = [], yMin, yMax, format = fmtNum }) {
      this.canvas = canvas;
      this.series = series;
      this.refs = refs;
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
      canvas.addEventListener("mouseleave", () => { this.hoverX = null; this.dirty = true; });
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
      let lo = Infinity, hi = -Infinity;
      for (const r of rows) for (const s of this.series) {
        const v = r[s.key];
        if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
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

      const xMin = rows.length ? rows[0].step : 0;
      const xMax = rows.length > 1 ? rows[rows.length - 1].step : xMin + 1;
      const X = (s) => x0 + ((s - xMin) / (xMax - xMin || 1)) * (x1 - x0);
      const Y = (v) => y1 - ((v - lo) / (hi - lo)) * (y1 - y0);

      // grid + axis labels
      ctx.font = FONT;
      ctx.lineWidth = 1;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i <= 4; i++) {
        const v = lo + ((hi - lo) * i) / 4;
        const y = Math.round(Y(v)) + 0.5;
        ctx.strokeStyle = "rgba(148,163,196,0.10)";
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
        ctx.fillStyle = "#7f8bab";
        ctx.fillText(this.format(v), x0 - 6, y);
      }
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      if (rows.length) {
        for (let i = 0; i <= 4; i++) {
          const s = xMin + ((xMax - xMin) * i) / 4;
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
        ctx.fillText("Waiting for the first update…", (x0 + x1) / 2, (y0 + y1) / 2);
        return;
      }

      // series lines with a soft glow and area fill
      for (const s of this.series) {
        ctx.beginPath();
        let started = false, firstX = 0, lastX = 0;
        for (const r of rows) {
          const v = r[s.key];
          if (!Number.isFinite(v)) { continue; }
          const x = X(r.step), y = Y(v);
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
        if (this.series.length <= 2) {
          ctx.lineTo(lastX, y1); ctx.lineTo(firstX, y1); ctx.closePath();
          const grad = ctx.createLinearGradient(0, y0, 0, y1);
          grad.addColorStop(0, s.color + "33");
          grad.addColorStop(1, s.color + "00");
          ctx.fillStyle = grad;
          ctx.fill();
        }
      }

      // hover crosshair + tooltip
      if (this.hoverX !== null && this.hoverX >= x0 && this.hoverX <= x1) {
        const target = xMin + ((this.hoverX - x0) / (x1 - x0)) * (xMax - xMin);
        let best = rows[0];
        for (const r of rows) if (Math.abs(r.step - target) < Math.abs(best.step - target)) best = r;
        const x = X(best.step);
        ctx.strokeStyle = "rgba(230,236,255,0.35)";
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
        const lines = [`update ${best.update} · ${fmtSteps(best.step)} moves`];
        for (const s of this.series) {
          const v = best[s.key];
          if (Number.isFinite(v)) {
            ctx.fillStyle = s.color;
            ctx.beginPath(); ctx.arc(x, Y(v), 3.5, 0, Math.PI * 2); ctx.fill();
            lines.push(`${s.label}: ${this.format(v)}`);
          }
        }
        ctx.font = FONT;
        const tw = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 16;
        const th = lines.length * 15 + 8;
        const tx = x + tw + 12 > x1 ? x - tw - 10 : x + 10;
        ctx.fillStyle = "rgba(10,14,28,0.92)";
        ctx.strokeStyle = "rgba(124,92,255,0.6)";
        ctx.beginPath(); ctx.roundRect(tx, y0 + 4, tw, th, 8); ctx.fill(); ctx.stroke();
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        lines.forEach((l, i) => {
          ctx.fillStyle = i === 0 ? "#94a3c4" : "#e6ecff";
          ctx.fillText(l, tx + 8, y0 + 9 + i * 15);
        });
      }
    }
  }

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

  return { setup, fmtSteps, fmtNum, LineChart, histogram };
})();
