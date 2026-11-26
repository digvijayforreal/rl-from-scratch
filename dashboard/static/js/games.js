"use strict";

// Game graphics. The Python game only sends object positions; everything you
// see (lighting, particles, smooth movement) is drawn here.
const Games = (() => {
  const GRID = 10;
  const NOOP = 0, UP = 1, RIGHT = 2, DOWN = 3, LEFT = 4;
  const COLORS = { catch: "#ffb020", reach: "#22d3ee", maze: "#f472b6", snake: "#4ade80" };
  const TITLES = { catch: "Catch", reach: "Reach", maze: "Maze", snake: "Snake" };
  const SNAKE_FOODS = 5; // food to eat to win (matches envs/snake.py)
  const DESCRIPTIONS = {
    catch: "A ball falls from the top. Slide the paddle under it. +1 catch, −1 miss.",
    reach: "Walk to the star. Start and goal are random every episode. +1 at the goal, −0.01 per step.",
    maze: "Walk to the star around the walls. The maze is different every episode. +1 at the goal, −0.01 per step.",
    snake: "Eat 5 apples to win (+0.2 each, −0.01 per other step). Hitting a wall or your own body ends the game with −1.",
  };
  const DIRS = { [UP]: [-1, 0], [RIGHT]: [0, 1], [DOWN]: [1, 0], [LEFT]: [0, -1] };

  const STARS = Array.from({ length: 80 }, () => ({
    x: Math.random(), y: Math.random() * 0.85, r: Math.random() * 1.3 + 0.3, tw: Math.random() * 6.28,
  }));

  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp01 = (t) => Math.max(0, Math.min(1, t));
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);

  // ---------- particles ----------
  class Particles {
    constructor() { this.items = []; }

    burst(x, y, color, n = 28, speed = 4, size = 3) {
      for (let i = 0; i < n; i++) {
        const a = Math.random() * Math.PI * 2, sp = speed * (0.3 + Math.random());
        this.items.push({ kind: "dot", x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - speed * 0.4,
          life: 1, decay: 0.015 + Math.random() * 0.02, color, size: size * (0.5 + Math.random()) });
      }
    }

    confetti(x, y, n = 40) {
      const palette = ["#fde047", "#f472b6", "#22d3ee", "#a78bfa", "#34d399", "#fb923c"];
      for (let i = 0; i < n; i++) {
        const a = Math.random() * Math.PI * 2, sp = 2 + Math.random() * 4;
        this.items.push({ kind: "rect", x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 3,
          life: 1, decay: 0.01 + Math.random() * 0.012, color: palette[i % palette.length],
          size: 3 + Math.random() * 4, rot: Math.random() * 6, vr: (Math.random() - 0.5) * 0.4 });
      }
    }

    text(x, y, str, color) {
      this.items.push({ kind: "text", x, y, vx: 0, vy: -0.9, life: 1, decay: 0.014, str, color });
    }

    step(ctx, dt, scale) {
      const k = Math.min(dt / 16.7, 3);
      for (const p of this.items) {
        p.x += p.vx * k;
        p.y += p.vy * k;
        if (p.kind !== "text") p.vy += 0.12 * k;
        if (p.rot !== undefined) p.rot += p.vr * k;
        p.life -= p.decay * k;
        if (p.life <= 0) continue;
        ctx.save();
        ctx.globalAlpha = clamp01(p.life);
        if (p.kind === "text") {
          ctx.font = `800 ${Math.round(22 * scale)}px Inter, Segoe UI, sans-serif`;
          ctx.textAlign = "center";
          ctx.fillStyle = p.color;
          ctx.shadowColor = p.color;
          ctx.shadowBlur = 14;
          ctx.fillText(p.str, p.x, p.y);
        } else if (p.kind === "rect") {
          ctx.translate(p.x, p.y);
          ctx.rotate(p.rot);
          ctx.fillStyle = p.color;
          ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
        } else {
          ctx.fillStyle = p.color;
          ctx.shadowColor = p.color;
          ctx.shadowBlur = 8;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }
      this.items = this.items.filter((p) => p.life > 0);
    }
  }

  // ---------- shared drawing helpers ----------
  function arrow(ctx, x, y, angle, len, width, color, alpha, outline) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(0, -width * 0.35);
    ctx.lineTo(len - width, -width * 0.35);
    ctx.lineTo(len - width, -width);
    ctx.lineTo(len, 0);
    ctx.lineTo(len - width, width);
    ctx.lineTo(len - width, width * 0.35);
    ctx.lineTo(0, width * 0.35);
    ctx.closePath();
    ctx.fill();
    if (outline) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    ctx.restore();
  }

  function star(ctx, x, y, outer, inner, points, rotation) {
    ctx.beginPath();
    for (let i = 0; i < points * 2; i++) {
      const r = i % 2 === 0 ? outer : inner;
      const a = rotation + (i * Math.PI) / points - Math.PI / 2;
      ctx.lineTo(x + Math.cos(a) * r, y + Math.sin(a) * r);
    }
    ctx.closePath();
  }

  // ---------- Catch ----------
  function catchScene(state) {
    return { ballR: state.ball[0], ballC: state.ball[1], paddle: state.paddle };
  }

  function drawCatch(ctx, S, s, o) {
    const c = S / GRID;
    const sky = ctx.createLinearGradient(0, 0, 0, S);
    sky.addColorStop(0, "#070b2a");
    sky.addColorStop(0.55, "#191245");
    sky.addColorStop(1, "#33124a");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, S, S);

    for (const st of STARS) {
      ctx.globalAlpha = 0.3 + 0.35 * Math.sin(o.time * 0.002 + st.tw);
      ctx.fillStyle = "#fff";
      const r = st.r * (o.mini ? 0.5 : 1);
      ctx.fillRect(st.x * S, st.y * S, r, r);
    }
    ctx.globalAlpha = 1;

    if (!o.mini) {
      // distant hills
      ctx.fillStyle = "rgba(76, 29, 149, 0.35)";
      ctx.beginPath();
      ctx.moveTo(0, S * 0.86);
      for (let x = 0; x <= S; x += S / 20) ctx.lineTo(x, S * 0.84 - Math.sin(x / S * 7 + 1) * S * 0.04 - Math.sin(x / S * 17) * S * 0.015);
      ctx.lineTo(S, S); ctx.lineTo(0, S);
      ctx.fill();
      // faint column guides
      ctx.strokeStyle = "rgba(167, 139, 250, 0.06)";
      for (let i = 1; i < GRID; i++) { ctx.beginPath(); ctx.moveTo(i * c, 0); ctx.lineTo(i * c, S); ctx.stroke(); }
    }

    // ground
    const groundTop = (GRID - 1) * c + c * 0.78;
    const ground = ctx.createLinearGradient(0, groundTop, 0, S);
    ground.addColorStop(0, "#5b21b6");
    ground.addColorStop(1, "#1e0b3a");
    ctx.fillStyle = ground;
    ctx.fillRect(0, groundTop, S, S - groundTop);
    ctx.fillStyle = "rgba(196, 181, 253, 0.6)";
    ctx.fillRect(0, groundTop, S, 1.5);

    const bx = (s.ballC + 0.5) * c, by = (s.ballR + 0.5) * c, br = c * 0.32;

    // light beam showing where the ball will land
    const beam = ctx.createLinearGradient(0, by, 0, groundTop);
    beam.addColorStop(0, "rgba(255, 176, 32, 0)");
    beam.addColorStop(1, "rgba(255, 176, 32, 0.16)");
    ctx.fillStyle = beam;
    ctx.fillRect(s.ballC * c + c * 0.1, by, c * 0.8, groundTop - by);

    // landing shadow, darker as the ball gets closer
    const near = clamp01(s.ballR / (GRID - 1));
    ctx.fillStyle = `rgba(0, 0, 0, ${0.15 + 0.35 * near})`;
    ctx.beginPath();
    ctx.ellipse(bx, groundTop + 2, br * (0.6 + 0.6 * near), br * 0.22, 0, 0, Math.PI * 2);
    ctx.fill();

    // ball trail
    if (o.trail) {
      o.trail.forEach((p, i) => {
        const t = (i + 1) / o.trail.length;
        ctx.fillStyle = `rgba(255, 140, 30, ${0.25 * t})`;
        ctx.beginPath();
        ctx.arc((p[1] + 0.5) * c, (p[0] + 0.5) * c, br * t * 0.9, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // ball
    ctx.save();
    ctx.shadowColor = "#ff9a1f";
    ctx.shadowBlur = o.mini ? 6 : 26;
    const ballGrad = ctx.createRadialGradient(bx - br * 0.35, by - br * 0.4, br * 0.1, bx, by, br);
    ballGrad.addColorStop(0, "#fffbe8");
    ballGrad.addColorStop(0.35, "#ffc34d");
    ballGrad.addColorStop(1, "#ff5f00");
    ctx.fillStyle = ballGrad;
    ctx.beginPath();
    ctx.arc(bx, by, br, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // paddle
    const px = (s.paddle + 0.5) * c, pw = c * 0.94, ph = c * 0.26;
    const py = groundTop - ph - c * 0.04;
    ctx.save();
    ctx.shadowColor = "#22d3ee";
    ctx.shadowBlur = o.mini ? 6 : 20;
    const padGrad = ctx.createLinearGradient(px - pw / 2, 0, px + pw / 2, 0);
    padGrad.addColorStop(0, "#22d3ee");
    padGrad.addColorStop(0.5, "#818cf8");
    padGrad.addColorStop(1, "#c084fc");
    ctx.fillStyle = padGrad;
    ctx.beginPath();
    ctx.roundRect(px - pw / 2, py, pw, ph, ph / 2);
    ctx.fill();
    ctx.restore();
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.fillRect(px - pw / 2 + ph / 2, py + 2, pw - ph, 1.5);

    // the agent's intentions: how likely each move is
    if (o.probs && !o.mini) {
      const pL = o.probs[LEFT] || 0, pR = o.probs[RIGHT] || 0, pN = o.probs[NOOP] || 0;
      const ay = py - c * 0.55;
      arrow(ctx, px - c * 0.15, ay, Math.PI, c * (0.3 + 0.9 * pL), c * 0.18, "#a5f3fc", 0.2 + 0.8 * pL, o.action === LEFT);
      arrow(ctx, px + c * 0.15, ay, 0, c * (0.3 + 0.9 * pR), c * 0.18, "#a5f3fc", 0.2 + 0.8 * pR, o.action === RIGHT);
      ctx.globalAlpha = 0.2 + 0.8 * pN;
      ctx.fillStyle = "#a5f3fc";
      ctx.beginPath();
      ctx.arc(px, ay, c * (0.06 + 0.12 * pN), 0, Math.PI * 2);
      ctx.fill();
      if (o.action === NOOP) { ctx.globalAlpha = 1; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke(); }
      ctx.globalAlpha = 1;
    }
  }

  // ---------- Reach ----------
  function reachScene(state) {
    return { ar: state.agent[0], ac: state.agent[1], gr: state.goal[0], gc: state.goal[1] };
  }

  function drawReach(ctx, S, s, o) {
    const c = S / GRID;
    ctx.fillStyle = "#050b18";
    ctx.fillRect(0, 0, S, S);
    for (let r = 0; r < GRID; r++) {
      for (let col = 0; col < GRID; col++) {
        ctx.fillStyle = (r + col) % 2 ? "#0f1b36" : "#0b152c";
        ctx.fillRect(col * c + 1, r * c + 1, c - 2, c - 2);
        if (!o.mini) {
          ctx.fillStyle = "rgba(125, 211, 252, 0.04)";
          ctx.fillRect(col * c + 1, r * c + 1, c - 2, 2);
        }
      }
    }

    // walls (only the Maze has them)
    if (s.walls) {
      for (const [wr, wc] of s.walls) {
        const x = wc * c, y = wr * c;
        const wall = ctx.createLinearGradient(x, y, x, y + c);
        wall.addColorStop(0, "#43306b");
        wall.addColorStop(1, "#1f1636");
        ctx.fillStyle = wall;
        ctx.beginPath();
        ctx.roundRect(x + 1, y + 1, c - 2, c - 2, Math.max(2, c * 0.14));
        ctx.fill();
        if (!o.mini) {
          ctx.fillStyle = "rgba(244, 114, 182, 0.22)";
          ctx.fillRect(x + c * 0.14, y + 2, c * 0.72, 2);
        }
      }
    }

    const gx = (s.gc + 0.5) * c, gy = (s.gr + 0.5) * c;
    const ax = (s.ac + 0.5) * c, ay = (s.ar + 0.5) * c;

    // glow of the goal on the floor
    const halo = ctx.createRadialGradient(gx, gy, 0, gx, gy, c * 2.2);
    halo.addColorStop(0, "rgba(250, 204, 21, 0.28)");
    halo.addColorStop(1, "rgba(250, 204, 21, 0)");
    ctx.fillStyle = halo;
    ctx.fillRect(0, 0, S, S);

    // footprints of this episode
    if (o.trail) {
      o.trail.forEach((p, i) => {
        const t = (i + 1) / o.trail.length;
        ctx.fillStyle = `rgba(34, 211, 238, ${0.08 + 0.3 * t})`;
        ctx.beginPath();
        ctx.arc((p[1] + 0.5) * c, (p[0] + 0.5) * c, c * 0.1 * (0.6 + t), 0, Math.PI * 2);
        ctx.fill();
      });
    }

    if (!o.mini && !s.walls) { // a straight guide line would be misleading with walls in the way
      ctx.save();
      ctx.setLineDash([4, 6]);
      ctx.lineDashOffset = -o.time * 0.02;
      ctx.strokeStyle = "rgba(250, 204, 21, 0.25)";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(gx, gy); ctx.stroke();
      ctx.restore();
    }

    // goal: rotating ring + pulsing star
    const pulse = 0.5 + 0.5 * Math.sin(o.time * 0.006);
    ctx.save();
    ctx.translate(gx, gy);
    ctx.rotate(o.time * 0.001);
    ctx.setLineDash([c * 0.12, c * 0.08]);
    ctx.strokeStyle = "rgba(253, 224, 71, 0.7)";
    ctx.lineWidth = Math.max(1, c * 0.04);
    ctx.beginPath(); ctx.arc(0, 0, c * (0.4 + 0.04 * pulse), 0, Math.PI * 2); ctx.stroke();
    ctx.restore();
    ctx.save();
    ctx.shadowColor = "#facc15";
    ctx.shadowBlur = o.mini ? 6 : 18 + 12 * pulse;
    const starGrad = ctx.createRadialGradient(gx, gy - c * 0.05, 0, gx, gy, c * 0.3);
    starGrad.addColorStop(0, "#fffbeb");
    starGrad.addColorStop(0.5, "#fde047");
    starGrad.addColorStop(1, "#f59e0b");
    ctx.fillStyle = starGrad;
    star(ctx, gx, gy, c * (0.27 + 0.03 * pulse), c * 0.12, 5, o.time * 0.0007);
    ctx.fill();
    ctx.restore();

    // agent: glowing orb with eyes looking where it is going
    const ar = c * 0.33;
    ctx.save();
    ctx.shadowColor = "#22d3ee";
    ctx.shadowBlur = o.mini ? 6 : 22;
    const body = ctx.createRadialGradient(ax - ar * 0.3, ay - ar * 0.35, ar * 0.1, ax, ay, ar);
    body.addColorStop(0, "#ecfeff");
    body.addColorStop(0.45, "#22d3ee");
    body.addColorStop(1, "#0e7490");
    ctx.fillStyle = body;
    ctx.beginPath(); ctx.arc(ax, ay, ar, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
    if (!o.mini) {
      const [dy, dx] = o.dir || [0, 1];
      for (const side of [-1, 1]) {
        const ex = ax + dx * ar * 0.35 + -dy * side * ar * 0.35;
        const ey = ay + dy * ar * 0.35 + dx * side * ar * 0.35 - ar * 0.1;
        ctx.fillStyle = "#fff";
        ctx.beginPath(); ctx.arc(ex, ey, ar * 0.2, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#082f49";
        ctx.beginPath(); ctx.arc(ex + dx * ar * 0.08, ey + dy * ar * 0.08, ar * 0.1, 0, Math.PI * 2); ctx.fill();
      }
    }

    // the agent's intentions
    if (o.probs && !o.mini) {
      for (const a of [UP, RIGHT, DOWN, LEFT]) {
        const p = o.probs[a] || 0;
        const [dy, dx] = DIRS[a];
        const angle = Math.atan2(dy, dx);
        arrow(ctx, ax + dx * ar * 1.05, ay + dy * ar * 1.05, angle, c * (0.15 + 0.55 * p), c * 0.15,
          "#a5f3fc", 0.15 + 0.85 * p, o.action === a);
      }
    }
  }

  // ---------- Maze (Reach with walls) ----------
  function mazeScene(state) {
    return { ...reachScene(state), walls: state.walls };
  }

  // ---------- Snake ----------
  function snakeScene(state) {
    return { hr: state.body[0][0], hc: state.body[0][1], fr: state.food[0], fc: state.food[1],
      eaten: state.eaten, body: state.body };
  }

  function drawSnake(ctx, S, s, o) {
    const c = S / GRID;
    ctx.fillStyle = "#04120d";
    ctx.fillRect(0, 0, S, S);
    for (let r = 0; r < GRID; r++) {
      for (let col = 0; col < GRID; col++) {
        ctx.fillStyle = (r + col) % 2 ? "#0b2018" : "#081811";
        ctx.fillRect(col * c + 1, r * c + 1, c - 2, c - 2);
      }
    }
    // the edge is deadly
    ctx.strokeStyle = "rgba(248, 113, 113, 0.55)";
    ctx.lineWidth = Math.max(2, c * 0.07);
    ctx.strokeRect(1, 1, S - 2, S - 2);

    // food
    const fx = (s.fc + 0.5) * c, fy = (s.fr + 0.5) * c;
    const pulse = 0.5 + 0.5 * Math.sin(o.time * 0.006);
    const halo = ctx.createRadialGradient(fx, fy, 0, fx, fy, c * 2);
    halo.addColorStop(0, "rgba(251, 113, 133, 0.25)");
    halo.addColorStop(1, "rgba(251, 113, 133, 0)");
    ctx.fillStyle = halo;
    ctx.fillRect(0, 0, S, S);
    ctx.save();
    ctx.shadowColor = "#fb7185";
    ctx.shadowBlur = o.mini ? 6 : 14 + 10 * pulse;
    const apple = ctx.createRadialGradient(fx - c * 0.08, fy - c * 0.1, 0, fx, fy, c * 0.3);
    apple.addColorStop(0, "#fecdd3");
    apple.addColorStop(0.5, "#fb7185");
    apple.addColorStop(1, "#be123c");
    ctx.fillStyle = apple;
    ctx.beginPath();
    ctx.arc(fx, fy, c * (0.25 + 0.02 * pulse), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // body: the head slides smoothly, the other segments sit on their cells
    const pts = [[(s.hc + 0.5) * c, (s.hr + 0.5) * c]];
    for (const p of s.body.slice(1)) pts.push([(p[1] + 0.5) * c, (p[0] + 0.5) * c]);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (let i = pts.length - 1; i >= 1; i--) {
      const t = 1 - i / pts.length; // 1 near the head, near 0 at the tail
      ctx.strokeStyle = `hsl(145, 65%, ${20 + 28 * t}%)`;
      ctx.lineWidth = c * (0.44 + 0.12 * t);
      ctx.beginPath();
      ctx.moveTo(pts[i][0], pts[i][1]);
      ctx.lineTo(pts[i - 1][0], pts[i - 1][1]);
      ctx.stroke();
    }

    // head with eyes that look where it is going
    const hx = pts[0][0], hy = pts[0][1], hr = c * 0.34;
    ctx.save();
    ctx.shadowColor = "#4ade80";
    ctx.shadowBlur = o.mini ? 5 : 18;
    const head = ctx.createRadialGradient(hx - hr * 0.3, hy - hr * 0.35, hr * 0.1, hx, hy, hr);
    head.addColorStop(0, "#ecfccb");
    head.addColorStop(0.45, "#4ade80");
    head.addColorStop(1, "#15803d");
    ctx.fillStyle = head;
    ctx.beginPath();
    ctx.arc(hx, hy, hr, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    if (!o.mini) {
      const [dy, dx] = o.dir || [0, 1];
      for (const side of [-1, 1]) {
        const ex = hx + dx * hr * 0.35 + -dy * side * hr * 0.35;
        const ey = hy + dy * hr * 0.35 + dx * side * hr * 0.35 - hr * 0.1;
        ctx.fillStyle = "#fff";
        ctx.beginPath(); ctx.arc(ex, ey, hr * 0.2, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#052e16";
        ctx.beginPath(); ctx.arc(ex + dx * hr * 0.08, ey + dy * hr * 0.08, hr * 0.1, 0, Math.PI * 2); ctx.fill();
      }
      // progress: apples eaten out of the 5 needed to win
      const eaten = Math.round(s.eaten);
      for (let i = 0; i < SNAKE_FOODS; i++) {
        ctx.fillStyle = i < eaten ? "#fb7185" : "rgba(148, 163, 184, 0.25)";
        ctx.beginPath();
        ctx.arc(c * 0.35 + i * c * 0.26, c * 0.3, c * 0.075, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // the agent's intentions
    if (o.probs && !o.mini) {
      for (const a of [UP, RIGHT, DOWN, LEFT]) {
        const p = o.probs[a] || 0;
        const [dy, dx] = DIRS[a];
        arrow(ctx, hx + dx * hr * 1.05, hy + dy * hr * 1.05, Math.atan2(dy, dx), c * (0.15 + 0.55 * p),
          c * 0.15, "#bbf7d0", 0.15 + 0.85 * p, o.action === a);
      }
    }
  }

  const SCENES = {
    catch: { toScene: catchScene, draw: drawCatch },
    reach: { toScene: reachScene, draw: drawReach },
    maze: { toScene: mazeScene, draw: drawReach },
    snake: { toScene: snakeScene, draw: drawSnake },
  };

  // ---------- live renderer with smooth movement and effects ----------
  class Renderer {
    constructor(canvas, game) {
      this.canvas = canvas;
      this.game = game;
      this.scene = SCENES[game];
      this.particles = new Particles();
      this.frame = null;
      this.t0 = 0;
      this.dur = 0;
      this.trail = [];
      this.ballTrail = [];
      this.dir = [0, 1];
      this.flash = 0;
      this.flashColor = "248,113,113";
      this.pending = null;
      this.last = performance.now();
    }

    push(frame, intervalMs) {
      this.frame = frame;
      this.t0 = performance.now();
      this.dur = frame.reset ? 0 : Math.min(intervalMs * 0.85, 350);
      if (frame.reset) { this.trail = []; this.ballTrail = []; }
      if ((this.game === "reach" || this.game === "maze") && !frame.reset) {
        this.trail.push(frame.prev.agent);
        if (this.trail.length > 40) this.trail.shift();
      }
      if (frame.action in DIRS) this.dir = DIRS[frame.action];
      if (frame.event) this.pending = { event: frame.event, at: this.t0 + this.dur };
    }

    _trigger(event, S) {
      const c = S / GRID, st = this.frame.state;
      if (event === "timeout") {
        this.flash = 0.6;
        this.flashColor = "148,163,196";
        this.particles.text(S / 2, S / 2, "time up", "#cbd5e1");
        return;
      }
      if (this.game === "catch") {
        const x = (st.paddle + 0.5) * c, y = (GRID - 1) * c + c * 0.4;
        if (event === "catch") {
          this.particles.burst(x, y, "#fde047", 34, 5, 3.5);
          this.particles.burst(x, y, "#34d399", 18, 3, 3);
          this.particles.text(x, y - c * 0.8, "+1", "#34d399");
        } else if (event === "miss") {
          this.flash = 1;
          this.flashColor = "248,113,113";
          this.particles.burst((st.ball[1] + 0.5) * c, y, "#f87171", 24, 3.5, 3);
          this.particles.text((st.ball[1] + 0.5) * c, y - c * 0.8, "−1", "#f87171");
        }
      } else if (this.game === "reach" || this.game === "maze") {
        const gx = (st.goal[1] + 0.5) * c, gy = (st.goal[0] + 0.5) * c;
        if (event === "goal") {
          this.particles.confetti(gx, gy, 46);
          this.particles.text(gx, gy - c * 0.6, "+1", "#fde047");
        }
      } else if (this.game === "snake") {
        const x = (st.body[0][1] + 0.5) * c, y = (st.body[0][0] + 0.5) * c;
        if (event === "food") {
          this.particles.burst(x, y, "#fb7185", 22, 4, 3);
          this.particles.text(x, y - c * 0.7, "+0.2", "#fb7185");
        } else if (event === "win") {
          this.particles.confetti(x, y, 46);
          this.particles.text(x, y - c * 0.7, "+0.2", "#fb7185");
          this.particles.text(S / 2, S * 0.4, "WIN", "#fde047");
        } else if (event === "crash") {
          this.flash = 1;
          this.flashColor = "248,113,113";
          this.particles.burst(x, y, "#f87171", 28, 4, 3);
          this.particles.text(x, y - c * 0.7, "−1", "#f87171");
        }
      }
    }

    draw(now) {
      const { ctx, w, h } = Charts.setup(this.canvas);
      const S = Math.min(w, h);
      const dt = now - this.last;
      this.last = now;
      ctx.clearRect(0, 0, w, h);
      if (!this.scene) return;
      if (!this.frame) {
        ctx.fillStyle = "#6b7796";
        ctx.font = "13px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Waiting for the live player…", S / 2, S / 2);
        return;
      }
      const t = this.dur ? clamp01((now - this.t0) / this.dur) : 1;
      const e = easeOut(t);
      const a = this.scene.toScene(this.frame.prev), b = this.scene.toScene(this.frame.state);
      const s = {};
      // numbers slide smoothly between the two moments; lists (walls, snake body) are taken as they are
      for (const k of Object.keys(b)) s[k] = typeof b[k] === "number" ? lerp(a[k], b[k], e) : b[k];

      if (this.game === "catch") {
        this.ballTrail.push([s.ballR, s.ballC]);
        if (this.ballTrail.length > 10) this.ballTrail.shift();
      }
      if (this.pending && now >= this.pending.at) {
        this._trigger(this.pending.event, S);
        this.pending = null;
      }

      this.scene.draw(ctx, S, s, {
        time: now, mini: false,
        probs: this.frame.probs, action: this.frame.action,
        trail: this.game === "catch" ? this.ballTrail : this.trail,
        dir: this.dir,
      });
      this.particles.step(ctx, dt, S / 400);
      if (this.flash > 0) {
        ctx.fillStyle = `rgba(${this.flashColor},${0.28 * this.flash})`;
        ctx.fillRect(0, 0, S, S);
        this.flash = Math.max(0, this.flash - dt / 500);
      }
    }
  }

  /** Small, effect-free drawing for the training arena thumbnails. */
  function drawMini(canvas, entry, now) {
    const { ctx, w, h } = Charts.setup(canvas);
    const S = Math.min(w, h);
    ctx.clearRect(0, 0, w, h);
    const scene = SCENES[entry.game];
    if (!scene) return;
    scene.draw(ctx, S, scene.toScene(entry.state), { time: now, mini: true });
  }

  return { Renderer, drawMini, COLORS, TITLES, DESCRIPTIONS, color: (g) => COLORS[g] || "#a78bfa" };
})();
