"""RL research lab server: live training, testing, experiments and run history.

Run from the project folder:
    uv run python -m dashboard.server
Then open http://127.0.0.1:8000

Uses only the Python standard library for the web part. It listens on 127.0.0.1
(your computer only) and has no login, so don't expose it to a network.
"""

import argparse
import json
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from dashboard import lab, storage
from dashboard.jobs import JobQueue, experiment_detail
from dashboard.trainer import Showcase, Trainer, clean, init_payload
from experiments.plasticity import DEFAULT_SPEC, INTERVENTIONS, QUICK_SPEC, SPEC_LIMITS
from experiments.train import GAMES

STATIC_DIR = (Path(__file__).resolve().parent / "static").resolve()
MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8", ".svg": "image/svg+xml"}
STREAM_HZ = 12
MAX_BODY = 64_000


def build_message(trainer, showcase, queue, cursor):
    """One update for the browser. `cursor` remembers what this browser already has."""
    msg = {
        "run": trainer.run_id,
        "status": trainer.status,
        "phase": trainer.phase,
        "phase_progress": trainer.phase_progress,
        "update": trainer.update,
        "num_updates": trainer.num_updates,
        "env_steps": trainer.env_steps,
        "sps": trainer.sps,
        "elapsed": trainer.elapsed(),
        "lr": trainer.lr,
        "delay_ms": trainer.delay_ms,
        "demo_fps": showcase.fps,
        "demo_mode": showcase.mode,
        "arena": trainer.arena,
        "jobs": queue.public(),
    }
    if cursor.get("run") != trainer.run_id:
        cursor.update(run=trainer.run_id, hist=0, brain=-1)
        msg["init"] = init_payload(trainer)

    history = trainer.history
    if len(history) > cursor["hist"]:
        msg["history"] = history[cursor["hist"]:]
        cursor["hist"] = len(history)

    events = trainer.events
    if len(events) > cursor.get("events", 0):
        msg["events"] = events[cursor.get("events", 0):]
        cursor["events"] = len(events)

    brain = trainer.brain
    if brain["version"] != cursor["brain"]:
        msg["brain"] = brain
        cursor["brain"] = brain["version"]

    demo = {}
    seen = cursor.setdefault("seq", {})
    for game, frame in list(showcase.frames.items()):
        if frame["seq"] != seen.get(game):
            demo[game] = frame
            seen[game] = frame["seq"]
    if demo:
        msg["demo"] = demo
    return clean(msg)


class Handler(BaseHTTPRequestHandler):
    trainer: Trainer = None
    showcase: Showcase = None
    queue: JobQueue = None

    def log_message(self, *args):  # keep the terminal quiet
        pass

    # ---------- helpers ----------

    def _send(self, code, body, content_type, extra_headers=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code=200):
        self._send(code, json.dumps(clean(data)).encode(), "application/json")

    def _error(self, exc, code=400):
        self._json({"ok": False, "error": str(exc)}, code)

    # ---------- routes ----------

    def do_GET(self):
        url = urlsplit(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = url.path
        try:
            if path == "/api/stream":
                return self._stream()
            if path == "/api/checkpoints":
                return self._json(storage.list_checkpoints())
            if path == "/api/lab":
                return self._json({
                    "games": list(GAMES), "interventions": INTERVENTIONS,
                    "default_spec": DEFAULT_SPEC, "quick_spec": QUICK_SPEC,
                    "spec_limits": {k: [lo, hi] for k, (_, lo, hi) in SPEC_LIMITS.items()},
                    "last_sps": self.queue.last_sps})
            if path == "/api/runs":
                return self._json({"runs": storage.list_runs(), "experiments": storage.list_experiments()})
            if path == "/api/experiments":
                return self._json(storage.list_experiments())
            if path == "/api/experiment":
                return self._json(experiment_detail(query.get("id", "")))
            if path == "/api/export":
                name, body = storage.export_csv(query.get("kind", ""), query.get("id", ""))
                return self._send(200, body, "text/csv; charset=utf-8",
                                  {"Content-Disposition": f'attachment; filename="{name}"'})
        except ValueError as exc:
            return self._error(exc, 404)

        if path == "/":
            path = "/index.html"
        file = (STATIC_DIR / path.lstrip("/")).resolve()
        if not file.is_relative_to(STATIC_DIR) or not file.is_file():
            return self._send(404, b"Not found", "text/plain")
        self._send(200, file.read_bytes(), MIME.get(file.suffix, "application/octet-stream"))

    def do_POST(self):
        if self.path != "/api/control":
            return self._send(404, b"Not found", "text/plain")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._error("Request too large", 413)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            result = self._command(body)
            self._json({"ok": True, **(result or {})})
        except (ValueError, KeyError, TypeError) as exc:
            self._error(exc)

    def _command(self, body):
        t, s, q = self.trainer, self.showcase, self.queue
        cmd = body.get("cmd")
        if cmd == "start":
            t.start()
        elif cmd == "pause":
            t.pause()
        elif cmd == "reset":
            t.reset(body.get("config"))
        elif cmd == "save":
            return {"file": t.save()}
        elif cmd == "load":
            t.load(body["name"])
        elif cmd == "speed":
            t.delay_ms = min(max(int(body["delay_ms"]), 0), 500)
        elif cmd == "demo":
            if "fps" in body:
                s.fps = min(max(float(body["fps"]), 1.0), 30.0)
            if body.get("mode") in ("greedy", "sample"):
                s.mode = body["mode"]
        elif cmd == "evaluate":
            return {"result": lab.evaluate_brain(t, body["checkpoint"], body.get("episodes", 200),
                                                 body.get("mode", "greedy"), body.get("seed", 999))}
        elif cmd == "record":
            return {"result": lab.record_episode(t, body["checkpoint"], body["game"],
                                                 body.get("seed", 0), body.get("mode", "greedy"))}
        elif cmd == "experiment":
            return {"id": q.submit(body.get("spec"))}
        elif cmd == "cancel_job":
            q.cancel(body["id"])
        elif cmd == "notes":
            storage.set_notes(body["kind"], body["id"], body.get("text", ""))
        else:
            raise ValueError(f"Unknown command {cmd!r}")
        return None

    def _stream(self):
        """Server-Sent Events: push a JSON update to the browser 12 times a second."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        cursor = {}
        try:
            while True:
                msg = build_message(self.trainer, self.showcase, self.queue, cursor)
                self.wfile.write(f"data: {json.dumps(msg, separators=(',', ':'))}\n\n".encode())
                self.wfile.flush()
                time.sleep(1 / STREAM_HZ)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return  # browser tab closed


def main():
    parser = argparse.ArgumentParser(description="RL research lab")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("Measuring random baselines and building the network...")
    Handler.trainer = Trainer()
    Handler.showcase = Showcase(Handler.trainer)
    Handler.queue = JobQueue(Handler.trainer)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}"
    print(f"RL Lab running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
