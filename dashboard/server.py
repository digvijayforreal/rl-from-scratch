"""Live RL playground server.

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

from dashboard.trainer import Showcase, Trainer, init_payload

STATIC_DIR = (Path(__file__).resolve().parent / "static").resolve()
MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8", ".svg": "image/svg+xml"}
STREAM_HZ = 12


def build_message(trainer, showcase, cursor):
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
    return msg


class Handler(BaseHTTPRequestHandler):
    trainer: Trainer = None
    showcase: Showcase = None

    def log_message(self, *args):  # keep the terminal quiet
        pass

    # ---------- helpers ----------

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code=200):
        self._send(code, json.dumps(data).encode(), "application/json")

    # ---------- routes ----------

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/stream":
            return self._stream()
        if path == "/api/checkpoints":
            return self._json(self.trainer.list_checkpoints())
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
        if length > 10_000:
            return self._json({"ok": False, "error": "Request too large"}, 413)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            result = self._command(body)
            self._json({"ok": True, **(result or {})})
        except (ValueError, KeyError, TypeError) as exc:
            self._json({"ok": False, "error": str(exc)}, 400)

    def _command(self, body):
        t, s = self.trainer, self.showcase
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
                msg = build_message(self.trainer, self.showcase, cursor)
                self.wfile.write(f"data: {json.dumps(msg, separators=(',', ':'))}\n\n".encode())
                self.wfile.flush()
                time.sleep(1 / STREAM_HZ)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return  # browser tab closed


def main():
    parser = argparse.ArgumentParser(description="Live RL playground")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("Measuring random baselines and building the network...")
    Handler.trainer = Trainer()
    Handler.showcase = Showcase(Handler.trainer)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}"
    print(f"RL Playground running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
