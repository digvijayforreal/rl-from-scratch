"""Where runs, experiments and checkpoints live on disk, and safe access to them.

results/
  runs/<run_id>/          one folder per live training run (config, log.csv, meta, checkpoints)
  experiments/<exp_id>/   one folder per plasticity experiment (spec, runs/*.csv, summary, meta)
"""

import csv
import io
import json
import os
import re
import time
from pathlib import Path

# Set RL_RESULTS_DIR to keep a separate results folder (e.g. for testing).
RESULTS_DIR = Path(os.environ.get("RL_RESULTS_DIR") or Path(__file__).resolve().parent.parent / "results").resolve()
RUNS_DIR = RESULTS_DIR / "runs"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


def check_id(value):
    """Only plain folder names are accepted, so a request can't reach outside results/."""
    value = str(value)
    if not _SAFE_ID.match(value) or ".." in value:
        raise ValueError("Invalid id")
    return value


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return default


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for attempt in range(5):  # Windows can briefly lock a file another thread is reading
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    tmp.replace(path)


def folder(kind, item_id):
    base = {"run": RUNS_DIR, "experiment": EXPERIMENTS_DIR}.get(kind)
    if base is None:
        raise ValueError("kind must be 'run' or 'experiment'")
    path = base / check_id(item_id)
    if not path.is_dir():
        raise ValueError(f"{kind} {item_id} not found")
    return path


def resolve_checkpoint(rel):
    root = RESULTS_DIR.resolve()
    path = (root / str(rel)).resolve()
    if not path.is_relative_to(root) or path.suffix != ".pt" or not path.is_file():
        raise ValueError(f"Checkpoint {rel} not found.")
    return path


def list_checkpoints():
    if not RESULTS_DIR.exists():
        return []
    files = sorted(RESULTS_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.relative_to(RESULTS_DIR).as_posix(),
             "folder": p.parent.relative_to(RESULTS_DIR).as_posix(),
             "size_kb": round(p.stat().st_size / 1024, 1),
             "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))}
            for p in files]


def _list(base):
    items = []
    for meta_path in base.glob("*/meta.json") if base.exists() else []:
        meta = read_json(meta_path)
        if meta:
            items.append(meta)
    return sorted(items, key=lambda m: m.get("created", ""), reverse=True)


def list_runs():
    runs = _list(RUNS_DIR)
    for meta in runs:
        d = RUNS_DIR / meta["id"]
        meta["checkpoints"] = sorted(p.relative_to(RESULTS_DIR).as_posix() for p in d.glob("*.pt"))
    return runs


def list_experiments():
    return _list(EXPERIMENTS_DIR)


def set_notes(kind, item_id, text):
    path = folder(kind, item_id) / "meta.json"
    meta = read_json(path, default={"id": item_id})
    meta["notes"] = str(text)[:5000]
    write_json(path, meta)


def export_csv(kind, item_id):
    """Returns (filename, bytes) of a CSV with every logged row."""
    d = folder(kind, item_id)
    if kind == "run":
        path = d / "log.csv"
        if not path.exists():
            raise ValueError("This run has no logged updates yet.")
        return f"{item_id}.csv", path.read_bytes()
    rows, fields = [], []
    for path in sorted((d / "runs").glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for name in reader.fieldnames or []:
                if name not in fields:
                    fields.append(name)
            rows.extend(reader)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return f"{item_id}.csv", buf.getvalue().encode("utf-8")
