"""Background queue for experiments. One job runs at a time, so they don't fight over the CPU."""

import threading
import time
import traceback

from dashboard import storage
from experiments.plasticity import Cancelled, run_experiment, summarize, total_steps, validate_spec


class JobQueue:
    def __init__(self, trainer):
        self.trainer = trainer
        self.jobs = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.last_sps = None  # measured speed of the last experiment (for time estimates)
        self._mark_interrupted()
        threading.Thread(target=self._worker, daemon=True).start()

    def _mark_interrupted(self):
        """Experiments that were running when the server stopped can't continue."""
        for meta in storage.list_experiments():
            if meta.get("status") in ("queued", "waiting", "running"):
                meta["status"] = "interrupted"
                storage.write_json(storage.EXPERIMENTS_DIR / meta["id"] / "meta.json", meta)

    # ---------- API ----------

    def submit(self, spec):
        spec = validate_spec(spec)
        base = f"exp_{time.strftime('%Y%m%d-%H%M%S')}"
        exp_id, n = base, 1
        while (storage.EXPERIMENTS_DIR / exp_id).exists():
            n += 1
            exp_id = f"{base}_{n}"
        folder = storage.EXPERIMENTS_DIR / exp_id
        folder.mkdir(parents=True)
        job = {"id": exp_id, "name": spec["name"], "status": "queued", "progress": 0.0,
               "detail": "Waiting in the queue", "eta": None, "spec": spec, "cancel": False,
               "total_steps": total_steps(spec)}
        storage.write_json(folder / "meta.json", {
            "id": exp_id, "kind": "experiment", "name": spec["name"],
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "queued", "progress": 0.0,
            "spec": spec, "notes": "", "total_steps": job["total_steps"]})
        with self.lock:
            self.jobs.append(job)
        self.wake.set()
        self.trainer.log("info", f"Experiment '{spec['name']}' queued ({exp_id}).")
        return exp_id

    def cancel(self, job_id):
        for job in self.jobs:
            if job["id"] == job_id:
                job["cancel"] = True
                if job["status"] == "queued":
                    job.update(status="cancelled", detail="Cancelled before it started")
                    self._write_meta(job)
                return
        raise ValueError(f"No job {job_id}")

    def public(self):
        with self.lock:
            return [{k: v for k, v in j.items() if k not in ("spec", "cancel")} for j in self.jobs[-20:]]

    # ---------- worker ----------

    def _write_meta(self, job, **extra):
        path = storage.EXPERIMENTS_DIR / job["id"] / "meta.json"
        meta = storage.read_json(path, default={}) or {}
        meta.update(status=job["status"], progress=round(job["progress"], 4), **extra)
        storage.write_json(path, meta)

    def _worker(self):
        while True:
            job = next((j for j in self.jobs if j["status"] == "queued"), None)
            if job is None:
                self.wake.wait(1.0)
                self.wake.clear()
                continue
            while self.trainer.status == "running" and not job["cancel"]:
                job.update(status="waiting", detail="Waiting: live training is using the CPU. Pause it to start.")
                time.sleep(1)
            if job["cancel"]:
                job.update(status="cancelled", detail="Cancelled")
                self._write_meta(job)
                continue

            job.update(status="running", detail="Starting…", started=time.time())
            self._write_meta(job, started=time.strftime("%Y-%m-%d %H:%M:%S"))
            self.trainer.log("info", f"Experiment '{job['name']}' started.")
            last_write = [0.0]

            def on_progress(fraction, detail, sps, job=job):
                job.update(progress=fraction, detail=detail, sps=round(sps))
                job["eta"] = (1 - fraction) * job["total_steps"] / sps if sps > 0 else None
                self.last_sps = sps
                if time.time() - last_write[0] > 5:
                    self._write_meta(job)
                    last_write[0] = time.time()

            folder = storage.EXPERIMENTS_DIR / job["id"]
            try:
                run_experiment(job["spec"], folder, on_progress, lambda job=job: job["cancel"])
                job.update(status="done", progress=1.0, detail="Finished", eta=0)
                self.trainer.log("success", f"Experiment '{job['name']}' finished. See the Experiments tab.")
            except Cancelled:
                job.update(status="cancelled", detail="Cancelled", eta=None)
                self.trainer.log("warn", f"Experiment '{job['name']}' cancelled.")
            except Exception as exc:
                job.update(status="failed", detail=f"Failed: {exc!r}", eta=None)
                self.trainer.log("warn", f"Experiment '{job['name']}' failed: {exc!r}")
                traceback.print_exc()
            self._write_meta(job, finished=time.strftime("%Y-%m-%d %H:%M:%S"))


def experiment_detail(exp_id):
    folder = storage.folder("experiment", exp_id)
    meta = storage.read_json(folder / "meta.json", default={}) or {}
    summary = summarize(folder) if (folder / "spec.json").exists() else None
    return {"meta": meta, "summary": summary}
