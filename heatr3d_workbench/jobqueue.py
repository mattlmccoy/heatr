"""On-disk job queue for workbench runs (server-side, stdlib only).

State lives under <store>/queue.json plus one run dir per job (the run dir is
the same outputs_eqs/_heatr3d/<12hex> contract the Results tab scans). The
queue survives server restarts: refresh() reconciles recorded PIDs against
reality, and a dead PID without results.json becomes stale_failed - loud,
never silently healthy.

States: queued -> running -> done | failed | stale_failed | cancelled.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ACTIVE = ("queued", "running")


class Queue:
    def __init__(self, store: Path, max_concurrent: int = 1) -> None:
        self.store = Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        self.qfile = self.store / "queue.json"
        self.max_concurrent = max_concurrent

    # ---- persistence -----------------------------------------------------
    def _load(self) -> List[Dict[str, Any]]:
        if not self.qfile.exists():
            return []
        try:
            return json.loads(self.qfile.read_text(encoding="utf-8"))["jobs"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("queue.json unreadable (%s); starting empty", e)
            return []

    def _save(self, jobs: List[Dict[str, Any]]) -> None:
        tmp = self.qfile.with_suffix(".tmp")
        tmp.write_text(json.dumps({"jobs": jobs}, indent=1), encoding="utf-8")
        tmp.replace(self.qfile)

    # ---- API -------------------------------------------------------------
    def run_dir(self, jid: str) -> Path:
        return self.store / jid

    def jobs(self) -> List[Dict[str, Any]]:
        return self._load()

    def get(self, jid: str) -> Optional[Dict[str, Any]]:
        for j in self._load():
            if j["id"] == jid:
                return j
        return None

    def enqueue(self, cfg: Dict[str, Any], campaign: Optional[str] = None) -> str:
        jid = uuid.uuid4().hex[:12]
        d = self.run_dir(jid)
        d.mkdir(parents=True, exist_ok=True)
        cfg = dict(cfg)
        cfg["out_dir"] = str(d)
        (d / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        jobs = self._load()
        jobs.append({"id": jid, "state": "queued", "pid": None,
                     "campaign": campaign, "created": time.time(),
                     "shape": cfg.get("library_shape") or cfg.get("shape")
                     or cfg.get("stl_name") or "?",
                     "cfg_summary": {k: cfg.get(k) for k in
                                     ("source", "n", "fgm", "densify", "phase_update")}})
        self._save(jobs)
        return jid

    def next_queued(self) -> Optional[str]:
        for j in self._load():
            if j["state"] == "queued":
                return j["id"]
        return None

    def claimable(self) -> bool:
        running = sum(1 for j in self._load() if j["state"] == "running")
        return running < self.max_concurrent

    def _set(self, jid: str, **updates: Any) -> None:
        jobs = self._load()
        for j in jobs:
            if j["id"] == jid:
                j.update(updates)
        self._save(jobs)

    def mark_running(self, jid: str, pid: int) -> None:
        self._set(jid, state="running", pid=pid, started=time.time())

    def cancel(self, jid: str) -> bool:
        j = self.get(jid)
        if j is None or j["state"] not in _ACTIVE:
            return False
        if j["state"] == "running" and j.get("pid"):
            try:
                os.kill(int(j["pid"]), signal.SIGTERM)
            except (ProcessLookupError, PermissionError) as e:
                logger.warning("cancel %s: kill failed (%s)", jid, e)
        self._set(jid, state="cancelled", ended=time.time())
        return True

    @staticmethod
    def _pid_alive(pid: Optional[int]) -> bool:
        if not pid:
            return False
        pid = int(pid)
        # Reap first: a crashed child of THIS process is a zombie until waited,
        # and os.kill(pid, 0) reports zombies as alive (live finding 2026-08-02).
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done == pid:
                return False
        except ChildProcessError:
            pass          # not our child; fall through to the signal probe
        except OSError:
            pass
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def refresh(self) -> None:
        """Reconcile running jobs against reality (restart-safe)."""
        jobs = self._load()
        changed = False
        for j in jobs:
            if j["state"] != "running":
                continue
            has_results = (self.run_dir(j["id"]) / "results.json").exists()
            if has_results:
                j["state"] = "done"; j["ended"] = time.time(); changed = True
            elif not self._pid_alive(j.get("pid")):
                j["state"] = "stale_failed"; j["ended"] = time.time(); changed = True
        if changed:
            self._save(jobs)
