"""Link from the control system to ai_worker.py (the local image-generation process).

Everything that touches the network runs on one daemon thread, so Controller.update() never blocks:
jobs are queued with submit(), the worker's /status is polled about once a second, and finished
"show when ready" jobs surface through pop_shown(). No worker, no token file, a timeout or a bad
reply all just read as {"available": False, ...}; nothing here raises into the caller.
"""
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

TIMEOUT = 3.0


class AIClient:
    def __init__(self, token_paths, url="http://127.0.0.1:8766", poll_s=1.0, timeout=TIMEOUT, start=True):
        self.token_paths = [Path(p) for p in token_paths]
        self.url, self.poll_s, self.timeout = url.rstrip("/"), poll_s, timeout
        self._lock = threading.Lock()
        self._jobs = []                 # (due time, job dict)
        self._cancel = False
        self._show_ids = set()
        self._shown = []
        self._status = {"available": False, "error": "starting" if start else "off"}
        self._note = ""                 # last submit error, shown in the panel until the next good submit
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        self.enabled = bool(start)
        self.hold = False               # governor.py: True = keep new jobs queued here (sent once released)
        if start:
            self._thread = threading.Thread(target=self._run, name="ai-link", daemon=True)
            self._thread.start()

    # ---------------------------------------------------------------- caller side (any thread)
    def submit(self, job, delay=0.0):
        with self._lock:
            if len(self._jobs) < 32:
                self._jobs.append((time.monotonic() + max(0.0, delay), dict(job)))
        self._wake.set()

    def set_hold(self, on):
        """Resource governor: hold (True) or release new jobs. Releasing wakes the link so queued jobs go now."""
        with self._lock:
            self.hold = bool(on)
        if not on:
            self._wake.set()

    def cancel(self):
        with self._lock:
            self._jobs = []
            self._cancel = True
        self._wake.set()

    def status(self):
        with self._lock:
            s = dict(self._status)
            if self._note:
                s["note"] = self._note
            s["pending"] = len(self._jobs)
            return s

    def pop_shown(self):
        if not self._shown:
            return []
        with self._lock:
            out, self._shown = self._shown, []
        return out

    def close(self):
        self._stop.set()
        self._wake.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=self.timeout + 1)

    # ---------------------------------------------------------------- worker thread
    def token(self):
        for p in self.token_paths:
            try:
                t = p.read_text(encoding="utf-8").strip()
                if t:
                    return t
            except OSError:
                pass
        return None

    def _http(self, path, body=None, timeout=None):
        """-> parsed JSON dict. Raises OSError / ValueError on any failure."""
        tok = self.token()
        if not tok:
            raise OSError("no .remote_token (start ai_worker.py)")
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={"X-Token": tok, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                out = json.loads(r.read(1 << 20).decode("utf-8"))
        except urllib.error.HTTPError as ex:
            try:
                msg = json.loads(ex.read(4096).decode("utf-8")).get("error", "")
            except Exception:
                msg = ""
            raise OSError(f"worker said {ex.code} {msg}".strip())
        if not isinstance(out, dict):
            raise ValueError("unexpected reply")
        return out

    def _take_due(self, now):
        """Pop the jobs whose time has come (none while held by the resource governor)."""
        with self._lock:
            if self.hold:
                return []
            due = [j for t, j in self._jobs if t <= now]
            self._jobs = [(t, j) for t, j in self._jobs if t > now]
            return due

    def _run(self):
        next_poll = 0.0
        while not self._stop.is_set():
            try:
                now = time.monotonic()
                due = self._take_due(now)
                with self._lock:
                    cancel, self._cancel = self._cancel, False
                    wait = min([t - now for t, _ in self._jobs if not self.hold] + [self.poll_s])
                if cancel:
                    self._try(lambda: self._http("/cancel", {}), "cancel")
                    with self._lock:
                        self._show_ids = set()
                for job in due:
                    self._send_job(job)
                if cancel or due or now >= next_poll:
                    ok = self.poll_once()
                    next_poll = time.monotonic() + (self.poll_s if ok else max(self.poll_s, 3.0))
                    wait = min(wait, self.poll_s)
            except Exception:
                wait = self.poll_s
            self._wake.wait(max(0.02, wait))
            self._wake.clear()

    def _try(self, fn, what):
        try:
            out = fn()
            self._note = ""
            return out
        except Exception as ex:
            self._note = f"{what} failed: {ex}"[:200]
            return None

    def _send_job(self, job):
        show = bool(job.get("show"))
        out = self._try(lambda: self._http("/job", job), "generate")
        if out is not None and show and "id" in out:
            with self._lock:
                self._show_ids.add(out["id"])
        return out

    def poll_once(self):
        """Fetch /status once and update the snapshot. Returns True if the worker answered."""
        try:
            s = self._http("/status", timeout=min(self.timeout, 2.0))
        except Exception as ex:
            with self._lock:
                self._status = {"available": False, "error": str(ex)[:200]}
            return False
        done = s.get("done") if isinstance(s.get("done"), list) else []
        queue_ = s.get("queue") if isinstance(s.get("queue"), list) else []
        clean = dict(available=True, loaded=bool(s.get("loaded")), loading=bool(s.get("loading")),
                     busy=bool(s.get("busy")), queue=len(queue_), error=str(s.get("error") or "")[:300],
                     base=str(s.get("base") or "")[:60], done=[])
        with self._lock:
            for d in done[-8:]:
                if not isinstance(d, dict):
                    continue
                name = d.get("name")
                clean["done"].append(dict(id=d.get("id"), name=str(name)[:120] if name else "",
                                          prompt=str(d.get("prompt") or "")[:80], secs=d.get("secs")))
            for d in done:
                if isinstance(d, dict) and d.get("id") in self._show_ids and isinstance(d.get("name"), str):
                    self._show_ids.discard(d["id"])
                    self._shown.append(d["name"])
            self._status = clean
        return True
