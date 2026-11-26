"""CLI backends for the ONLINE plugin. Every call here is a child process that may use the internet, so this
module is only imported once the ONLINE switch is on, and every call runs on a worker thread (never on the
render or HTTP threads).

Each tool uses its OWN existing login (Claude Code's OAuth, agy's / the gemini CLI's Google account). Nothing here
reads, copies or stores an API key, and no environment variable is added to the child.

  ClaudeBackend   `claude -p` (resolved with shutil.which: the npm shim is a .CMD), prompt on stdin, cwd = the
                  user's home, tools off, MCP servers off, no session saved.
  AgyBackend      Antigravity CLI. It only renders its answer to a real console, so it runs under a ConPTY
                  (pywinpty) from a small private folder; the prompt travels in prompt.txt and the answer comes
                  back in reply.json (screen text is the fallback). Slow to start: hard cap 600 s by default.
"""
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

IS_WIN = os.name == "nt"
NO_WINDOW = 0x08000000 if IS_WIN else 0


class BackendError(Exception):
    pass


class BackendTimeout(BackendError):
    pass


class BackendCancelled(BackendError):
    pass


def kill_tree(proc):
    """Kill a child we started and everything under it (a .CMD shim's real process is a grandchild)."""
    try:
        if proc.poll() is not None:
            return
        if IS_WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=10,
                           creationflags=NO_WINDOW)
        if proc.poll() is None:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_cli(argv, stdin_text="", cwd=None, timeout=120.0, idle=None, cancel=None, progress=None):
    """Run argv, feed stdin_text, return (returncode, stdout, stderr) as text.

    timeout = wall-clock cap; idle = max seconds without any output (None = no idle limit); cancel = threading.Event.
    On timeout / cancel the whole process tree is killed and BackendTimeout / BackendCancelled is raised.
    progress(chars_so_far) is called from this thread about twice a second.
    """
    try:
        proc = subprocess.Popen(list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=str(cwd) if cwd else None, creationflags=NO_WINDOW)
    except OSError as ex:
        raise BackendError(f"could not start {Path(str(argv[0])).name}: {ex}") from ex
    out, err = [], []
    last = [time.monotonic()]

    def pump(stream, sink):
        try:
            while True:
                chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
                if not chunk:
                    break
                sink.append(chunk)
                last[0] = time.monotonic()
        except Exception:
            pass

    def feed():
        try:
            if stdin_text:
                proc.stdin.write(stdin_text.encode("utf-8"))
            proc.stdin.close()
        except Exception:
            pass

    threads = [threading.Thread(target=pump, args=(proc.stdout, out), daemon=True),
               threading.Thread(target=pump, args=(proc.stderr, err), daemon=True),
               threading.Thread(target=feed, daemon=True)]
    for t in threads:
        t.start()
    start = time.monotonic()
    try:
        while proc.poll() is None:
            now = time.monotonic()
            if cancel is not None and cancel.is_set():
                raise BackendCancelled("cancelled")
            if now - start > timeout:
                raise BackendTimeout(f"no answer within {timeout:.0f} s")
            if idle is not None and now - last[0] > idle:
                raise BackendTimeout(f"no output for {idle:.0f} s")
            if progress is not None:
                try:
                    progress(sum(len(c) for c in out))
                except Exception:
                    pass
            time.sleep(0.05)
    except BackendError:
        kill_tree(proc)
        raise
    finally:
        if proc.poll() is None:
            kill_tree(proc)
        for t in threads[:2]:
            t.join(timeout=2)
    dec = lambda parts: b"".join(parts).decode("utf-8", "replace")
    return proc.returncode, dec(out), dec(err)


class ClaudeBackend:
    """Claude Code CLI in print mode. One call = one `claude -p` child process."""
    name = "claude"
    label = "Claude Code"

    def __init__(self, timeout=180.0, model=None, exe=None):
        self.timeout, self.model, self._exe = float(timeout), model, exe

    def exe(self):
        return self._exe or shutil.which("claude")

    def available(self):
        return bool(self.exe())

    def argv(self):
        # tools off + restricted + no MCP servers: the model can only answer with text; no session file is kept
        a = [self.exe(), "-p", "--output-format", "text", "--restricted", "--strict-mcp-config",
             "--disable-slash-commands", "--no-session-persistence"]
        if self.model:
            a += ["--model", self.model]
        return a

    def ask(self, prompt, cancel=None, progress=None):
        if not self.available():
            raise BackendError("the `claude` CLI is not installed (or not on PATH)")
        rc, out, err = run_cli(self.argv(), prompt, cwd=Path.home(), timeout=self.timeout, cancel=cancel,
                               progress=progress)
        if rc != 0 or not out.strip():
            raise BackendError(f"claude exited {rc}: {(err or out).strip()[-300:] or 'no output'}")
        return out


# --------------------------------------------------------------------------------------------- agy (ConPTY)

_ANSI = [re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"), re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]"),
         re.compile(r"\x1b[()][0-9A-B]"), re.compile(r"\x1b[=>]")]
PREAMBLE = ("Warning: True color", "Ripgrep is not available", "YOLO mode", "Loaded cached",
            "Tips for getting started", "data is collected")


def strip_ansi(s):
    for rx in _ANSI:
        s = rx.sub("", s)
    return s.replace("\r", "")


def clean_screen(raw):
    lines = [ln for ln in strip_ansi(raw).splitlines() if not any(m in ln for m in PREAMBLE)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip())


def fresh_dir(root, prefix):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    d = root / f"{prefix}_{time.strftime('%H%M%S')}_{os.getpid()}_{threading.get_ident() % 100000}_{int(time.monotonic() * 1000) % 100000}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def agy_pty(exe, instruction, cwd, timeout=600.0, done_file=None, settle=1.5, idle_after=12.0, cancel=None,
            progress=None, model=None, extra=()):
    """Run `agy --print <instruction>` under a ConPTY in cwd. Returns the cleaned screen text.

    Ends when agy exits, when done_file exists and has been stable for `settle` s, or when real (non-preamble)
    output appeared and then stayed silent for idle_after s. Before any real output only the hard `timeout` applies
    (agy can take minutes to start). The instruction is one line with no double quotes (it rides on argv).
    """
    try:
        import winpty
    except ImportError as ex:
        raise BackendError("agy needs the pywinpty package in this venv (pip install pywinpty)") from ex
    line = re.sub(r"\s+", " ", instruction).replace('"', "'").strip()
    exe = str(exe)                                    # winpty takes a bare path; quote only if it has a space
    cmd = (f'"{exe}"' if " " in exe else exe) + f' --print "{line}" --dangerously-skip-permissions --print-timeout {int(timeout)}s'
    if model:
        cmd += f" --model {model}"
    for x in extra:
        cmd += f" {x}"
    try:
        pty = winpty.PtyProcess.spawn(cmd, cwd=str(cwd), dimensions=(50, 250))
    except Exception as ex:
        raise BackendError(f"could not start agy: {ex}") from ex
    buf, stop = [], threading.Event()

    def reader():
        while not stop.is_set():
            try:
                data = pty.read(2048)
            except Exception:
                break
            if data:
                buf.append(data)
            else:
                time.sleep(0.03)

    threading.Thread(target=reader, daemon=True).start()
    start = time.monotonic()
    prev, grew, done_seen, done_size = 0, None, None, -1
    try:
        while True:
            time.sleep(0.1)
            now = time.monotonic()
            if cancel is not None and cancel.is_set():
                raise BackendCancelled("cancelled")
            if not pty.isalive():
                break
            if now - start > timeout:
                raise BackendTimeout(f"agy gave no answer within {timeout:.0f} s")
            n = len(clean_screen("".join(buf)))
            if n > prev:
                prev, grew = n, now
            if progress is not None:
                try:
                    progress(n)
                except Exception:
                    pass
            if done_file is not None:
                try:
                    size = Path(done_file).stat().st_size
                except OSError:
                    size = -1
                if size > 0:
                    if size != done_size:
                        done_size, done_seen = size, now
                    elif now - done_seen > settle:
                        break
            if grew is not None and now - grew > idle_after:
                break
    finally:
        stop.set()
        try:
            if pty.isalive():
                pty.terminate(force=True)
        except Exception:
            pass
    return clean_screen("".join(buf))


class AgyBackend:
    """Antigravity CLI (Gemini). Slow to start; worst case several minutes."""
    name = "agy"
    label = "agy (Gemini)"
    INSTRUCTION = ("Read the file prompt.txt in the current folder and follow it exactly. Do not run commands and do "
                   "not read or change any other file. Write your complete reply, one JSON object and nothing else, "
                   "to the file reply.json in the current folder, then print the same JSON.")

    def __init__(self, work_root, timeout=600.0, model=None, exe=None):
        self.work_root, self.timeout, self.model, self._exe = Path(work_root), float(timeout), model, exe

    def exe(self):
        return self._exe or shutil.which("agy")

    def available(self):
        return bool(self.exe())

    def ask(self, prompt, cancel=None, progress=None):
        if not self.available():
            raise BackendError("the `agy` CLI is not installed (or not on PATH)")
        work = fresh_dir(self.work_root, "agy")
        try:
            (work / "prompt.txt").write_text(prompt, encoding="utf-8")
            reply = work / "reply.json"
            screen = agy_pty(self.exe(), self.INSTRUCTION, work, timeout=self.timeout, done_file=reply, cancel=cancel,
                             progress=progress, model=self.model)
            try:
                text = reply.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                text = ""
            text = text or screen
            if not text.strip():
                raise BackendError("agy produced no output (is it logged in? run `agy` once by hand)")
            return text
        finally:
            shutil.rmtree(work, ignore_errors=True)
