"""Resource governor: keeps the show from bloating until the laptop chokes, without ever turning it off.

It watches frame time (p50 / p95 / long frames), per-pass GPU cost (timer queries, sampled), GPU util / VRAM /
temperature (pynvml if installed, else `nvidia-smi` at a low rate), CPU / RAM (psutil, optional), battery + AC
state, and whether the AI worker is busy. When the system is strained it lowers costs one step at a time,
cheapest-to-notice first, and puts every change back when headroom returns.

Modes (persisted in the controller's live_state.json settings under ext.governor.mode):
  off      telemetry only, never changes anything
  suggest  (default) warns in the panel with one-click fixes: [Apply] [Ignore 10 min] [Always auto]
  auto     applies the fix itself after a 5 s countdown (cancellable from the panel)

Policy (hysteresis, one step at a time, never oscillates):
  strained   frame p95 > 20 ms, GPU > 87 C, nvidia thermal slowdown (throttle bits 0x20/0x40), RAM > 90 %
  critical   frame p95 > 33 ms, VRAM > 3.5 GB (of 4 GB)
  act        after 3 s of sustained strain (and 3 s after the previous action)
  recover    after 15 s of headroom (p95 < 18.5 ms, VRAM < 3.3 GB, < 83 C, RAM < 85 %), restore ONE step (last in,
             first out). A step that brings the strain back within 30 s of being restored sticks for 2, 4, 8 ... min.
  order      1 offscreen render scales (fractal_res, land_res, any other <layer>_res), costliest measured pass first
             2 detail + counts (fractal_detail, being_count, serpent_count)
             3 hold new AI worker jobs (first, when the cause is VRAM / RAM)
             4 render scales to their minimum
  never      blackout, master, layer opacities or triggers. Visible continuous params fade over ~1 s; render scales
             and counts step (scales are snapped to 0.05 by the passes so they do not thrash the allocator).
  user wins  if a governed fader's base value moves (hand, MIDI, scene), that reduction is dropped at once and the
             step is left alone for 60 s.

Battery: on battery the panel shows "Running on battery: X%, ~N min"; below 20 % the low-power profile is suggested
(auto: applied); below 10 % a red warning. Back on AC power restores everything the battery profile changed.

The render thread calls only Governor.frame(vals, base) (< 0.02 ms typical) and the PassTimer start/stop hooks;
sensors run on their own daemon thread. Commands arrive as {"cmd": "gov", "op": ...} (validate_command).
"""
import ctypes
import math
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque

try:
    from params import PARAMS as _PARAMS
except Exception:                                   # pragma: no cover
    _PARAMS = {}

MODES = ("off", "suggest", "auto")

# ---- thresholds
TARGET_MS = 1000.0 / 60.0
STRAINED_P95 = 20.0
CRITICAL_P95 = 33.0
HEADROOM_P95 = 18.5
LONG_FRAME_MS = 33.0
VRAM_CRIT_MB = 3584.0
VRAM_OK_MB = 3300.0
TEMP_STRAINED = 87.0
TEMP_OK = 83.0
RAM_STRAINED = 90.0
RAM_OK = 85.0
THERMAL_BITS = 0x20 | 0x40        # nvidia throttle reasons: SW thermal slowdown | HW thermal slowdown
BATT_LOW = 20.0
BATT_CRIT = 10.0

# ---- timing (seconds)
TICK_S = 0.5
ENTER_S = 3.0
SETTLE_S = 3.0
RECOVER_S = 15.0
COUNTDOWN_S = 5.0
FADE_S = 1.0
CLEAR_S = 5.0            # a SUGGEST banner goes away after this long without strain
REGRESS_S = 30.0         # restored step that is needed again within this -> sticky
STICKY_S = 120.0
USER_COOLDOWN_S = 60.0
SNOOZE_S = 600.0
MIN_SAMPLES = 20
WINDOW = 120             # frames in the p95 window (2 s at 60 fps)
RECENT = 60              # frames in the headroom window

LAYER_NAMES = {"l_fractal": "Fractal", "l_land": "3D landscape", "l_swarm": "Particle swarm", "l_beings": "Beings",
               "l_art": "Art", "serpent": "Serpents", "l_text": "Text", "scene": "Main scene"}

ID_RE = re.compile(r"^[a-z_]+(:[a-z0-9_]+)?$")
OPS = ("mode", "apply", "ignore", "always_auto", "cancel", "restore", "low_power")


# ====================================================================== steps

class Step:
    """One reversible reduction. fn(v) -> reduced value of `param` (None for the AI hold)."""

    def __init__(self, id, cat, param, gate, cost, fn, kind, label, battery=True):
        self.id, self.cat, self.param, self.gate, self.cost = id, cat, param, tuple(gate), cost
        self.fn, self.kind, self.label, self.battery = fn, kind, label, battery

    @property
    def fades(self):
        return self.kind == "fade"


def _snap(v, lo):
    return max(lo, math.floor(v / 0.05 + 0.5) * 0.05)


def build_steps(params=None):
    """The degrade ladder for the params that exist. Steps whose param is missing are simply not built."""
    P = _PARAMS if params is None else params
    steps = []

    def lo(n):
        return float(P[n][0])

    # 1. render scales: known ones + any "<prefix>_res" whose "l_<prefix>" layer exists (art, beings, text ...)
    res = [("fractal_res", ("l_fractal",), "l_fractal"), ("land_res", ("l_land",), "l_land")]   # swarm ignores land_res
    for n in P:
        if n.endswith("_res") and n not in ("fractal_res", "land_res"):
            layer = "l_" + n[:-4]
            if layer in P:
                res.append((n, (layer,), layer))
    for n, gate, cost in res:
        if n in P:
            f = (lambda m: lambda v: _snap(v * 0.7, m))(lo(n))
            steps.append(Step(f"res:{n}", 0, n, gate, cost, f, "res", f"{LAYER_NAMES.get(gate[0], gate[0])} render scale"))
    # 2. detail + counts
    if "fractal_detail" in P:
        m = lo("fractal_detail")
        steps.append(Step("detail:fractal_detail", 1, "fractal_detail", ("l_fractal",), "l_fractal",
                          lambda v, m=m: max(m, v * 0.5), "fade", "Fractal detail"))
    for n, gate, cost, label in (("being_count", ("l_beings",), "l_beings", "Beings count"),
                                 ("serpent_count", ("serpent",), "scene", "Serpent count"),
                                 ("swarm_count", ("l_swarm",), "l_land", "Swarm count")):
        if n in P:
            m = lo(n)
            steps.append(Step(f"count:{n}", 1, n, gate, cost,
                              lambda v, m=m: max(m, float(math.ceil(round(v) / 2.0))), "count", label))
    # 3. AI worker: hold new jobs (the one in flight finishes)
    steps.append(Step("ai:pause", 2, None, (), None, None, "ai", "AI generation paused"))
    # 4. last resort: render scales to their minimum
    for n, gate, cost in res:
        if n in P:
            steps.append(Step(f"floor:{n}", 3, n, gate, cost, (lambda m: lambda v: m)(lo(n)), "res",
                              f"{LAYER_NAMES.get(gate[0], gate[0])} render scale (minimum)", battery=False))
    return steps


# ====================================================================== frame stats + GPU pass timer

class FrameStats:
    def __init__(self, maxlen=WINDOW):
        self._d = deque(maxlen=maxlen)

    @property
    def n(self):
        return len(self._d)

    def add(self, ms):
        self._d.append(ms)

    def clear(self):
        self._d.clear()

    def values(self):
        return list(self._d)

    def _pct(self, q, last=None):
        d = list(self._d)[-last:] if last else list(self._d)
        if not d:
            return None
        d.sort()
        return d[min(len(d) - 1, int(q * len(d)))]

    def p50(self, last=None):
        return self._pct(0.5, last)

    def p95(self, last=None):
        return self._pct(0.95, last)

    def long_frames(self):
        return sum(1 for x in self._d if x > LONG_FRAME_MS)


class PassTimer:
    """GPU time per named section, sampled every `every` frames and read `lag` frames later (so reading the
    result does not stall the pipeline). `factory()` -> a query usable as a context manager with `.elapsed` (ns),
    e.g. lambda: ctx.query(time=True). Sections must not nest: start() ends the open one. Any failure disables it."""

    def __init__(self, factory, every=30, lag=5, sink=None, alpha=0.35):
        self.factory, self.every, self.lag, self.alpha = factory, max(1, int(every)), max(1, int(lag)), alpha
        self.sink = sink if sink is not None else {}
        self.enabled = True
        self._pool, self._used, self._open = {}, [], None
        self._frame, self._sampling, self._sample_frame = 0, False, None

    def begin_frame(self):
        if not self.enabled:
            return
        try:
            self._end()
            self._frame += 1
            if self._sample_frame is not None and self._frame - self._sample_frame >= self.lag:
                self._read()
            self._sampling = self._sample_frame is None and self._frame % self.every == 0
            if self._sampling:
                self._sample_frame, self._used = self._frame, []
        except Exception:
            self.enabled = False

    def start(self, name):
        if not (self.enabled and self._sampling):
            return
        try:
            self._end()
            q = self._pool.get(name)
            if q is None:
                q = self._pool[name] = self.factory()
            q.__enter__()
            self._open = q
            if name not in self._used:
                self._used.append(name)
        except Exception:
            self.enabled, self._open = False, None

    def stop(self):
        if self._open is not None:
            try:
                self._end()
            except Exception:
                self.enabled = False

    def _end(self):
        q, self._open = self._open, None
        if q is not None:
            q.__exit__(None, None, None)

    def _read(self):
        got = {n: self._pool[n].elapsed / 1e6 for n in self._used}
        for n in [n for n in self.sink if n not in got]:
            del self.sink[n]                         # a pass that stopped running costs nothing
        for n, ms in got.items():
            old = self.sink.get(n)
            self.sink[n] = round(ms if old is None else old + (ms - old) * self.alpha, 3)
        self._sample_frame, self._used = None, []


class NullTimer:
    enabled = False

    def begin_frame(self):
        pass

    def start(self, name):
        pass

    def stop(self):
        pass


# ====================================================================== sensors

def _clean(v):
    """Non-finite floats -> None (the panel's /state JSON is encoded with allow_nan=False)."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def parse_smi(text):
    try:
        parts = [p.strip() for p in text.strip().splitlines()[0].split(",")]
        if len(parts) < 4:
            return {}

        def num(s):
            try:
                v = float(s)
            except ValueError:
                return None
            return v if math.isfinite(v) else None
        vals = [num(p) for p in parts[:4]]
        if all(v is None for v in vals):
            return {}
        out = dict(gpu_util=vals[0], vram_mb=vals[1], vram_total_mb=vals[2], gpu_temp=vals[3])
        if len(parts) > 4:                            # clocks_throttle_reasons.active, hex bitmask
            try:
                tr = int(parts[4], 16)
            except ValueError:
                tr = None
            out.update(throttle=tr, thermal=None if tr is None else bool(tr & THERMAL_BITS))
        return out
    except Exception:
        return {}


def _nvml_throttle(pynvml, h):
    try:
        tr = int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h))
        return dict(throttle=tr, thermal=bool(tr & THERMAL_BITS))
    except Exception:
        return {}


def make_gpu_reader():
    """pynvml if installed, else nvidia-smi (hidden window, 4 s timeout), else None."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)

        def nvml():
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            return dict(gpu_util=float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu), vram_mb=m.used / 2 ** 20,
                        vram_total_mb=m.total / 2 ** 20, gpu_temp=float(pynvml.nvmlDeviceGetTemperature(h, 0)),
                        **_nvml_throttle(pynvml, h))
        return nvml
    except Exception:
        pass
    exe = shutil.which("nvidia-smi")
    return make_smi_reader(exe) if exe else None


def make_smi_reader(exe, run=subprocess.run):
    """nvidia-smi reader (hidden window, no stdin, 4 s timeout). The throttle-reason field is dropped only after the
    5-field query failed 3 times in a row WHILE the plain query worked (a driver without that field); until then a
    failed 5-field query falls back to the plain one and reports throttle / thermal as unknown (None)."""
    flags = 0x08000000 if sys.platform == "win32" else 0          # CREATE_NO_WINDOW: never flash a console
    q = "utilization.gpu,memory.used,memory.total,temperature.gpu"
    state = dict(throttle=True, rejects=0)

    def one(extra):
        r = run([exe, "--query-gpu=" + q + extra, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=4, creationflags=flags, stdin=subprocess.DEVNULL)
        return parse_smi(r.stdout or "") if r.returncode == 0 else {}

    def smi():
        if state["throttle"]:
            d = one(",clocks_throttle_reasons.active")
            if d:
                state["rejects"] = 0
                return d
        d = one("")
        if d and state["throttle"]:
            state["rejects"] += 1
            if state["rejects"] >= 3:
                state["throttle"] = False             # this driver does not know the field
        if d:
            d.update(throttle=None, thermal=None)
        return d
    return smi


def battery_from_psutil(b):
    if b is None:
        return {}
    secs = getattr(b, "secsleft", None)
    secs = int(secs) if isinstance(secs, (int, float)) and not isinstance(secs, bool) and secs > 0 else None
    return dict(batt_pct=_clean(float(b.percent)), plugged=bool(b.power_plugged), batt_secs=secs)


def read_battery():
    try:
        import psutil
        return battery_from_psutil(psutil.sensors_battery())
    except Exception:
        pass
    if sys.platform != "win32":
        return {}
    try:
        class SPS(ctypes.Structure):
            _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("pct", ctypes.c_ubyte),
                        ("saver", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
        s = SPS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)) or s.flag == 128 or s.pct == 255:
            return {}
        return dict(batt_pct=float(s.pct), plugged=s.ac == 1 if s.ac in (0, 1) else None,
                    batt_secs=int(s.life) if s.life != 0xFFFFFFFF else None)
    except Exception:
        return {}


def read_host():
    try:
        import psutil
        return dict(cpu=_clean(float(psutil.cpu_percent(None))), ram=_clean(float(psutil.virtual_memory().percent)))
    except Exception:
        return {}


class Sensors:
    """Background sampler. Readers are callables returning dicts. Each reader's latest reading REPLACES its previous
    one (keys it stops reporting disappear) and expires after max(3 x interval, 15 s) without a fresh one, so a
    reader that starts failing can never pin a stale "thermal" / VRAM value. A failing reader backs off
    (interval x 2, 4, 8 ... capped at 30 s); the next poll is scheduled after the read returns."""

    def __init__(self, gpu=None, battery=None, host=None, interval=1.0, gpu_interval=2.0, battery_interval=5.0,
                 clock=time.monotonic):
        self.readers = [(gpu, gpu_interval), (battery, battery_interval), (host, interval)]
        self.interval, self.clock = interval, clock
        self._next = [0.0, 0.0, 0.0]
        self._fails = [0, 0, 0]
        self._last = [None, None, None]               # (time, dict) per reader
        self._lock = threading.Lock()
        self._stop, self._thread = threading.Event(), None

    @classmethod
    def default(cls):
        return cls(gpu=make_gpu_reader(), battery=read_battery, host=read_host)

    def poll_once(self, force=False):
        for i, (fn, every) in enumerate(self.readers):
            if fn is None or (not force and self.clock() < self._next[i]):
                continue
            if self._stop.is_set() and not force:
                return
            try:
                d = fn() or {}
            except Exception:
                d = {}
            now = self.clock()
            if d:
                self._fails[i] = 0
                with self._lock:
                    self._last[i] = (now, {k: _clean(v) for k, v in d.items()})
                self._next[i] = now + every
            else:
                self._fails[i] += 1
                self._next[i] = now + min(30.0, every * 2 ** self._fails[i])

    def snapshot(self):
        now = self.clock()
        out = {}
        with self._lock:
            for i, got in enumerate(self._last):
                if got is not None and now - got[0] <= max(3.0 * self.readers[i][1], 15.0):
                    out.update(got[1])
        return out

    def _run(self):
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.interval)

    def start(self):
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="governor-sensors", daemon=True)
            self._thread.start()

    def stop(self):
        """Never holds up the show's exit: the thread is a daemon, so a hung nvidia-smi is simply abandoned."""
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=1.0)

    def alive(self):
        return self._thread is not None and self._thread.is_alive()


# ====================================================================== commands

def validate_command(c):
    """{"cmd": "gov", "op": ...} -> normalized passthrough dict. Raises ValueError."""
    if not isinstance(c, dict) or c.get("cmd", c.get("type")) != "gov":
        raise ValueError("not a governor command")
    op = c.get("op")
    if op not in OPS:
        raise ValueError("'op' must be one of " + ", ".join(OPS))
    out = {"cmd": "gov", "type": "gov", "op": op}
    if op == "mode":
        if c.get("mode") not in MODES:
            raise ValueError("'mode' must be off, suggest or auto")
        out["mode"] = c["mode"]
    if "id" in c and c["id"] is not None:
        if not isinstance(c["id"], str) or len(c["id"]) > 64 or not ID_RE.match(c["id"]):
            raise ValueError("bad 'id'")
        out["id"] = c["id"]
    elif op in ("apply", "restore"):
        raise ValueError("'id' is required")
    if op == "ignore":
        m = c.get("minutes", 10)
        if not isinstance(m, (int, float)) or isinstance(m, bool) or not math.isfinite(m):
            raise ValueError("bad 'minutes'")
        out["minutes"] = int(min(120, max(1, m)))
    if op == "low_power":
        on = c.get("on", True)
        if not isinstance(on, bool):
            raise ValueError("'on' must be true or false")
        out["on"] = on
    return out


# ====================================================================== governor

def _key(step):
    """Snooze / cooldown key: the param a step changes (res: and floor: share it), or the step id (AI hold)."""
    return step.param or step.id


class _Red:
    __slots__ = ("step", "reasons", "w", "dir", "base_ref", "order", "frm", "to")

    def __init__(self, step, order):
        self.step, self.reasons, self.w, self.dir, self.base_ref, self.order = step, set(), 0.0, 1, None, order
        self.frm = self.to = None


class Governor:
    def __init__(self, sensors=None, clock=time.monotonic, log=print, mode=None, store=None, ai_hold=None,
                 params=None, steps=None, ai_source=None, edit_source=None, panic_source=None):
        self.sensors, self.clock, self._log, self.ai_hold, self.ai_source = sensors, clock, log, ai_hold, ai_source
        # edit_source() -> {param: count of direct moves} (Controller.edits): when given, only a direct move counts as
        # "the user wins"; scene recalls / morphs / xfades / autopilot rewrite the base without dropping a reduction
        # (every reduction is relative to the live value). Without it, any base change counts. panic_source() -> int.
        self.edit_source, self.panic_source, self._panic_n = edit_source, panic_source, None
        self.timer = NullTimer()
        self.params = _PARAMS if params is None else params
        self.steps = steps if steps is not None else build_steps(self.params)
        self._step = {s.id: s for s in self.steps}
        self.store = store if store is not None else {}
        saved = (self.store.get("governor") or {}).get("mode") if isinstance(self.store.get("governor"), dict) else None
        self.mode = mode if mode in MODES else saved if saved in MODES else "suggest"
        self.stats = FrameStats()
        self.pass_ms = {}
        self.ai_status = None
        self.skins_source = None        # materials.SkinEngine.status (set by the engine): resident skins + their VRAM
        self.camera_source = None       # camera_layer.CameraLayer.cost (set by the engine): shown in status(); the camera
                                        # steps its own fps / res down from self.level (camera_layer._watch)
        self.snap = {}
        self._reds, self._n = {}, 0
        self._raw = {}
        now = clock()
        self._last_frame_t = self._last_tick = now
        self.level, self.reasons, self.cause, self.headroom = 0, [], None, False
        self.bad_since = self.good_since = self.calm_since = None
        self.last_action = -1e9
        self.snooze_until = -1e9
        self.step_snooze, self.cooldown, self.restored_at, self.sticky = {}, {}, {}, {}
        self.suggestion = None          # dict(id, kind, text, deadline)
        self.warnings, self.advice = [], ""
        self._adv_t = -1e9
        self.low_power = False
        self._lp_snooze = -1e9
        self._prev_batt = None

    # ------------------------------------------------------------------ render thread
    def frame(self, vals, base=None):
        """Once per frame, after the controller's values are final. Rewrites governed keys of `vals` in place."""
        now = self.clock()
        dt = now - self._last_frame_t
        self._last_frame_t = now
        if 0.0 < dt < 0.25:
            self.stats.add(dt * 1000.0)
        if now - self._last_tick >= TICK_S:
            self._last_tick = now
            try:
                self.tick(vals, now)
            except Exception as ex:                  # policy bugs must never take the picture down
                self.log(f"tick failed: {ex}")
        if not self._reds:
            return vals
        fdt = min(dt, 0.1) if dt > 0 else 0.0
        edits = None
        if self.edit_source is not None:
            try:
                edits = self.edit_source()
            except Exception:
                edits = None
        done = []
        for r in sorted(self._reds.values(), key=lambda r: r.order):
            st = r.step
            if st.fades:
                r.w = min(1.0, max(0.0, r.w + r.dir * fdt / FADE_S))
            else:
                r.w = 1.0 if r.dir > 0 else 0.0
            if r.dir < 0 and r.w <= 0.0:
                done.append(st.id)
                continue
            if st.id not in self._reds:                  # dropped by a user override earlier in this loop
                continue
            p = st.param
            if p is None or p not in vals:
                continue
            if r.dir > 0 and (edits is not None or base is not None):
                b = edits.get(p, 0) if edits is not None else base.get(p)
                if r.base_ref is None:
                    r.base_ref = b
                elif b is not None and abs(b - r.base_ref) > 1e-4:
                    self._user_override(r, now)
                    continue
            v = vals[p]
            t = st.fn(v)
            r.frm, r.to = v, t
            vals[p] = v + r.w * (t - v) if st.fades else (t if r.w >= 0.5 else v)
        for i in done:
            self._reds.pop(i, None)
        return vals

    # ------------------------------------------------------------------ policy (2 Hz)
    def tick(self, vals, now=None):
        now = self.clock() if now is None else now
        self._raw = {k: vals[k] for k in self._watched() if k in vals}
        if self.ai_source is not None:
            try:
                self.ai_status = self.ai_source()
            except Exception:
                self.ai_status = None
        snap = self.sensors.snapshot() if self.sensors is not None else {}
        self.snap = snap
        self._pressure(snap)
        self._battery(now, snap)
        if self.mode == "off":
            self.suggestion = None
            return
        lvl = self.level
        if lvl > 0:
            self.bad_since = self.bad_since if self.bad_since is not None else now
            self.good_since = self.calm_since = None
        else:
            self.bad_since = None
            self.calm_since = self.calm_since if self.calm_since is not None else now
            self.good_since = (self.good_since if self.good_since is not None else now) if self.headroom else None

        if self.panic_source is not None:
            try:
                n = self.panic_source()
            except Exception:
                n = self._panic_n
            if self._panic_n is not None and n != self._panic_n:
                if self.suggestion and self.suggestion["kind"] == "perf":
                    self.log(f"PANIC: dropped the pending {self.suggestion['id']}")
                    self.suggestion = None
                self.bad_since = None
            self._panic_n = n
        for sid in [i for i, (k, until) in self.sticky.items() if now >= until and i not in self._reds
                    and now - self.restored_at.get(i, -1e9) >= 5 * REGRESS_S]:
            del self.sticky[sid]                     # clean for a while: the backoff starts over
        sug = self.suggestion
        if sug and sug["kind"] == "perf" and sug["id"] in self._step and not self._candidate(self._step[sug["id"]], now):
            self.log(f"dropped the {sug['id']} suggestion: nothing to lower there any more")
            self.suggestion = sug = None
        if sug and sug["kind"] == "perf" and lvl == 0:
            if sug["deadline"] is not None or now - self.calm_since >= CLEAR_S:
                if sug["deadline"] is not None:
                    self.log(f"strain cleared before the countdown ended: not applying {sug['id']}")
                self.suggestion = sug = None
        if sug and sug["deadline"] is not None and now >= sug["deadline"]:
            self.suggestion = None
            if sug["id"] == "low_power":
                self.set_low_power(True)
            else:
                self._apply(sug["id"], "perf", now, why=sug.get("why", ""))
            sug = None

        if (sug is None and lvl > 0 and self.bad_since is not None and now - self.bad_since >= ENTER_S and now >= self.snooze_until
                and now - self.last_action >= SETTLE_S):
            step = self._pick(now)
            if step is not None:
                why = "; ".join(self.reasons)
                self.suggestion = dict(id=step.id, kind="perf", text=self._text(step), why=why,
                                       deadline=now + COUNTDOWN_S if self.mode == "auto" else None, t=now)
                self.advice = ""
                self.log(("auto in 5 s: " if self.mode == "auto" else "suggest: ") + self.suggestion["text"] + f" ({why})")
            else:
                adv = ("Nothing left the governor can lower." +
                       (f" The main scene costs {self.pass_ms['scene']:.1f} ms/frame:" if self.pass_ms.get("scene") else "") +
                       " Turn a heavy layer down, or restart at a lower --render size (e.g. 960x600).")
                if not self.advice or now - self._adv_t >= 60.0:     # once, then at most once a minute
                    self._adv_t = now
                    self.log(adv)
                self.advice = adv
        elif lvl == 0:
            self.advice = ""

        if self.good_since is not None and now - self.good_since >= RECOVER_S and self.suggestion is None:
            perf = sorted((r for r in self._reds.values() if "perf" in r.reasons), key=lambda r: r.order)
            for r in reversed(perf):                     # newest first, skipping steps held by the backoff
                if now >= self.sticky.get(r.step.id, (0, -1e9))[1]:
                    self._release(r, "perf")
                    self.restored_at[r.step.id] = now
                    self.log(f"headroom for {RECOVER_S:.0f} s: restored {r.step.id}")
                    self.good_since = now
                    self.last_action = now
                    break

    def _watched(self):
        out = set()
        for s in self.steps:
            if s.param:
                out.add(s.param)
            out.update(s.gate)
        return out

    def _pressure(self, snap):
        lvl, rs, cause = 0, [], None
        p95 = self.stats.p95() if self.stats.n >= MIN_SAMPLES else None
        if p95 is not None and p95 > STRAINED_P95:
            lvl = 2 if p95 > CRITICAL_P95 else 1
            rs.append(f"frame time p95 {p95:.1f} ms")
            cause = "frame"
        vram, temp, ram = snap.get("vram_mb"), snap.get("gpu_temp"), snap.get("ram")
        if vram is not None and vram > VRAM_CRIT_MB:
            lvl = 2
            rs.append(f"VRAM {vram / 1024:.2f} / {(snap.get('vram_total_mb') or 4096) / 1024:.1f} GB")
            cause = "vram"
        if temp is not None and temp > TEMP_STRAINED:
            lvl = max(lvl, 1)
            rs.append(f"GPU hot ({temp:.0f} C)")
            cause = cause or "heat"
        thermal = snap.get("thermal")
        if thermal:
            lvl = max(lvl, 1)
            rs.append(f"GPU thermal slowdown (throttle 0x{snap.get('throttle') or 0:x})")
            cause = cause or "heat"
        if ram is not None and ram > RAM_STRAINED:
            lvl = max(lvl, 1)
            rs.append(f"RAM {ram:.0f}%")
            cause = "ram" if cause in (None, "heat") else cause
        recent = self.stats.p95(RECENT) if self.stats.n >= MIN_SAMPLES else None
        self.headroom = (lvl == 0 and recent is not None and recent < HEADROOM_P95
                         and (vram is None or vram < VRAM_OK_MB) and (temp is None or temp < TEMP_OK)
                         and (ram is None or ram < RAM_OK) and not thermal)
        self.level, self.reasons, self.cause = lvl, rs, cause

    def _effective(self, param):
        v = self._raw.get(param)
        if v is None:
            return None
        for r in sorted(self._reds.values(), key=lambda r: r.order):
            if r.step.param == param and r.dir > 0:
                v = r.step.fn(v)
        return v

    def _candidate(self, s, now, ignore_snooze=False):
        r = self._reds.get(s.id)
        if r is not None and r.dir > 0:
            return False
        if now < self.cooldown.get(_key(s), -1e9):
            return False                             # the user just moved this param
        if not ignore_snooze and now < self.step_snooze.get(_key(s), -1e9):
            return False
        if s.kind == "ai":
            a = self.ai_status or {}
            return bool(a.get("available")) and self.ai_hold is not None
        if s.gate and not any((self._raw.get(g) or 0.0) > 0.001 for g in s.gate):
            return False
        v = self._effective(s.param)
        if v is None:
            return False
        return s.fn(v) < v - (0.5 if s.kind == "count" else 1e-3)

    def _pick(self, now):
        cands = [s for s in self.steps if self._candidate(s, now)]
        if not cands:
            return None
        first_ai = self.cause in ("vram", "ram")

        def key(ix):
            i, s = ix
            cat = -1 if (s.kind == "ai" and first_ai) else s.cat
            return (cat, -(self.pass_ms.get(s.cost, 0.0) if s.cost else 0.0), i)
        return min(((self.steps.index(s), s) for s in cands), key=key)[1]

    def _text(self, s):
        if s.kind == "ai":
            return ("AI generation is competing for the GPU" + (" and VRAM" if self.cause == "vram" else "") +
                    ": hold new AI jobs until there is headroom?")
        name = LAYER_NAMES.get(s.gate[0], s.gate[0]) if s.gate else s.param
        ms = self.pass_ms.get(s.cost) if s.cost else None
        if s.cost == "scene":                         # serpents are drawn inside the main scene pass
            cost = f"{name} (part of the main scene, {ms:.1f} ms/frame)" if ms else name
        else:
            cost = f"{name} layer is costing {ms:.1f} ms/frame" if ms else f"{name} layer"
        v = self._effective(s.param)
        t = s.fn(v) if v is not None else None
        if s.kind == "count":
            return f"{cost}: show {t:.0f} instead of {v:.0f}?"
        if s.kind == "fade":
            return f"{cost}: lower its detail from {v:.2f} to {t:.2f}?"
        if s.cat == 3:
            return f"Still strained. {cost}: drop its render scale to the minimum {t:.2f}?"
        return f"{cost}: drop its resolution from {v:.2f} to {t:.2f}?"

    # ------------------------------------------------------------------ apply / release
    def _apply(self, sid, reason, now=None, why=""):
        s = self._step.get(sid)
        if s is None:
            return False
        now = self.clock() if now is None else now
        r = self._reds.get(sid)
        if r is None:
            self._n += 1
            r = self._reds[sid] = _Red(s, self._n)
        new = r.dir < 0 or not r.reasons
        r.reasons.add(reason)
        r.dir = 1
        if new:
            r.base_ref, r.frm, r.to = None, None, None
            if s.kind == "ai" and self.ai_hold is not None:
                self.ai_hold(True)
            if reason == "perf" and now - self.restored_at.get(sid, -1e9) < REGRESS_S:
                k = self.sticky.get(sid, (0, 0))[0]
                self.sticky[sid] = (k + 1, now + STICKY_S * 2 ** min(k, 4))
                self.log(f"{sid} was needed again right after it was restored: holding it for "
                         f"{STICKY_S * 2 ** min(k, 4) / 60:.0f} min")
            v = self._raw.get(s.param) if s.param else None
            detail = f": {s.param} {v:.2f} -> {s.fn(v):.2f}" if v is not None else ""
            self.log(f"applied {sid} [{reason}]{detail}" + (f" ({why})" if why else ""))
        if reason == "perf":
            self.last_action = now
            self.bad_since = None
            self.stats.clear()
        if self.suggestion and self.suggestion["id"] == sid:
            self.suggestion = None
        return True

    def _release(self, r, reason=None):
        if reason is None:
            r.reasons.clear()
        else:
            r.reasons.discard(reason)
        if r.reasons or r.dir < 0:
            return
        r.dir = -1
        if r.step.kind == "ai" and self.ai_hold is not None:
            self.ai_hold(False)
        if not r.step.fades:
            self._reds.pop(r.step.id, None)

    def _user_override(self, r, now):
        p = r.step.param
        ids = [i for i, x in self._reds.items() if x.step.param == p]
        self.log(f"user moved {p}: dropping {', '.join(ids)}, the user wins")
        for i in ids:
            self._reds.pop(i, None)
        self.cooldown[_key(r.step)] = now + USER_COOLDOWN_S
        if self.suggestion and self.suggestion["id"] in self._step and self._step[self.suggestion["id"]].param == p:
            self.suggestion = None

    def release_all(self, snooze=0.0):
        now = self.clock()
        for r in list(self._reds.values()):
            if r.reasons and snooze:
                self.step_snooze[_key(r.step)] = now + snooze
            self._release(r)
        self.low_power = False

    def set_low_power(self, on):
        now = self.clock()
        if on:
            n = 0
            for s in self.steps:
                if s.battery and (self._candidate(s, now, ignore_snooze=True) or (s.id in self._reds and self._reds[s.id].dir > 0)):
                    self._apply(s.id, "battery", now)
                    n += 1
            self.low_power = True
            self.log(f"low-power profile on ({n} reductions)")
        else:
            for r in list(self._reds.values()):
                if "battery" in r.reasons:
                    self._release(r, "battery")
            if self.low_power:
                self.log("low-power profile off: restoring")
            self.low_power = False
        if self.suggestion and self.suggestion["id"] == "low_power":
            self.suggestion = None

    def _battery(self, now, snap):
        pct, plugged, secs = snap.get("batt_pct"), snap.get("plugged"), snap.get("batt_secs")
        warn = []
        on_batt = plugged is False and pct is not None
        if on_batt:
            mins = f", ~{secs / 60:.0f} min" if isinstance(secs, (int, float)) and secs > 0 else ""
            if pct < BATT_CRIT:
                warn.append(dict(level="critical", text=f"BATTERY {pct:.0f}%{mins}: plug the laptop in now"))
            else:
                warn.append(dict(level="strained" if pct < BATT_LOW else "info", text=f"Running on battery: {pct:.0f}%{mins}"))
            if self._prev_batt is not True:
                self.log(f"running on battery ({pct:.0f}%)")
            if pct < BATT_LOW and self.mode != "off" and not self.low_power and now >= self._lp_snooze:
                if self.suggestion is None or (self.suggestion["kind"] == "perf" and self.suggestion["deadline"] is None):
                    self.suggestion = dict(id="low_power", kind="battery", why=f"battery {pct:.0f}%", t=now,
                                           text=f"Battery {pct:.0f}%: switch to the low-power profile (lower render "
                                                f"scales, fewer beings, AI held)?",
                                           deadline=now + COUNTDOWN_S if self.mode == "auto" else None)
                    self.log(("auto in 5 s: " if self.mode == "auto" else "suggest: ") + self.suggestion["text"])
        else:
            if self._prev_batt is True:
                self.log("back on AC power")
            if self.low_power or any("battery" in r.reasons for r in self._reds.values()):
                self.set_low_power(False)
            if self.suggestion and self.suggestion["id"] == "low_power":
                self.suggestion = None
        self._prev_batt = on_batt if plugged is not None else None
        self.warnings = warn

    # ------------------------------------------------------------------ commands + status
    def set_mode(self, m):
        if m not in MODES:
            return
        if m != self.mode:
            self.log(f"mode {self.mode} -> {m}")
        self.mode = m
        self.store["governor"] = dict(self.store.get("governor") or {}, mode=m)
        if m == "off":
            self.release_all()
            self.suggestion = None

    def command(self, c):
        try:
            c = validate_command(c)
        except ValueError as ex:
            self.log(f"bad command: {ex}")
            return
        op, sid, now = c["op"], c.get("id"), self.clock()
        if op == "mode":
            self.set_mode(c["mode"])
        elif op in ("apply", "always_auto"):
            if op == "always_auto":
                self.set_mode("auto")
            if self.mode == "off":
                self.log("apply ignored: the governor is off")
                return
            sid = sid or (self.suggestion or {}).get("id")
            if sid == "low_power":
                self.set_low_power(True)
            elif sid in self._step and self._candidate(self._step[sid], now, ignore_snooze=True):
                self._apply(sid, "perf", now, why="from the panel")
            elif sid:
                self.log(f"apply {sid} ignored: nothing to lower there now")
                if self.suggestion and self.suggestion["id"] == sid:
                    self.suggestion = None
        elif op in ("ignore", "cancel"):
            mins = c.get("minutes", 10)
            sid = sid or (self.suggestion or {}).get("id")
            if sid == "low_power":
                self._lp_snooze = now + mins * 60
            else:
                self.snooze_until = now + mins * 60
            self.log(f"{'cancelled' if op == 'cancel' else 'ignored'} {sid or 'suggestion'} for {mins} min")
            self.suggestion = None
        elif op == "restore":
            if sid == "all":
                self.release_all(snooze=SNOOZE_S)
                self.snooze_until = now + SNOOZE_S
                self._lp_snooze = now + SNOOZE_S
                self.log("restored everything (panel); quiet for 10 min")
            elif sid == "low_power":
                self.set_low_power(False)
                self._lp_snooze = now + SNOOZE_S
            elif sid in self._reds:
                st = self._reds[sid].step
                for r in [r for r in self._reds.values() if r.step is st or (st.param and r.step.param == st.param)]:
                    self._release(r)
                self.step_snooze[_key(st)] = now + SNOOZE_S
                self.log(f"restored {sid}" + (f" (all of {st.param})" if st.param else "") + " (panel)")
        elif op == "low_power":
            if c["on"] and self.mode == "off":
                self.log("low-power ignored: the governor is off")
                return
            self.set_low_power(c["on"])
            if not c["on"]:
                self._lp_snooze = now + SNOOZE_S

    def log(self, msg):
        try:
            self._log(f"[governor] {msg}")
        except Exception:
            pass

    def status(self):
        now = self.clock()
        p50, p95 = self.stats.p50(), self.stats.p95()
        s = self.snap
        sug = None
        if self.suggestion:
            d = self.suggestion["deadline"]
            sug = dict(id=self.suggestion["id"], text=self.suggestion["text"], why=self.suggestion.get("why", ""),
                       countdown=round(max(0.0, d - now), 1) if d is not None else None)
        act = []
        for r in sorted(self._reds.values(), key=lambda r: r.order):
            if r.dir > 0 and r.reasons:
                act.append(dict(id=r.step.id, label=r.step.label, reasons=sorted(r.reasons), param=r.step.param,
                                frm=None if r.frm is None or not math.isfinite(r.frm) else round(float(r.frm), 3),
                                to=None if r.to is None or not math.isfinite(r.to) else round(float(r.to), 3)))
        r1 = lambda x: None if x is None or not math.isfinite(float(x)) else round(float(x), 1)
        return dict(mode=self.mode, modes=list(MODES), level=("ok", "strained", "critical")[self.level],
                    reasons=list(self.reasons), fps=r1(1000.0 / p50) if p50 else None, p50=r1(p50), p95=r1(p95),
                    long_frames=self.stats.long_frames(), gpu_util=r1(s.get("gpu_util")), vram_mb=r1(s.get("vram_mb")),
                    vram_total_mb=r1(s.get("vram_total_mb")), gpu_temp=r1(s.get("gpu_temp")), cpu=r1(s.get("cpu")),
                    ram=r1(s.get("ram")), batt_pct=r1(s.get("batt_pct")), plugged=s.get("plugged"),
                    batt_secs=s.get("batt_secs"),
                    throttle=None if s.get("throttle") is None else f"0x{s['throttle']:x}", thermal=bool(s.get("thermal")),
                    ai_busy=bool((self.ai_status or {}).get("busy")),
                    passes={k: v for k, v in self.pass_ms.items() if isinstance(v, (int, float)) and math.isfinite(v)}, active=act, suggestion=sug, warnings=list(self.warnings),
                    advice=self.advice, low_power=self.low_power, camera=self._camera(), skins=self._skins(),
                    thresholds=dict(strained_p95=STRAINED_P95, critical_p95=CRITICAL_P95, vram_crit_mb=VRAM_CRIT_MB,
                                    temp_strained=TEMP_STRAINED, ram_strained=RAM_STRAINED))

    def _skins(self):
        try:
            s = self.skins_source() if self.skins_source is not None else None
            return s if isinstance(s, dict) else None
        except Exception:
            return None

    def _camera(self):
        try:
            c = self.camera_source() if self.camera_source is not None else None
            return c if isinstance(c, dict) and c.get("active") else None
        except Exception:
            return None

    def close(self):
        if self.sensors is not None:
            try:
                self.sensors.stop()
            except Exception:
                pass


# ====================================================================== engine wiring

def for_engine(ctx, controller, mode=None, log=print):
    """The governor live_visuals.py runs: real sensors (started), GPU timer queries on `ctx`, the AI hold wired to
    controller.ai, and the mode persisted in controller.ext (live_state.json). `mode` (CLI) overrides the saved one."""
    ai = getattr(controller, "ai", None)
    store = getattr(controller, "ext", None)
    g = Governor(sensors=Sensors.default(), mode=mode, store=store if isinstance(store, dict) else {}, log=log,
                 ai_hold=getattr(ai, "set_hold", None) or ((lambda on: setattr(ai, "hold", bool(on))) if ai is not None else None),
                 ai_source=getattr(ai, "status", None),
                 edit_source=(lambda: controller.edits) if isinstance(getattr(controller, "edits", None), dict) else None,
                 panic_source=(lambda: controller.panics) if hasattr(controller, "panics") else None)
    if ctx is not None:
        g.timer = PassTimer(lambda: ctx.query(time=True), sink=g.pass_ms)
    g.sensors.start()
    names = ("GPU", "battery", "CPU/RAM")
    have = [n for n, (fn, _) in zip(names, g.sensors.readers) if fn is not None]
    g.log(f"mode {g.mode}; sensors: {', '.join(have) or 'none'}; GPU pass timing: {'on' if ctx is not None else 'off'}")
    return g
