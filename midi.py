"""MIDI input for the control system (used by control.Controller).

A background thread opens every MIDI input port (hardware or virtual, e.g. loopMIDI), re-scans every few
seconds for hot-plug, and pushes plain tuples onto a thread-safe queue. The render thread drains them with
poll() inside Controller.update(). Nothing here raises into the caller: with no `mido` / `python-rtmidi`
installed, no device present, or a device unplugged mid-show, the manager just reports nothing.

Messages on the queue:  (port_name, kind, channel, number, value, t)
  kind: "cc" | "note_on" | "note_off" | "clock" | "start" | "stop" | "continue" | "songpos"
  t:    time.perf_counter() at arrival (used for MIDI clock tempo)

Bindings (persisted to midi_map.json):
  {"type": "cc"|"note", "ch": 0..15 or null (any), "num": 0..127, "target": "<target>",
   "port": null or a lowercase substring of the port name, "mode": "abs"|"rel"}
"""
import json
import os
import queue
import threading
import time
from pathlib import Path

CLOCK_KINDS = ("clock", "start", "stop", "continue", "songpos")

# ---------------------------------------------------------------- default maps
# Akai APC mini mk2 (numbers from Akai's "APC mini mk2 Communications Protocol" as mirrored by community
# references; not checked against hardware here):
#   faders 1-8 = CC 48-55, master fader = CC 56 (channel 1)
#   8x8 pads = notes 0-63, bottom-left = 0, left to right, bottom to top
#   track buttons (under the grid) = notes 100-107, scene launch (right column, top to bottom) = notes 112-119
#   shift = note 122
APC = "apc mini"
APC_FADERS = ["zoom", "rotate", "symmetry", "speed", "swim", "hue", "chase", "trails"]   # faders 1-8; 9 = master
APC_TRIGGER_ROW = ["trigger:flash", "trigger:punch", "trigger:strobe", "trigger:invert",
                   "trigger:freeze", "trigger:blackout", "tap", "autopilot"]               # bottom row, notes 0-7
APC_TRACK = ["palette:0", "palette:1", "palette:2", "palette:3", "palette:4", "resync", "next_video", "panic"]
APC_SCENE = ["morph:0", "morph:1", "morph:2", "morph:4", "morph:8", "quantize:beat", "quantize:bar", "quantize:phrase"]

# Novation Launch Control XL (mk1/mk2, factory template; any channel is accepted):
#   send A knobs CC 13-20, send B knobs CC 29-36, pan/device knobs CC 49-56, faders CC 77-84
#   track focus buttons notes 41-44, 57-60; track control buttons notes 73-76, 89-92
LCXL = "launch control"
LCXL_FADERS = ["master", "l_kene", "l_beads", "l_lines", "l_ai", "l_echo", "l_video", "trails"]
LCXL_SEND_A = ["zoom", "pan_x", "pan_y", "rotate", "symmetry", "folds", "speed", "flow"]
LCXL_SEND_B = ["swim", "ripple", "twirl", "bulge", "wave", "density", "flow_dir", "ripple_decay"]
LCXL_PAN = ["palette", "hue", "saturation", "chase", "luma_key", "bpm", None, None]
LCXL_FOCUS = [41, 42, 43, 44, 57, 58, 59, 60]      # -> scenes 1-8
LCXL_CONTROL = [73, 74, 75, 76, 89, 90, 91, 92]    # -> triggers, tap, autopilot


def default_bindings():
    b = []

    def add(typ, num, target, port):
        b.append(dict(type=typ, ch=None, num=num, target=target, port=port, mode="abs"))

    for i, name in enumerate(APC_FADERS):
        add("cc", 48 + i, "param:" + name, APC)
    add("cc", 56, "param:master", APC)
    for i, target in enumerate(APC_TRIGGER_ROW):
        add("note", i, target, APC)
    for row in range(1, 6):                       # rows 2-6 from the bottom: an 8x5 ripple pad
        for col in range(8):
            add("note", row * 8 + col, f"ripple:{(col + 0.5) / 8:.4f}:{(row - 0.5) / 5:.4f}", APC)
    for i in range(8):
        add("note", 56 + i, f"scene:{i}", APC)        # top row    = scenes 1-8
        add("note", 48 + i, f"scene:{8 + i}", APC)    # second row = scenes 9-16
    for i, target in enumerate(APC_TRACK):
        add("note", 100 + i, target, APC)
    for i, target in enumerate(APC_SCENE):
        add("note", 112 + i, target, APC)
    add("note", 122, "shift", APC)                # shift + scene pad = save

    for i in range(8):
        add("cc", 77 + i, "param:" + LCXL_FADERS[i], LCXL)
        add("cc", 13 + i, "param:" + LCXL_SEND_A[i], LCXL)
        add("cc", 29 + i, "param:" + LCXL_SEND_B[i], LCXL)
        if LCXL_PAN[i]:
            add("cc", 49 + i, ("" if LCXL_PAN[i] == "bpm" else "param:") + LCXL_PAN[i], LCXL)
        add("note", LCXL_FOCUS[i], f"scene:{i}", LCXL)
        add("note", LCXL_CONTROL[i], APC_TRIGGER_ROW[i], LCXL)
    return b


def _clean_binding(d, valid_target):
    """Normalized copy of one binding dict, or None if it is malformed."""
    if not isinstance(d, dict) or d.get("type") not in ("cc", "note"):
        return None
    num, ch, port, target = d.get("num"), d.get("ch"), d.get("port"), d.get("target")
    if isinstance(num, bool) or not isinstance(num, int) or not 0 <= num <= 127:
        return None
    if ch is not None and (isinstance(ch, bool) or not isinstance(ch, int) or not 0 <= ch <= 15):
        return None
    if port is not None and not isinstance(port, str):
        return None
    if not isinstance(target, str) or not valid_target(target):
        return None
    return dict(type=d["type"], ch=ch, num=num, target=target, port=port.lower()[:64] if port else None,
                mode="rel" if d.get("mode") == "rel" else "abs")


class MidiManager:
    def __init__(self, map_path=None, scan=True, rescan_s=3.0, valid_target=None, persist=True):
        self.map_path = Path(map_path) if map_path else None
        self.persist = bool(persist)    # False: midi_map.json is read but never written, renamed or quarantined
        self._map_ok = True             # False after a read error: saving is off until a later load works
        self.valid_target = valid_target or (lambda t: True)
        self.q = queue.Queue(maxsize=8192)
        self.available = False
        self.error = "" if scan else "MIDI input is switched off"
        self.ports = []                 # names of the input ports currently open
        self.bindings = default_bindings()
        self.load()
        self._ins, self._outs, self._led = {}, {}, {}
        self._led_want = None           # latest wanted pad colours, picked up by the LED thread
        self._led_wake = threading.Event()
        self._stop = threading.Event()
        self._rescan_s = rescan_s
        self._thread = None
        self._led_thread = None
        if scan:
            self._thread = threading.Thread(target=self._run, name="midi-scan", daemon=True)
            self._thread.start()
            self._led_thread = threading.Thread(target=self._led_run, name="midi-led", daemon=True)
            self._led_thread.start()

    # ------------------------------------------------------------ queue
    def inject(self, kind, ch=0, num=0, val=0, port="test", t=None):
        """Push a message as if a device had sent it (tests, and the fake injector)."""
        try:
            self.q.put_nowait((port, kind, int(ch), int(num), int(val), time.perf_counter() if t is None else t))
        except Exception:
            pass

    def poll(self):
        out = []
        try:
            while True:
                out.append(self.q.get_nowait())
        except queue.Empty:
            pass
        except Exception:
            pass
        return out

    def _cb(self, port, msg):
        try:
            t, typ = time.perf_counter(), msg.type
            if typ == "control_change":
                item = (port, "cc", msg.channel, msg.control, msg.value, t)
            elif typ == "note_on":
                item = (port, "note_on" if msg.velocity > 0 else "note_off", msg.channel, msg.note, msg.velocity, t)
            elif typ == "note_off":
                item = (port, "note_off", msg.channel, msg.note, 0, t)
            elif typ in ("clock", "start", "stop", "continue"):
                item = (port, typ, 0, 0, 0, t)
            elif typ == "songpos":
                item = (port, "songpos", 0, 0, msg.pos, t)
            else:
                return
            self.q.put_nowait(item)
        except Exception:
            pass                        # queue full or an odd message: drop it

    # ------------------------------------------------------------ ports (scan thread)
    def _run(self):
        try:
            import mido
            mido.get_input_names()      # forces the rtmidi backend to load
            self.available = True
        except Exception as ex:
            self.error = f"{type(ex).__name__}: {ex}"[:160]
            return
        while not self._stop.is_set():
            try:
                self._scan(mido)
            except Exception as ex:
                self.error = f"{type(ex).__name__}: {ex}"[:160]
            self._stop.wait(self._rescan_s)
        self._close_ports()

    def _scan(self, mido):
        names = list(dict.fromkeys(mido.get_input_names()))
        for name in [n for n in self._ins if n not in names]:      # unplugged
            self._safe_close(self._ins.pop(name, None))
        for name in names:
            if name not in self._ins:
                try:
                    self._ins[name] = mido.open_input(name, callback=lambda m, _n=name: self._cb(_n, m))
                except Exception:
                    pass                # busy (another app holds it) or vanished: retry next scan
        self.ports = list(self._ins)
        want = [n for n in dict.fromkeys(mido.get_output_names()) if APC in n.lower()]
        outs, fresh = dict(self._outs), False
        gone = [outs.pop(n) for n in list(outs) if n not in want]
        for name in want:
            if name not in outs:
                try:
                    outs[name] = mido.open_output(name)     # slow driver call: no lock is held here
                    fresh = True
                except Exception:
                    pass
        self._outs = outs               # one atomic swap; the LED thread works from whichever dict it grabbed
        if fresh:
            self._led = {}              # repaint a freshly plugged controller
            self._led_wake.set()
        for port in gone:
            self._safe_close(port)

    @staticmethod
    def _safe_close(port):
        try:
            if port is not None:
                port.close()
        except Exception:
            pass

    def _close_ports(self):
        ins, self._ins, self.ports = self._ins, {}, []
        outs, self._outs = self._outs, {}
        for port in list(ins.values()) + list(outs.values()):
            self._safe_close(port)

    def close(self):
        self._stop.set()
        self._led_wake.set()
        for attr in ("_thread", "_led_thread"):
            t = getattr(self, attr)
            setattr(self, attr, None)
            if t is not None:
                t.join(timeout=2)

    # ------------------------------------------------------------ pad LEDs (APC mini mk2, best effort)
    def leds(self, wanted):
        """wanted: {note: colour velocity}. Called from the render thread: it only hands the dict over.
        The MIDI output itself happens on the LED thread, so a slow or unplugged device never stalls a frame."""
        self._led_want = wanted
        self._led_wake.set()

    def _led_run(self):
        while not self._stop.is_set():
            self._led_wake.wait(1.0)
            self._led_wake.clear()
            want, outs = self._led_want, list(self._outs.values())
            if not want or not outs:
                continue
            try:
                import mido
                sent = self._led
                for note, vel in want.items():
                    if sent.get(note) != vel:
                        for port in outs:       # pads take channel 7 (full brightness), the round buttons channel 1
                            port.send(mido.Message("note_on", channel=6 if note < 64 else 0, note=note, velocity=vel))
                        sent[note] = vel
            except Exception:
                self._led = {}                  # unplugged mid-send: start clean when it comes back

    # ------------------------------------------------------------ bindings
    def lookup(self, port, typ, ch, num):
        pl = (port or "").lower()
        return [b for b in self.bindings
                if b["type"] == typ and b["num"] == num and (b["ch"] is None or b["ch"] == ch)
                and (b["port"] is None or b["port"] in pl)]

    def bind(self, typ, ch, num, target, port=None):
        """Bind one physical control to `target`, replacing whatever that control did before."""
        if typ not in ("cc", "note") or not self.valid_target(target):
            return False
        self._retry_load()
        old = self.lookup(port, typ, ch, num)
        self.bindings = [b for b in self.bindings if not any(b is o for o in old)]
        self.bindings.append(dict(type=typ, ch=ch, num=num, target=target, port=None, mode="abs"))
        self.save()
        return True

    def clear(self, target=None):
        self._retry_load()
        self.bindings = [] if target is None else [b for b in self.bindings if b["target"] != target]
        self.save()

    def reset_defaults(self):
        self.bindings = default_bindings()
        self.save()

    def targets(self):
        """{target: ["cc 48", "note 3 ch1", ...]} for the panel."""
        out = {}
        for b in self.bindings:
            if b["target"].startswith("ripple:"):
                continue                # the 40 ripple pads would bloat the polled state
            label = f"{b['type']} {b['num']}" + (f" ch{b['ch'] + 1}" if b["ch"] is not None else "")
            out.setdefault(b["target"], []).append(label)
        return out

    def load(self):
        """Bindings from midi_map.json. Never raises. Returns True if the file was loaded.

        A missing file leaves the defaults. An unreadable file (OSError) also leaves the defaults but
        switches saving off until a later load works, so a transient error cannot replace the user's map.
        Only a file that reads fine and is not valid is moved aside (midi_map.json.<time>.bad).
        """
        self._map_ok = True
        if self.map_path is None:
            return False
        try:
            text = self.map_path.read_text(encoding="utf-8") if self.map_path.exists() else None
        except OSError:
            self._map_ok = False
            self.bindings = default_bindings()
            return False
        if text is None:
            return False
        try:
            raw = json.loads(text)["bindings"]
            if not isinstance(raw, list):
                raise ValueError("bindings must be a list")
        except (ValueError, KeyError, TypeError, RecursionError):
            if self.persist:
                quarantine(self.map_path)
            self.bindings = default_bindings()
            return False
        self.bindings = [c for c in (_clean_binding(d, self.valid_target) for d in raw[:2048]) if c]
        return True

    def _retry_load(self):
        if not self._map_ok:
            self.load()

    def save(self):
        if self.map_path is None or not self.persist or not self._map_ok:
            return False
        return atomic_write_json(self.map_path, dict(version=1, bindings=self.bindings))


def quarantine(path):
    """Move a corrupt state file aside under a unique name (kept for a post-mortem). Never raises."""
    try:
        path = Path(path)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for n in range(100):
            bad = path.with_name(f"{path.name}.{stamp}{'-%d' % n if n else ''}.bad")
            if not bad.exists():
                os.replace(path, bad)
                return bad
    except OSError:
        pass
    return None


def atomic_write_json(path, obj):
    """Write JSON via a temp file + os.replace so a crash never leaves a half-written file. Never raises."""
    tmp = None
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(obj, allow_nan=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if tmp is not None and tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False
