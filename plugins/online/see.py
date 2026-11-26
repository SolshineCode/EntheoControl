"""SEE THE OUTPUT: the assistant may look at ONE small picture of the projector output with a message, and (opt-in) run a
LOOK-AND-ADJUST loop: propose -> (performer applies / AUTOPILOT applies) -> the morph settles -> a new picture ->
the assistant compares before / after and refines, at most LOOP_MAX model calls.

Where the picture comes from: the engine (live_visuals, `[online-see>` block) grabs the guarded program output (after
filters and the flash guard, before mapping) with its own 640x400 fx_stack.PreviewTap: one small blit plus an
ASYNCHRONOUS readback (no GPU stall); the JPEG is encoded on a worker thread and handed to `protocol.snapshot_in`.
Nothing is captured unless the assistant asked for it, and only while ONLINE is on.

Privacy: pictures live in memory only (SnapStore, the last SNAP_KEEP), are served to the panel so the performer sees
the exact bytes that are sent, and are dropped when ONLINE goes off. Backends send them inline (Claude Code: an image
block on stdin; Gemini API: an inline image part) or copy them into agy's throw-away sandbox folder under the system
temp folder (deleted after the call). Nothing here writes a picture to disk.

People: the output can contain the crowd camera (or a video loop with people in it). People cannot be greyed out of
the OUTPUT the way AUTO MAP does it for the raw camera picture (the camera is composited through effects, warps and
blends, so a camera-space person mask does not line up with it), so instead a picture taken while such a layer is
visible is held back until the performer confirms it (`people_risk`).
"""
import collections
import threading

SNAP_SIZE = (640, 400)       # what the engine captures (PreviewTap size)
SNAP_KEEP = 12               # pictures kept in memory for the panel
SNAP_MAX_BYTES = 1 << 20     # a 640x400 JPEG is ~40-90 KB; anything bigger is not ours
SNAP_TIMEOUT_S = 4.0         # no frame within this: the engine is not capturing (headless without a tap, no panel)
LOOP_DEFAULT = 3
LOOP_MAX = 6
SETTLE_S = 1.5               # after a batch is applied and its morph / fades are over, wait this long before looking
SETTLE_MAX_S = 30.0          # look anyway after this long (a very long morph)
APPLY_WAIT_S = 180.0         # SUGGEST: a loop proposal nobody applies within this ends the loop
RATE_WAIT_S = 75.0           # a loop step held back by the request limit gives up after this
PEOPLE_MIN = 0.02            # a layer amount above this counts as visible
PEOPLE_LAYERS = (("l_camera", "the crowd camera layer is visible"),
                 ("l_video", "a video loop is visible (it may show people)"))
PEOPLE_FX = ("crowd_push", "crowd_swarm", "crowd_beings", "crowd_ripples", "crowd_serpent")


def is_jpeg(b):
    return isinstance(b, (bytes, bytearray)) and 100 < len(b) <= SNAP_MAX_BYTES and b[:3] == b"\xff\xd8\xff"


class SnapStore:
    """id -> JPEG bytes, in memory only, the newest SNAP_KEEP. Read by the HTTP thread, written by the GL thread."""

    def __init__(self, keep=SNAP_KEEP):
        self.keep = int(keep)
        self._d = collections.OrderedDict()
        self._lock = threading.Lock()
        self._n = 0

    def put(self, jpg):
        with self._lock:
            self._n += 1
            self._d[self._n] = bytes(jpg)
            while len(self._d) > self.keep:
                self._d.popitem(last=False)
            return self._n

    def get(self, i):
        with self._lock:
            return self._d.get(i)

    def drop(self, i):
        with self._lock:
            self._d.pop(i, None)

    def clear(self):
        with self._lock:
            self._d.clear()

    def __len__(self):
        return len(self._d)


def people_risk(values, bypass=()):
    """Reasons why people could be in the output picture right now ([] = none). Uses the OUTPUT values (after mods and
    bypass fades), so a layer that is up but switched off does not count, and one brought in by a modulator does."""
    out = []
    for name, why in PEOPLE_LAYERS:
        try:
            if float(values.get(name, 0.0)) > PEOPLE_MIN:
                out.append(why)
        except (TypeError, ValueError):
            out.append(why)
    try:
        fx = [n for n in PEOPLE_FX if float(values.get(n, 0.0)) > PEOPLE_MIN]
    except (TypeError, ValueError):
        fx = list(PEOPLE_FX)
    if fx:
        out.append("crowd camera effects are on (" + ", ".join(fx) + "): they draw people's silhouettes")
    return out


def touch_sig(ctl):
    """What only the performer changes: direct fader moves (`edits`), the bypass set, PANICs. A change between two
    loop steps = the performer took over, and the loop stops."""
    try:
        edits = sum(int(v) for v in (getattr(ctl, "edits", None) or {}).values())
    except Exception:
        edits = 0
    return (edits, frozenset(getattr(ctl, "bypass", ()) or ()), int(getattr(ctl, "panics", 0) or 0))


SEE_ONE = ("PICTURE: the file output.jpg is a {w}x{h} picture of the projector output RIGHT NOW (after filters and the "
           "flash guard, before the projection mapping). Look at it and use it to judge what to change; describe briefly "
           "in \"say\" what you see and why you change what you change.")
SEE_LOOP = ("LOOK-AND-ADJUST step {step} of {n}. The performer's wish is above (PERFORMER). before.jpg is the output "
            "before your previous changes, output.jpg is the output now, after they were applied and the morph "
            "settled ({w}x{h}, after filters and the flash guard, before mapping). Your previous changes: {prev}. "
            "Compare the two pictures. In \"say\" write in one or two sentences what changed and what you refine now. "
            "If the look already matches the wish, answer with \"commands\": [] and say so: that ends the loop.")
SEE_FIRST_OF_LOOP = (" This is LOOK-AND-ADJUST step 1 of {n}: after your changes are applied you will get a new picture "
                     "and can refine.")


def see_text(step=1, n=1, prev="", size=SNAP_SIZE):
    w, h = size
    if step <= 1:
        return SEE_ONE.format(w=w, h=h) + (SEE_FIRST_OF_LOOP.format(n=n) if n > 1 else "")
    return SEE_LOOP.format(step=int(step), n=int(n), prev=(prev or "none")[:400], w=w, h=h)
