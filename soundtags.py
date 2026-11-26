"""SOUND TAGS: recognise a musical theme from the loopback stream and say how close the music is to a tagged one.

Pure numpy, no GL, no I/O, no audio device. The engine hangs one tap on the existing capture thread
(live_visuals.Audio.taps: fn(block (n, ch) float32, t)), so there is still exactly ONE loopback stream.

What is computed (never the audio itself, only FEATURES):
* every 512 samples (93.75 Hz at 48 kHz): log energy flux in 3 bands of a 512-point FFT -> onset envelope.
* every 4096 samples (one FRAME, 85 ms, 11.7 frames / s), from a 4096-point FFT of that hop:
    chroma (12-bin pitch-class profile, 100 Hz - 5 kHz), MFCC 1-12 (26 mel bands 60 Hz - 12 kHz, c0 dropped),
    spectral centroid, roll-off (85 %), flatness, band energy ratios (bass / mid / high), and from the onset
    envelope: onset rate, tempo (autocorrelation, octave-folded into 80-160 BPM) and beat strength.
    Every one of these is LEVEL-INVARIANT (ratios, a floor relative to the frame's own peak, c0 dropped), so a tag
    matches the same music played louder or quieter. Loudness (dBFS) is kept separately, only for the silence gate.
* every MATCH_DT (0.25 s, i.e. 4 Hz) while matching is active and at least one template is loaded: the similarity
  of the last WINDOW_S (4 s) of music to every template -> `self.sims` {id: 0..1} (replaced whole: atomic).

Similarity (`similarity()`):
* "vibe" = timbre / energy distribution: the window's mean feature vector against an example's mean, each dim
  scaled by sqrt(example sd^2 + SCALE^2); sim = exp(-mean(z^2) / 2). Key-free by construction (no chroma in it).
* "section" = same song section: the window's chroma SEQUENCE (bins of 4 frames = 0.34 s, mean-removed, i.e.
  Pearson per bin) slid along the example's chroma sequence; each window bin may match the example bin +-1 bin
  (tempo / alignment tolerance); best diagonal mean. With key_inv the example is also tried in all 12 transpositions.
  section sim = chroma^0.7 * vibe^0.3 (the chords AND roughly the sound).
* more than half of the window silent (< SILENCE_DB) = 0.

A template example is compact JSON (`make_example`): {"secs", "mu": [21], "sd": [21], "ch": [[12] x <= 44 bins]}
(about 3 kB). Several examples per tag: the best one counts.

`Decider` is the pure, deterministic trigger logic (threshold from sensitivity, min hold, cooldown, re-arm
hysteresis, global min gap, priority, blocks); `suggest()` clusters recent audio into recurring sections.

Thread model: `feed` runs on the audio thread; `sims`, `count`, `frames_since`, `set_templates`, `active` are read
/ written from the GL thread. The frame ring is guarded by a lock that is only held for a few microseconds.
"""
import math
import threading
import time

import numpy as np

SR = 48000
CHUNK = 512                     # onset envelope hop
N = 4096                        # frame FFT size == frame hop (no overlap)
ONSET_HZ = SR / CHUNK           # 93.75
FRAME_HZ = SR / N               # 11.72
HISTORY_S = 600.0               # frames kept for tagging / suggestions (10 min = 7032 frames x 34 x 4 B = 0.96 MB)
WINDOW_S = 4.0                  # the music that is compared with a template
MATCH_DT = 0.25                 # s between two similarity rounds (4 Hz)
MATCH_BUDGET = 0.006            # matching may use at most 0.6 % of a core (8 tags x 4 examples x 12 keys: ~2.2 Hz)
SILENCE_DB = -55.0              # a frame quieter than this (RMS dBFS) is silence
BIN = 4                         # frames per chroma bin (0.34 s)
MAX_EX_BINS = 44                # an example keeps at most 44 chroma bins (15 s)
MIN_EX_S = 3.0                  # an example needs at least 3 s of non-silent music
MAX_EXAMPLES = 4

# frame layout
CH = slice(0, 12)               # chroma (sums to 1)
MF = slice(12, 24)              # mfcc 1..12
CENT, ROLL, FLAT = 24, 25, 26
BASS, MID, HIGH = 27, 28, 29
ONS, TEMPO, BEAT = 30, 31, 32
D = 33
VIBE = np.r_[12:33]             # 21 dims: everything but chroma
NV = len(VIBE)
# expected spread of each vibe dim across different music (the floor of the per-dim scale)
SCALE = np.array([0.35] * 12 + [0.06, 0.06, 0.08] + [0.06, 0.06, 0.06] + [0.08, 0.12, 0.15], dtype=np.float64)

MODES = ("vibe", "section")


# ------------------------------------------------------------------ precomputed tables

def _mel(n_fft, sr, n_mels=26, lo=60.0, hi=12000.0):
    hz = np.fft.rfftfreq(n_fft, 1.0 / sr)
    m = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)                                    # noqa: E731
    pts = 700.0 * (10 ** (np.linspace(m(lo), m(hi), n_mels + 2) / 2595.0) - 1.0)
    fb = np.zeros((n_mels, len(hz)))
    for i in range(n_mels):
        a, b, c = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((hz - a) / (b - a), (c - hz) / (c - b)), 0.0, None)
    return fb


def _dct(n_out, n_in):
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    d = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in)) * math.sqrt(2.0 / n_in)
    d[0] /= math.sqrt(2.0)
    return d


def _chroma(n_fft, sr, lo=100.0, hi=5000.0):
    hz = np.fft.rfftfreq(n_fft, 1.0 / sr)
    cm = np.zeros((12, len(hz)))
    for j, f in enumerate(hz):
        if lo <= f <= hi:
            p = 12.0 * math.log2(f / 440.0) + 69.0                  # fractional MIDI note
            a = math.floor(p)
            w = p - a
            cm[a % 12, j] += 1.0 - w                                # split between the two nearest pitch classes
            cm[(a + 1) % 12, j] += w
    return cm


class _Tables:
    def __init__(self, sr):
        self.win = np.hanning(N)
        self.win_c = np.hanning(CHUNK)
        self.hz = np.fft.rfftfreq(N, 1.0 / sr)
        self.mel = _mel(N, sr)
        self.dct = _dct(13, 26)[1:]                                  # c1..c12 (c0 = level: dropped)
        self.chroma = _chroma(N, sr)
        hz = self.hz
        ix = lambda f: int(np.searchsorted(hz, f))                                         # noqa: E731
        self.bedges = np.array([ix(20), ix(250), ix(4000), ix(16000)])                    # bass / mid / high
        self.spec = slice(ix(20), ix(16000))
        self.flat = slice(ix(60), ix(12000))
        self.hz_spec = hz[self.spec]
        hc = np.fft.rfftfreq(CHUNK, 1.0 / sr)
        self.cedges = np.array([1, int(np.searchsorted(hc, 200.0)), int(np.searchsorted(hc, 4000.0)), len(hc) - 1])
        nz = np.nonzero(self.chroma.any(axis=0))[0]          # the chroma / mel tables are sparse: use their spans
        self.c_lo, self.c_hi = int(nz[0]), int(nz[-1]) + 1
        self.chroma_s = self.chroma[:, self.c_lo:self.c_hi]
        nz = np.nonzero(self.mel.any(axis=0))[0]
        self.m_lo, self.m_hi = int(nz[0]), int(nz[-1]) + 1
        self.mel_s = self.mel[:, self.m_lo:self.m_hi]


_TABLES = {}


def tables(sr=SR):
    t = _TABLES.get(sr)
    if t is None:
        t = _TABLES[sr] = _Tables(sr)
    return t


# ------------------------------------------------------------------ the engine


class SoundTags:
    """Feature extraction + continuous matching. feed() on the audio thread; everything else on any thread."""

    def __init__(self, sr=SR, history_s=HISTORY_S, clock=time.monotonic):
        self.sr = int(sr)
        self.tb = tables(self.sr)
        self.clock = clock
        self.maxf = max(64, int(history_s * self.sr / N))
        self._ring = np.zeros((self.maxf, D), np.float32)
        self._loud = np.full(self.maxf, -120.0, np.float32)
        self._lock = threading.Lock()
        self.count = 0                                  # frames produced so far (monotonic)
        self._tail = np.zeros(0, np.float32)
        self._fbuf = np.zeros(N, np.float64)
        self._fpos = 0
        self._env = np.zeros(1024)                      # onset envelope ring (10.9 s at 93.75 Hz)
        self._env_n = 0
        self._prev_e = None
        self._onset = (0.0, 0.5, 0.0)                   # onset rate, tempo, beat strength (latest)
        self._nfeat = 0
        self.templates = {}                             # id -> compiled template (replaced whole by set_templates)
        self.sims = {}                                  # id -> similarity of the last round (replaced whole)
        self.sims_t = 0.0                               # frame count of the last round (deterministic clock)
        self._match_at = 0
        self.active = False                             # the GL thread switches matching on (armed / learn mode)
        self.match_dt = MATCH_DT
        self.cost_s = 0.0                               # CPU seconds spent in feed (tests / the panel)
        self.blocks = 0
        self.last_t = None                              # clock() of the last feed (is audio arriving?)

    # ---- audio thread
    def feed(self, block, t=None):
        t0 = time.perf_counter()
        x = np.asarray(block, dtype=np.float32)
        if x.ndim == 2:
            x = (x[:, 0] + x[:, 1]) * 0.5 if x.shape[1] == 2 else (x.mean(axis=1) if x.shape[1] > 1 else x[:, 0])
        x = x.ravel()
        if not math.isfinite(float(x.sum())):        # a device glitch: one reduce in the normal case
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        tail = np.concatenate((self._tail, x)) if len(self._tail) else x
        k = 0
        while len(tail) - k >= CHUNK:
            ch = tail[k:k + CHUNK]
            k += CHUNK
            self._onset_step(ch)
            self._fbuf[self._fpos:self._fpos + CHUNK] = ch
            self._fpos += CHUNK
            if self._fpos == N:
                self._fpos = 0
                self._frame(self._fbuf)
        self._tail = tail[k:].copy()
        self.blocks += 1
        self.last_t = self.clock()
        self.cost_s += time.perf_counter() - t0

    def _onset_step(self, ch):
        f = np.fft.rfft(ch * self.tb.win_c)
        mag2 = f.real * f.real + f.imag * f.imag
        b = np.add.reduceat(mag2, self.tb.cedges)[:3]
        floor = float(b.sum()) * 1e-4 + 1e-12           # relative floor: the flux does not change with the level
        e = np.log(b + floor)
        if self._prev_e is None:
            flux = 0.0
        else:
            flux = min(20.0, float(np.maximum(e - self._prev_e, 0.0).sum()))   # a hit out of silence: capped
        loud = 10.0 * math.log10(float(np.dot(ch, ch)) / len(ch) + 1e-12)
        if loud < SILENCE_DB:
            flux = 0.0
        self._prev_e = e
        self._env[self._env_n % len(self._env)] = flux
        self._env_n += 1

    def _onset_feats(self):
        n = min(self._env_n, len(self._env))
        if n < 64:
            return 0.0, 0.5, 0.0
        idx = (np.arange(self._env_n - n, self._env_n)) % len(self._env)
        env = self._env[idx]
        # onset rate over the last 3 s: peaks above an adaptive threshold
        w = env[-int(3 * ONSET_HZ):]
        thr = max(0.6, float(np.median(w)) + 0.6)
        pk = (w[1:-1] > thr) & (w[1:-1] >= w[:-2]) & (w[1:-1] > w[2:])
        rate = float(pk.sum()) / (len(w) / ONSET_HZ)
        # tempo: autocorrelation of the last 512 values (5.5 s)
        a = env[-512:] - env[-512:].mean()
        f = np.fft.rfft(a, 1024)
        ac = np.fft.irfft(f * np.conj(f))[:200]
        if ac[0] <= 1e-9:
            return min(rate, 10.0) / 10.0, 0.5, 0.0
        lo, hi = int(60 * ONSET_HZ / 200), int(60 * ONSET_HZ / 60) + 1      # 60..200 BPM
        lag = lo + int(np.argmax(ac[lo:hi]))
        strength = max(0.0, min(1.0, float(ac[lag] / ac[0])))
        bpm = 60.0 * ONSET_HZ / lag
        while bpm < 80.0:
            bpm *= 2.0
        while bpm >= 160.0:
            bpm /= 2.0
        return min(rate, 10.0) / 10.0, math.log2(bpm / 80.0), strength

    def _frame(self, x):
        tb = self.tb
        ms = float(np.mean(x * x))
        loud = 10.0 * math.log10(ms + 1e-12)
        mag = np.abs(np.fft.rfft(x * tb.win))
        p = mag * mag
        v = np.zeros(D, np.float32)
        if loud >= SILENCE_DB and p.sum() > 0:
            c = tb.chroma_s @ mag[tb.c_lo:tb.c_hi]
            v[CH] = c / max(c.sum(), 1e-12)
            mel = tb.mel_s @ p[tb.m_lo:tb.m_hi]
            v[MF] = (tb.dct @ np.log10(mel + mel.max() * 1e-4 + 1e-20)) / 4.0
            ps = p[tb.spec]
            hz = tb.hz_spec
            tot = ps.sum() + 1e-20
            cen = float((hz * ps).sum() / tot)
            v[CENT] = math.log2(max(cen, 50.0) / 50.0) / 9.0
            cs = np.cumsum(ps)
            ro = float(hz[min(len(hz) - 1, int(np.searchsorted(cs, 0.85 * cs[-1])))])
            v[ROLL] = math.log2(max(ro, 50.0) / 50.0) / 9.0
            pf = p[tb.flat]
            pf = pf + pf.max() * 1e-6 + 1e-20
            fl = math.exp(float(np.mean(np.log(pf)))) / float(np.mean(pf))
            v[FLAT] = min(1.0, max(0.0, (math.log10(max(fl, 1e-6)) + 6.0) / 6.0))
            b = np.sqrt(np.add.reduceat(p, tb.bedges)[:3])
            v[BASS:HIGH + 1] = b / max(b.sum(), 1e-20)
            self._nfeat += 1
            if self._nfeat % 3 == 1:                     # rhythm moves slowly: refreshed every 3rd frame (0.26 s)
                self._onset = self._onset_feats()
            v[ONS], v[TEMPO], v[BEAT] = self._onset
        with self._lock:
            i = self.count % self.maxf
            self._ring[i] = v
            self._loud[i] = loud
            self.count += 1
        if self.active and self.templates and self.count >= self._match_at:
            t0 = time.perf_counter()
            self.match()
            cost = time.perf_counter() - t0
            # 4 Hz, slowed down when many tags make a round expensive: matching stays under MATCH_BUDGET of a core
            self.match_dt = max(MATCH_DT, cost / MATCH_BUDGET)
            self._match_at = self.count + max(1, int(math.ceil(self.match_dt * FRAME_HZ - 1e-9)))

    # ---- any thread
    def frames_since(self, start, end=None):
        """(frames (n, D) float64, loudness (n,)) for frame indices [start, end). Older than the ring = cut."""
        with self._lock:
            end = self.count if end is None else min(end, self.count)
            start = max(start, end - self.maxf, 0)
            if end <= start:
                return np.zeros((0, D)), np.zeros(0)
            idx = np.arange(start, end) % self.maxf
            return self._ring[idx].astype(np.float64), self._loud[idx].astype(np.float64)

    def recent(self, seconds):
        n = int(round(seconds * FRAME_HZ))
        return self.frames_since(self.count - n)

    def set_templates(self, compiled):
        """{id: compile_tag(...)} (replaced whole: the audio thread never sees a half-built dict)."""
        batch = build_batch(compiled)
        self.templates = dict(compiled)
        self._batch = batch
        self.sims = {k: v for k, v in self.sims.items() if k in self.templates}
        self._match_at = self.count

    def match(self):
        """One similarity round over every template (audio thread while active; tests call it directly)."""
        tpl = self.templates
        if not tpl:
            self.sims = {}
            return {}
        fr, loud = self.recent(WINDOW_S)
        win = window(fr, loud)
        batch = getattr(self, "_batch", None)
        if batch is None or batch["ids"] != tuple(tpl):
            batch = self._batch = build_batch(tpl)
        out = batch_sims(win, batch)
        self.sims = out
        self.sims_t = self.count
        return out

    def receiving(self, now=None, within=2.0):
        now = self.clock() if now is None else now
        return self.last_t is not None and now - self.last_t <= within


# ------------------------------------------------------------------ matching


def _bins(ch):
    """Chroma frames -> bins of BIN frames (mean), mean-removed and L2-normalised (Pearson per bin)."""
    n = len(ch) // BIN
    if n == 0:
        return np.zeros((0, 12))
    b = ch[:n * BIN].reshape(n, BIN, 12).mean(axis=1)
    b = b - b.mean(axis=1, keepdims=True)
    return b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)


def window(frames, loud):
    """The music to compare: None when it is mostly silence, else its vibe mean and chroma bins."""
    if len(frames) == 0:
        return None
    on = loud >= SILENCE_DB
    if on.mean() < 0.5 or on.sum() < 8:
        return None
    f = frames[on]
    return dict(mu=f[:, VIBE].mean(axis=0), ch=_bins(f[:, CH]))


def make_example(frames, loud):
    """Frames of a TAG recording -> a compact example, or ValueError (too short / too quiet)."""
    on = loud >= SILENCE_DB
    f = frames[on]
    secs = len(f) / FRAME_HZ
    if secs < MIN_EX_S:
        raise ValueError(f"only {secs:.1f} s of music was heard (at least {MIN_EX_S:.0f} s needed)")
    f = f[-int(MAX_EX_BINS * BIN):]
    mu = f[:, VIBE].mean(axis=0)
    sd = f[:, VIBE].std(axis=0)
    ch = f[:, CH]
    n = len(ch) // BIN
    raw = ch[:n * BIN].reshape(n, BIN, 12).mean(axis=1)
    return {"secs": round(float(len(f) / FRAME_HZ), 2), "mu": [round(float(x), 4) for x in mu],
            "sd": [round(float(x), 4) for x in sd], "ch": [[round(float(x), 3) for x in row] for row in raw]}


def clean_example(raw):
    """Untrusted JSON -> a valid example dict, or None."""
    if not isinstance(raw, dict):
        return None
    try:
        mu = [float(x) for x in raw.get("mu")]
        sd = [float(x) for x in raw.get("sd")]
        ch = [[float(x) for x in row] for row in raw.get("ch")][:MAX_EX_BINS]
        secs = float(raw.get("secs", 0.0))
    except (TypeError, ValueError):
        return None
    if len(mu) != NV or len(sd) != NV or not ch or any(len(r) != 12 for r in ch):
        return None
    vals = mu + sd + [x for r in ch for x in r] + [secs]
    if not all(math.isfinite(x) and abs(x) < 1e3 for x in vals):
        return None
    return {"secs": round(min(max(secs, 0.0), 60.0), 2), "mu": [round(x, 4) for x in mu],
            "sd": [round(max(x, 0.0), 4) for x in sd], "ch": [[round(min(max(x, 0.0), 1.0), 3) for x in r] for r in ch]}


def compile_tag(examples, mode="vibe", key_inv=False):
    """Clean examples -> the arrays the matcher uses (built once, on the GL thread, when a tag changes)."""
    ex = []
    for e in examples or ():
        mu, sd = np.array(e["mu"], np.float64), np.array(e["sd"], np.float64)
        scale = np.sqrt(sd ** 2 + SCALE ** 2)
        raw = np.array(e["ch"], np.float64)
        b = raw - raw.mean(axis=1, keepdims=True)
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
        rolls = np.stack([np.roll(b, k, axis=1) for k in range(12)]) if key_inv else b[None]
        ex.append((mu, scale, rolls))
    return dict(mode=mode if mode in MODES else "vibe", ex=ex)


def vibe_sim(mu_w, mu, scale):
    z = (mu_w - mu) / scale
    return float(math.exp(-0.5 * float(np.mean(z * z))))


def chroma_sim(wb, rolls):
    """Best diagonal mean Pearson of window bins wb (W, 12) against example bins (R, L, 12), +-1 bin tolerance."""
    W = len(wb)
    R, L = rolls.shape[0], rolls.shape[1]
    if W == 0 or L == 0:
        return 0.0
    if W > L:
        wb, W = wb[-L:], L
    S = (rolls.reshape(R * L, 12) @ wb.T).reshape(R, L, W)          # (R, L, W): one matmul
    if L > 2:                                                       # tolerance: each bin may pair with its neighbours
        Sd = S.copy()
        np.maximum(Sd[:, 1:], S[:, :-1], out=Sd[:, 1:])
        np.maximum(Sd[:, :-1], S[:, 1:], out=Sd[:, :-1])
        S = Sd
    n_o = L - W + 1
    acc = np.zeros((R, n_o))
    for i in range(W):                                              # diagonal sums for every offset at once
        acc += S[:, i:i + n_o, i]
    return max(0.0, float(acc.max()) / W)


def similarity(win, tpl):
    """0..1: how close the window (window()) is to a compiled template (compile_tag())."""
    if win is None or not tpl["ex"]:
        return 0.0
    best = 0.0
    for mu, scale, rolls in tpl["ex"]:
        v = vibe_sim(win["mu"], mu, scale)
        if tpl["mode"] == "section":
            c = chroma_sim(win["ch"], rolls)
            s = (c ** 0.7) * (v ** 0.3) if c > 0 and v > 0 else 0.0
        else:
            s = v
        best = max(best, s)
    return best


def build_batch(templates):
    """Every example of every template in a few padded float32 arrays, so one round is ~20 numpy calls whatever
    the count (same numbers as similarity() to float32 precision; tests/test_soundtags.py checks they agree)."""
    ids = tuple(templates)
    rows = [(k, t["mode"], e) for k, t in templates.items() for e in t["ex"]]
    if not rows:
        return dict(ids=ids, n=0)
    E = len(rows)
    L = max(r[2][2].shape[1] for r in rows)
    R = np.full((E, 12, L, 12), np.nan, np.float32)                 # NaN padding: np.fmax ignores it
    lens = np.zeros(E, np.int64)
    for j, (_, _, (mu, scale, rolls)) in enumerate(rows):
        n = rolls.shape[1]
        R[j, :, :n] = rolls if rolls.shape[0] == 12 else np.repeat(rolls, 12, axis=0)
        lens[j] = n
    return dict(ids=ids, n=E, owner=[k for k, _, _ in rows], section=np.array([m == "section" for _, m, _ in rows]),
                mu=np.stack([e[0] for _, _, e in rows]), scale=np.stack([e[1] for _, _, e in rows]),
                Rt=np.ascontiguousarray(R.reshape(E * 12 * L, 12).T), lens=lens, L=L)


def batch_sims(win, b):
    """{template id: sim} for one window against a build_batch() result."""
    out = {k: 0.0 for k in b["ids"]}
    if win is None or b["n"] == 0:
        return out
    z = (win["mu"][None, :] - b["mu"]) / b["scale"]
    v = np.exp(-0.5 * np.mean(z * z, axis=1))                       # (E,)
    s = v
    sec = b["section"]
    wb = win["ch"]
    if sec.any() and len(wb):
        E, L, lens = b["n"], b["L"], b["lens"]
        W = int(min(len(wb), lens[sec].min()))
        wb = np.ascontiguousarray(wb[-W:], np.float32)
        S = (wb @ b["Rt"]).reshape(W, E, 12, L)                     # one small matmul for everything
        if L > 2:                                                   # tolerance: each bin may pair with its neighbours
            Sd = S.copy()
            np.fmax(Sd[..., 1:], S[..., :-1], out=Sd[..., 1:])
            np.fmax(Sd[..., :-1], S[..., 1:], out=Sd[..., :-1])
            S = Sd
        n_o = L - W + 1
        acc = S[0, :, :, 0:n_o].copy()
        for i in range(1, W):
            acc += S[i, :, :, i:i + n_o]
        acc = np.where(np.arange(n_o)[None, None, :] > (lens - W)[:, None, None], -np.inf, acc)
        c = np.maximum(acc.reshape(E, -1).max(axis=1).astype(np.float64) / W, 0.0)
        s = np.where(sec, np.where((c > 0) & (v > 0), c ** 0.7 * v ** 0.3, 0.0), v)
    for k, x in zip(b["owner"], s):
        if x > out[k]:
            out[k] = float(x)
    return out


# ------------------------------------------------------------------ trigger logic (pure, deterministic)

HYST = 0.08                     # a tag must drop this far under its threshold before it can fire again
SMOOTH_S = 0.5                  # EMA time constant of the similarity the decider looks at
QUANTS = ("none", "beat", "bar", "phrase")
TAG_DEFAULTS = dict(mode="vibe", key_inv=False, sens=0.5, hold=3.0, cooldown=30.0, quant="bar", morph=None,
                    prio=0, on=True, only_auto=False)


def threshold(sens):
    """Sensitivity 0..1 -> similarity threshold (0.5 -> 0.7; 1 -> 0.5; 0 -> 0.9)."""
    return 0.9 - 0.4 * min(1.0, max(0.0, float(sens)))


class Decider:
    """Which tag fires, given a stream of similarities. step() is pure apart from its own per-tag state."""

    def __init__(self):
        self.st = {}                    # id -> dict(s, t, over, rearm, last)
        self.last_any = -1e9

    def reset(self, keep_cooldowns=True):
        for s in self.st.values():
            s["over"] = None
            s["rearm"] = False if keep_cooldowns else True

    def step(self, now, sims, tags, armed=True, block=None, consume=False, autopilot=False, gap=8.0, order=()):
        """now: seconds; sims {id: raw sim} (only for ids with a new value); tags {id: cfg}. Returns
        (fire id or None, meters [[id, sim, thr, st]], [(id, reason)] of tags that were ready but held).
        st: 0 below, 1 near, 2 over (holding), 3 cooldown, 4 tag off. consume=True: a ready tag that is
        blocked uses up its trigger (it must leave and come back), e.g. while the performer has the controls."""
        for k in [k for k in self.st if k not in tags]:
            del self.st[k]
        ready, meters, held = [], [], []
        for k in (order or sorted(tags)):
            cfg = tags.get(k)
            if cfg is None:
                continue
            s = self.st.get(k)
            if s is None:
                # a new tag (or one seen for the first time after ARM) is not re-armed yet: music that is ALREADY
                # playing never fires; the tag fires when its music arrives (drops under, then comes over)
                s = self.st[k] = dict(s=0.0, t=None, over=None, rearm=False, last=-1e9)
            if k in sims:
                x = float(sims[k])
                if s["t"] is None:
                    s["s"] = x
                else:
                    a = 1.0 - math.exp(-max(0.0, now - s["t"]) / SMOOTH_S)
                    s["s"] += (x - s["s"]) * a
                s["t"] = now
            thr = threshold(cfg.get("sens", 0.5))
            v = s["s"]
            if s["t"] is None:                              # no similarity yet: nothing to decide on
                meters.append([k, 0.0, round(thr, 3), 4 if not cfg.get("on", True) else 0])
                continue
            if v >= thr:
                if s["over"] is None:
                    s["over"] = now
            elif v < thr - HYST:
                s["over"] = None
                s["rearm"] = True
            else:
                s["over"] = None                    # in the hysteresis band: not holding, not re-armed
            cool = now - s["last"] < float(cfg.get("cooldown", 30.0))
            if not cfg.get("on", True):
                code = 4
            elif cool:
                code = 3
            elif s["over"] is not None:
                code = 2
            elif v >= thr - 0.1:
                code = 1
            else:
                code = 0
            meters.append([k, round(v, 3), round(thr, 3), code])
            if (cfg.get("on", True) and s["over"] is not None and now - s["over"] >= float(cfg.get("hold", 3.0))
                    and s["rearm"] and not cool and (autopilot or not cfg.get("only_auto"))):
                ready.append(k)
        if not armed:
            return None, meters, []
        if ready and (block is not None or now - self.last_any < gap):
            why = block or "gap"
            for k in ready:
                held.append((k, why))
                if consume and block is not None:
                    self.st[k]["rearm"] = False
                    self.st[k]["over"] = None
            return None, meters, held
        if not ready:
            return None, meters, []
        ready.sort(key=lambda k: (-int(tags[k].get("prio", 0)), -self.st[k]["s"], k))
        k = ready[0]
        s = self.st[k]
        s["last"], s["rearm"], s["over"] = now, False, None
        self.last_any = now
        return k, meters, [(o, "priority") for o in ready[1:]]


# ------------------------------------------------------------------ suggestions


SEG_S = 4.0
MIN_GAP_S = 12.0                # a section "comes back" only after at least this much other music


def suggest(frames, loud, min_count=2, limit=5, thr=0.72):
    """Recent audio -> recurring sections: [{label, n, secs, at (segment start, s before the end), example}].
    Segments of 4 s, greedy leader clustering on vibe similarity; a cluster counts when it occurs at least
    `min_count` times with more than MIN_GAP_S of other music in between (the section came BACK)."""
    seg = int(round(SEG_S * FRAME_HZ))
    n = len(frames) // seg
    segs = []
    for i in range(n):
        f, lo = frames[i * seg:(i + 1) * seg], loud[i * seg:(i + 1) * seg]
        on = lo >= SILENCE_DB
        if on.mean() < 0.6:
            continue
        g = f[on]
        segs.append(dict(i=i, mu=g[:, VIBE].mean(axis=0), sd=g[:, VIBE].std(axis=0)))
    clusters = []
    for s in segs:
        best, bi = 0.0, -1
        for ci, c in enumerate(clusters):
            lead = c[0]
            v = vibe_sim(s["mu"], lead["mu"], np.sqrt(lead["sd"] ** 2 + SCALE ** 2))     # the sound, not the chord
            if v > best:
                best, bi = v, ci
        if best >= thr:
            clusters[bi].append(s)
        else:
            clusters.append([s])
    out = []
    for c in clusters:
        idx = sorted(x["i"] for x in c)
        runs = [[idx[0]]]
        for j in idx[1:]:
            if (j - runs[-1][-1]) * SEG_S <= MIN_GAP_S:          # a short break inside a section is not a return
                runs[-1].append(j)
            else:
                runs.append([j])
        if len(runs) < min_count:
            continue
        longest = max(runs, key=len)
        a, b = longest[0] * seg, min((longest[-1] + 1) * seg, longest[0] * seg + MAX_EX_BINS * BIN)
        try:
            ex = make_example(frames[a:b], loud[a:b])
        except ValueError:
            continue
        out.append(dict(n=len(runs), secs=round(len(idx) * SEG_S, 1),
                        at=[round((len(frames) - r[0] * seg) / FRAME_HZ, 1) for r in runs], example=ex))
    out.sort(key=lambda x: (-x["n"], -x["secs"]))
    for i, x in enumerate(out[:limit]):
        x["label"] = f"Section {chr(65 + i)}: heard {x['n']}x, {x['secs']:.0f} s"
    return out[:limit]
