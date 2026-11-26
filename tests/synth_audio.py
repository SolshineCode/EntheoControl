"""Synthetic music for the sound-tag tests: chords, drums, noise bursts, two different 'songs'. Deterministic."""
import numpy as np

SR = 48000


def note(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def tone(freqs, secs, amp=0.2, harm=(1.0, 0.5, 0.25), saw=False, sr=SR, phase=0.0):
    t = np.arange(int(secs * sr)) / sr + phase
    y = np.zeros_like(t)
    for f in freqs:
        if saw:
            for k in range(1, 9):
                y += np.sin(2 * np.pi * f * k * t) / k
        else:
            for k, a in enumerate(harm, 1):
                y += a * np.sin(2 * np.pi * f * k * t)
    return amp * y / max(1, len(freqs))


def drums(secs, bpm, pattern="four", seed=0, sr=SR, kick=0.6, hat=0.15):
    rng = np.random.default_rng(seed)
    n = int(secs * sr)
    y = np.zeros(n)
    beat = 60.0 / bpm
    kl = int(0.15 * sr)
    tk = np.arange(kl) / sr
    kick_s = kick * np.sin(2 * np.pi * (50 + 80 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 18)
    hl = int(0.04 * sr)
    hat_s = hat * rng.standard_normal(hl) * np.exp(-np.arange(hl) / sr * 90)
    steps = {"four": [0, 1, 2, 3], "break": [0, 1.5, 2.5]}[pattern]
    b = 0.0
    while b * beat < secs:
        for s in steps:
            i = int((b + s) * beat * sr)
            if i + kl < n:
                y[i:i + kl] += kick_s
        for s in np.arange(0, 4, 0.5):
            i = int((b + s + 0.25) * beat * sr)
            if i + hl < n:
                y[i:i + hl] += hat_s
        b += 4
    return y


def song_a(secs=16.0, shift=0, gain=1.0, offset=0.0):
    """Pop-ish: C G Am F, one chord per bar at 120 BPM (2 s), sine pads + four-on-the-floor."""
    prog = [[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]]
    out = []
    bar = 2.0
    total = 0.0
    k = int(offset / bar)
    while total < secs:
        ch = [note(m + shift) for m in prog[k % 4]]
        out.append(tone(ch, bar, amp=0.35))
        total += bar
        k += 1
    y = np.concatenate(out)[:int(secs * SR)]
    return gain * (y + drums(secs, 120, "four"))


def song_b(secs=16.0, gain=1.0):
    """Different music: saw-wave jazz chords Dm7 G7 Cmaj7 A7b9 at 90 BPM, a broken beat, a noisy texture."""
    prog = [[62, 65, 69, 72], [55, 59, 62, 65], [60, 64, 67, 71], [57, 61, 64, 70]]
    bar = 60.0 / 90 * 4
    out, total, k = [], 0.0, 0
    while total < secs:
        out.append(tone([note(m) for m in prog[k % 4]], bar, amp=0.25, saw=True))
        total += bar
        k += 1
    y = np.concatenate(out)[:int(secs * SR)]
    rng = np.random.default_rng(7)
    noise = 0.05 * rng.standard_normal(len(y))
    return gain * (y + drums(secs, 90, "break", seed=3, kick=0.4, hat=0.3) + noise)


def noise_bursts(secs=12.0, gain=1.0):
    rng = np.random.default_rng(11)
    y = np.zeros(int(secs * SR))
    for s in np.arange(0, secs, 0.75):
        i = int(s * SR)
        L = int(0.2 * SR)
        y[i:i + L] += 0.4 * rng.standard_normal(min(L, len(y) - i))
    return gain * y


def silence(secs):
    return np.zeros(int(secs * SR))


def feed(eng, y, block=512):
    """Push a signal through the engine in capture-sized stereo blocks (what Audio.taps hands it)."""
    y = np.asarray(y, np.float32)
    for i in range(0, len(y) - block + 1, block):
        b = y[i:i + block]
        eng.feed(np.stack([b, b], axis=1))
