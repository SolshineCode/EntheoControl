"""Cost budget of the DEFAULT look (every new layer off): what is switched off must cost ~nothing.

Regression 2026-11-26: headless default fps fell 138 -> 86 between show_preview4 and show_preview5. Cause: the scene
uniform upload did `if name in scene_prog` for every uniform every frame; moderngl.Program has no __contains__, so
each test walked all program members through a Python generator. ~80 new params (each a `p_*` entry in `uni`, most
not in the shader, so a full walk each) pushed it to ~37k generator steps a frame. Fix: live_visuals.prog_names
(a cached frozenset). These tests pin both the mechanism (deterministic) and the frame rate (loose floor).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASELINE = ROOT / "tests" / "golden" / "perf_baseline.json"


def _run(tmp_path, frames, driver=True, *extra):
    cmd = [sys.executable, str(ROOT / ("tests/perf_engine_driver.py" if driver else "live_visuals.py")),
           "--test", str(frames), "--out", str(tmp_path / f"f{frames}"), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "[test]" in r.stdout, r.stdout[-800:]
    return r.stdout


def test_prog_names_is_a_cached_set():
    import live_visuals as LV

    class FakeProg:
        walks = 0

        def __iter__(self):
            FakeProg.walks += 1
            yield from ("p_master", "texA", "pal")

    p = FakeProg()
    names = LV.prog_names(p)
    assert names == {"p_master", "texA", "pal"} and isinstance(names, frozenset)
    for _ in range(100):
        assert "p_master" in LV.prog_names(p) and "p_coil_count" not in LV.prog_names(p)
    assert FakeProg.walks == 1, "the members are walked once per program, not per lookup"


def test_default_look_does_no_per_frame_program_walks(tmp_path):
    """Deterministic: Program.__iter__ steps per frame of the default look (startup cancels between two run lengths).
    Before the fix: ~37,000 a frame. Budget: 50 (a few one-off lookups may land mid-run, never one per uniform)."""
    def steps(frames):
        out = _run(tmp_path, frames)
        m = re.search(r"\[perf\] program_iter_steps=(\d+)", out)
        assert m, out[-800:]
        return int(m.group(1))
    a, b = steps(30), steps(150)
    per_frame = (b - a) / 120.0
    assert per_frame < 50, f"{per_frame:.0f} Program member walks per frame (use live_visuals.prog_names)"


def test_default_look_frame_rate_floor(tmp_path):
    """Loose wall-clock budget against the recorded baseline (tests/golden/perf_baseline.json). The floor is half the
    baseline, so machine noise / other agents' CPU load does not flake it, while the 2026-11-26 regression
    (~0.4x under load) fails. Best of 2 runs."""
    base = json.loads(BASELINE.read_text())
    best = 0.0
    for _ in range(2):
        out = _run(tmp_path, base["frames"], False)
        best = max(best, float(re.search(r"= (\d+) fps", out).group(1)))
        if best >= base["fps"] * base["floor"]:
            break
    if best < base["fps"] * base["floor"]:
        pytest.fail(f"default look {best:.0f} fps < {base['floor']:.0%} of baseline {base['fps']} fps "
                    f"({base['note']})")
