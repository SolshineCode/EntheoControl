"""Headless engine run with a REAL control.Controller and one scheduled scene recall (tests/test_transitions.py).

    python tests/transitions_engine_driver.py KIND[:FRAME[:SECS[:SLOT]]] [live_visuals.py arguments ...]

KIND "none" = no recall (idle). Otherwise slot 1 (a different look saved by this driver into a temp state dir, never
the show's) is recalled at FRAME (default 30) with transition KIND lasting SECS (default 1.0). The engine prints
`[transition] draws=.. timed=.. b_ms=.. comp_ms=..` at exit (headless). Set LV_TRANS_TIMING=1 for GPU timings.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import control  # noqa: E402
import live_visuals as LV  # noqa: E402

B_LOOK = dict(zoom=0.45, hue=0.35, symmetry=0.0, twirl=0.6, palette=2.0, l_beads=0.9, density=0.8)


def main():
    spec = sys.argv[1].split(":")
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    kind = spec[0]
    at = int(spec[1]) if len(spec) > 1 else 30
    secs = float(spec[2]) if len(spec) > 2 else 1.0
    tmp = tempfile.mkdtemp(prefix="lv_trans_")

    def make(*a, **k):
        ctl = control.Controller(tmp, restore=False, midi=False, ai=False, threads=False)
        ctl.quantize, ctl.morph_bars = "off", 0.0
        base = dict(ctl.base)
        ctl.scene_save(0)
        ctl.base.update(B_LOOK)
        ctl.scene_save(1)
        ctl.base.update(base)
        ctl._sm = dict(ctl.base)
        ctl.cur = 0
        update, n = ctl.update, [0]

        def stepped(dt, bands):
            n[0] += 1
            if kind != "none" and n[0] == at:
                ctl.apply([{"cmd": "scene_recall", "i": 1, "transition": {"kind": kind, "secs": secs}}])
            return update(dt, bands)
        ctl.update = stepped
        return ctl
    LV.make_controller = make
    LV.main()


if __name__ == "__main__":
    main()
