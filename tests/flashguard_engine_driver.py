"""Headless engine run with one trigger hammered at a fixed rate (used by tests/test_flashguard.py).

    python tests/flashguard_engine_driver.py TRIGGER HZ [live_visuals.py arguments ...]

The static headless controller holds a trigger steadily; this wraps it so TRIGGER (strobe, flash, invert, blackout,
punch) is gated on and off HZ times a second on the engine's own frame clock, the way the show controller gates the
strobe on the beat. The engine prints `[flashguard] ... rate_in=.. rate_out=..` at exit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live_visuals as LV  # noqa: E402


def main():
    name, hz = sys.argv[1], float(sys.argv[2])
    sys.argv = [sys.argv[0], *sys.argv[3:]]
    make = LV.make_controller

    def wrapped(*a, **k):
        ctl = make(*a, **k)
        update, clock = ctl.update, [0.0]

        def gated(dt, bands):
            fr = update(dt, bands)
            clock[0] += dt
            fr.values = dict(fr.values or {})
            fr.values[name] = 1.0 if (clock[0] * hz) % 1.0 < 0.5 else 0.0
            return fr
        ctl.update = gated
        return ctl
    LV.make_controller = wrapped
    LV.main()


if __name__ == "__main__":
    main()
