"""Headless engine run that counts moderngl.Program member walks (used by tests/test_perf_budget.py).

    python tests/perf_engine_driver.py [live_visuals.py arguments ...]

moderngl.Program has no __contains__, so `name in program` falls back to __iter__ and walks every member. This wraps
Program.__iter__ with a counter and prints `[perf] program_iter_steps=N` at exit, so a test can tell the per-frame
Python cost of program lookups from two runs of different length (startup cost cancels).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import moderngl  # noqa: E402

_steps = [0]
_orig_iter = moderngl.Program.__iter__


def _counting_iter(self):
    for name in _orig_iter(self):
        _steps[0] += 1
        yield name


moderngl.Program.__iter__ = _counting_iter

import live_visuals as LV  # noqa: E402


def main():
    try:
        LV.main()
    finally:
        print(f"[perf] program_iter_steps={_steps[0]}", flush=True)


if __name__ == "__main__":
    main()
