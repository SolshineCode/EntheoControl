"""Math stack, GL side (tiny offscreen renders, 160x100): seamless wraps, antialiasing, blends, empty = no draw.
Uses the GPU: run under .locks/gpu_suite like every other render test."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import stack  # noqa: E402

moderngl = pytest.importorskip("moderngl")
SIZE = (160, 100)


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context()
    except Exception as ex:
        pytest.skip(f"no GL context: {ex}")
    yield c
    c.release()


@pytest.fixture(scope="module")
def sp(ctx):
    p = stack.StackPass(ctx, SIZE)
    yield p
    p.release()


def state(layers):
    st = stack.StackState()
    for lay in layers:
        st.add(lay.get("gen", 0), lay)
    st.restore(st.to_json())                                   # no fade-in
    return st


def shot(sp, st, dt=0.0, k=0, hold=False):
    st.step(dt, dict(bpm=120.0, beat=0.0), aspect=SIZE[0] / SIZE[1], k=k, hold=hold)
    sp.render(st.frame, 1.0)
    return np.asarray(sp.read(), dtype=np.int16)


def lay_for(gen, space, rng):
    return dict(gen=gen, space=space, wave=rng.randrange(4), freq=rng.choice([3, 5.5, 9]), angle=rng.random(),
                count=rng.randint(2, 9), folds=rng.randint(3, 9), warp=rng.random() * 0.6, grad=rng.choice([0.0, 1.0]),
                speed=1.0, spin=0.0, hue=0.2, soft=0.1,
                stops=[[i / n, rng.random(), rng.random(), rng.random()] for n in [rng.randint(2, 6)] for i in range(n)])


@pytest.mark.parametrize("gen", range(8))
def test_whole_wrap_periods_do_not_change_the_picture(sp, gen):
    """Every wrapped phase + K whole periods must give the same frame: the wrap is at a true period of the
    pattern (band index, colour cycle, per-band hash, angular counts) in every generator x space."""
    rng = random.Random(gen)
    for space in range(8):
        st = state([lay_for(gen, space, rng)])
        st._phase(1).update(scroll=7.37, spin=1.1, warp=0.3, hue=0.4)
        a = shot(sp, st, hold=True)
        b = shot(sp, st, hold=True, k=2)
        diff = np.abs(a - b)
        assert a[..., :3].max() > 40, (gen, space, "black frame")
        assert (diff > 24).mean() < 0.004, (gen, space, float((diff > 24).mean()))   # float32 at edges only


@pytest.mark.parametrize("gen", range(8))
def test_no_pop_when_the_scroll_phase_wraps(sp, gen):
    """Scroll straddling its wrap point: the frame just before and just after differ no more than two ordinary
    neighbouring frames do."""
    rng = random.Random(100 + gen)
    for space in (0, 1, 2, 3, 7):
        st = state([lay_for(gen, space, rng)])
        st.step(0.0)
        W = st.frame[0]["W"]
        step = 0.004
        ph = st._phase(1)
        ph["scroll"] = W * 3 - 2.5 * step
        frames = []
        for _ in range(5):                                      # two frames before the wrap, three after
            frames.append(shot(sp, st, hold=True))
            ph["scroll"] += step
        d = [float(np.abs(x - y).mean()) for x, y in zip(frames, frames[1:])]
        assert max(d) <= 3.0 * (sorted(d)[1] + 0.3), (gen, space, d)


def test_empty_stack_is_black_and_layers_light_it(sp):
    assert shot(sp, state([])).max() == 0
    img = shot(sp, state([dict(gen=0, freq=4, soft=0.0, angle=0.0)]))
    assert img[..., :3].max() > 200 and (img[..., :3].max(axis=2) < 8).mean() > 0.3      # crisp bands on true black
    assert img[..., 3].max() == 0                                                        # screen blend hides nothing


def test_stripes_are_antialiased_not_stair_stepped(sp):
    img = shot(sp, state([dict(gen=0, freq=5, soft=0.0, angle=0.07, stops=[[0, 1, 1, 1], [0.5, 1, 1, 1]])]))
    lum = img[..., 0]
    mid = ((lum > 20) & (lum < 235)).mean()
    assert 0.01 < mid < 0.2                                     # some in-between pixels along the edges, not a blur


def test_blend_modes_and_alpha(sp):
    base = dict(gen=0, freq=3, soft=0.0, duty=0.5, stops=[[0, 1, 0.5, 0.2], [0.5, 1, 0.5, 0.2]])
    top = dict(gen=0, freq=3, soft=0.0, duty=0.5, angle=0.25, stops=[[0, 0.3, 0.6, 1], [0.5, 0.3, 0.6, 1]])
    out = {}
    for b, name in enumerate(stack.BLENDS):
        out[name] = shot(sp, state([base, dict(top, blend=b)]))
    assert len({o.tobytes() for o in out.values()}) == len(stack.BLENDS)                 # seven different pictures
    assert out["Add"][..., 3].max() == 0 and out["Difference"][..., 3].max() == 0
    assert out["Replace"][..., 3].min() >= 250                                           # opaque everywhere
    a = out["Mask below"][..., 3]
    assert a.max() >= 250 and a.min() == 0                                               # hides only where it is dark
    assert out["Mask below"][..., 2].max() <= 55                                          # a mask adds no light of its own
    assert out["Over"][..., 3].max() >= 250 and out["Over"][..., 3].min() == 0


def test_amount_scales_light_and_cover(sp):
    st = state([dict(gen=0, blend=5)])
    st.step(0.0)
    sp.render(st.frame, 0.5)
    half = np.asarray(sp.read(), dtype=np.int16)
    assert 120 <= half[..., 3].max() <= 135 and half[..., :3].max() <= 135


def test_palette_layers_take_the_show_palette(sp):
    st = state([dict(gen=0, color=1, freq=5, soft=0.0)])
    st.step(0.0)
    sp.render(st.frame, 1.0, [(0, 1, 0)] * 5)
    img = np.asarray(sp.read(), dtype=np.int16)
    assert img[..., 1].max() > 240 and img[..., 0].max() < 8 and img[..., 2].max() < 8
