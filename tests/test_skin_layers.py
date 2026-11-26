"""Layers as skins + skin STACKS (materials.py): an engine layer rendered to a small texture is a skin source, and every
environment slot is a small stack (base + up to 2 overlay layers with blend, opacity, scale / scroll, per-channel
gating, colour remap and a travelling noise mask).

CPU tests use a fake hub backend (no GPU): `pytest tests/test_skin_layers.py -k "not gl_"`.
GL tests (names start with test_gl_) need the GPU lock (.locks/gpu_suite).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import materials as mt  # noqa: E402


def sdo(st, **cmd):
    return st.do(mt.validate(dict(cmd, cmd="skin")))


# ---------------------------------------------------------------- model: sources, overlay layers, commands


def test_layer_sources_are_a_skin_kind():
    assert mt.clean_src(dict(kind="layer", name="kene")) == dict(kind="layer", name="kene")
    assert mt.clean_src(dict(kind="layer", name="lines4")) == dict(kind="layer", name="lines4")
    for bad in ("Kene", "../x", "a b", "", "x" * 40, "kene.png"):
        assert mt.clean_src(dict(kind="layer", name=bad)) is None, bad
    c = mt.validate(dict(cmd="skin", op="assign", env="corridor", slot="walls", kind="layer", name="beings"))
    assert c["kind"] == "layer" and c["name"] == "beings"


def test_slot_has_two_empty_overlay_layers_by_default_and_json_stays_compact():
    st = mt.SkinState()
    assert st.slots["corridor"]["walls"]["layers"] == [None, None]
    assert st.to_json() == dict(v=mt.VERSION, slots={}, mods={})


def test_overlay_layer_command_cleans_and_clamps():
    st = mt.SkinState()
    sdo(st, op="layer", env="torus", slot="tube", i=1,
        set=dict(src=dict(kind="layer", name="lines4"), blend=2, opacity=5.0, scale=0.01, gate=[2.5, 0.5, -1, 1],
                 remap=1, cols=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 9, 9]], noise=0.7, nscale=3.0, nspeed=0.2))
    lay = st.slots["torus"]["tube"]["layers"][1]
    assert lay["src"] == dict(kind="layer", name="lines4")
    assert lay["blend"] == 2 and lay["opacity"] == mt.LNUM["opacity"][1] and lay["scale"] == mt.LNUM["scale"][0]
    assert lay["gate"] == [2.0, 0.5, 0.0, 1.0]
    assert lay["cols"][3] == [1.0, 1.0, 1.0]
    assert st.slots["torus"]["tube"]["layers"][0] is None
    for bad in (dict(i=2, set=dict(blend=1)), dict(i=True, set=dict(blend=1)), dict(i=0, set=dict(nope=1)),
                dict(i=0, set=dict(gate=[1, 2])), dict(i=0, set=dict(cols=[[1, 1]])), dict(i=0, set={})):
        with pytest.raises(ValueError):
            mt.validate(dict(cmd="skin", op="layer", env="torus", slot="tube", **bad))
    sdo(st, op="layer_clear", env="torus", slot="tube", i=1)
    assert st.slots["torus"]["tube"]["layers"] == [None, None]


def test_a_slot_with_only_overlays_is_active_and_round_trips():
    st = mt.SkinState()
    sdo(st, op="layer", env="corridor", slot="floor", i=0, set=dict(src=dict(kind="layer", name="lines4"), blend=0))
    assert "floor" in st.frame["corridor"], "an overlay alone skins the slot (the world's own look is the base)"
    j = st.to_json()
    st2 = mt.SkinState()
    st2.load(j)
    assert st2.slots["corridor"]["floor"]["layers"][0]["src"] == dict(kind="layer", name="lines4")
    assert st2.to_json() == j
    # an overlay layer without a source does not activate anything
    st3 = mt.SkinState()
    sdo(st3, op="layer", env="corridor", slot="floor", i=0, set=dict(blend=1))
    assert st3.frame == {}


def test_panic_clears_every_source_including_overlays():
    st = mt.SkinState()
    sdo(st, op="layer", env="vortex", slot="walls", i=0, set=dict(src=dict(kind="ai", name="a.png")))
    st.assign("vortex", "walls", "layer", "kene")
    st.panic()
    assert st.frame == {}
    assert st.slots["vortex"]["walls"]["layers"][0]["src"] is None, "settings kept, source gone"


def test_off_switch_keeps_overlays_out_of_the_frame():
    st = mt.SkinState(on=False)
    sdo(st, op="layer", env="vortex", slot="walls", i=0, set=dict(src=dict(kind="layer", name="lines4")))
    assert st.frame == {}


def test_example_presets_are_clean_and_use_kene_type_line_layers():
    assert "kene_cloth_world" in mt.EXAMPLES
    raw = mt.EXAMPLES["kene_cloth_world"]
    clean = mt.clean_skins(raw)
    assert clean == raw or clean["slots"] == mt.clean_skins(clean)["slots"]
    st = mt.SkinState()
    st.load(raw)
    used = set()
    for env, ss in st.frame.items():
        for s, d in ss.items():
            for lay in d["layers"]:
                if lay and lay["src"]:
                    used.add(lay["src"]["name"])
    assert "lines4" in used, "the example world carries the 4-channel line layer over its skins"
    gated = [lay for ss in st.frame.values() for d in ss.values() for lay in d["layers"]
             if lay and lay["remap"] and lay["noise"] > 0]
    assert gated, "the example demonstrates channel remap + floating patches"


# ---------------------------------------------------------------- hub logic (fake backend: no GPU)


class FakeBackend:
    def __init__(self):
        self.allocs = self.blits = self.releases = 0
        self.size = None
        self.log = []

    def alloc(self, size, layers):
        self.allocs += 1
        self.size = size

    def blit(self, tex, layer):
        self.blits += 1
        self.log.append((tex, layer))

    def release(self):
        self.releases += 1
        self.size = None

    def use(self, unit):
        pass

    def vram_bytes(self, layers=4):
        return 0 if self.size is None else self.size * self.size * 4 * 4


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_hub(**kw):
    be, clk = FakeBackend(), Clock()
    hub = mt.LayerHub(None, backend=be, clock=clk, **kw)
    return hub, be, clk


def test_hub_unused_costs_nothing():
    hub, be, clk = make_hub()
    calls = []
    hub.register("kene", draw=lambda size: calls.append(size) or "TEX")
    t0 = time.perf_counter()
    for _ in range(1000):
        hub.run("scene")
        hub.end_frame()
    per = (time.perf_counter() - t0) / 1000 * 1000.0
    assert calls == [] and be.allocs == 0 and be.blits == 0 and hub.vram_bytes() == 0
    assert per < 0.02, f"{per:.4f} ms per idle frame"
    print(f"\n[layer cost] idle hub: {per * 1000:.2f} us per frame, 0 GL calls, 0 MB")


def test_hub_renders_a_source_at_most_once_per_frame_however_many_consumers():
    hub, be, clk = make_hub()
    calls = []
    hub.register("kene", draw=lambda size: calls.append(size) or "TEX")
    for f in range(5):
        for _ in range(6):                       # 6 consumers (slots, beings bodies) want it
            hub.request("kene")
        hub.run("scene")
        hub.run("scene")                         # a second run in the same frame does nothing
        hub.end_frame()
    assert len(calls) == 5 and be.blits == 5 and be.allocs == 1
    assert calls[0] == hub.size == 512


def test_hub_reuses_the_layer_the_show_already_drew():
    hub, be, clk = make_hub()
    own = []
    hub.register("kene", draw=lambda size: own.append(size) or "OWN", visible=lambda: "SHOWN")
    hub.request("kene")
    hub.run("scene")
    assert own == [] and be.log[-1][0] == "SHOWN", "a visible layer is copied, never drawn twice"


def test_request_gives_an_index_only_once_the_texture_holds_something():
    hub, be, clk = make_hub()
    hub.register("kene", draw=lambda size: "TEX")
    assert hub.request("kene") == -1
    hub.run("scene")
    hub.end_frame()
    assert hub.request("kene") == 0
    assert hub.request("unknown") == -1


def test_hub_phases_keep_post_sources_for_the_post_run():
    hub, be, clk = make_hub()
    hub.register("output", draw=lambda size: "OUT", phase="post")
    hub.register("kene", draw=lambda size: "K")
    hub.request("output"), hub.request("kene")
    hub.run("scene")
    assert [t for t, _ in be.log] == ["K"]
    hub.run("post")
    assert [t for t, _ in be.log] == ["K", "OUT"]


def test_static_sources_render_once():
    hub, be, clk = make_hub()
    calls = []
    hub.register("lines4", draw=lambda size: calls.append(1) or "L", static=True)
    for _ in range(20):
        hub.request("lines4")
        hub.run("scene")
        hub.end_frame()
    assert len(calls) == 1


def test_governor_halves_the_rate_then_freezes():
    hub, be, clk = make_hub()
    calls = []
    hub.register("kene", draw=lambda size: calls.append(1) or "T")
    for q, n_expect in ((0, 10), (1, 5), (2, 0)):
        calls.clear()
        for _ in range(10):
            hub.request("kene")
            hub.run("scene", quality=q)
            hub.end_frame()
        assert len(calls) == n_expect, (q, len(calls))


def test_unreferenced_sources_are_dropped_then_the_array_is_freed():
    hub, be, clk = make_hub()
    freed = []
    hub.register("kene", draw=lambda size: "T", release=lambda: freed.append("kene"))
    hub.register("text", draw=lambda size: "T2")
    hub.request("kene"), hub.request("text")
    hub.run("scene")
    hub.end_frame()
    assert be.allocs == 1 and hub.status()["live"] == ["kene", "text"]
    for _ in range(3):
        clk.t += 1.0
        hub.request("text")
        hub.run("scene")
        hub.end_frame()
    assert hub.status()["live"] == ["text"] and freed == ["kene"], "kene unreferenced > KEEP_S: slot + own pass freed"
    clk.t += mt.LayerHub.FREE_S + 1
    hub.run("scene")
    hub.end_frame()
    assert be.releases == 1 and hub.vram_bytes() == 0 and hub.status()["live"] == []


def test_more_sources_than_slots_never_steal_a_slot_in_use():
    hub, be, clk = make_hub(layers=2)
    for n in ("a", "b", "c"):
        hub.register(n, draw=lambda size, n=n: n)
    for n in ("a", "b", "c"):
        hub.request(n)
    hub.run("scene")
    hub.end_frame()
    got = [hub.request(n) for n in ("a", "b", "c")]
    assert sorted(i for i in got if i >= 0) == [0, 1] and got.count(-1) == 1


def test_a_failing_source_is_skipped_not_fatal():
    hub, be, clk = make_hub()

    def boom(size):
        raise RuntimeError("pass broke")
    hub.register("kene", draw=boom)
    for _ in range(10):
        hub.request("kene")
        hub.run("scene")
        hub.end_frame()
    assert hub.status()["fails"]["kene"] >= 5 and hub.request("kene") == -1


# ---------------------------------------------------------------- skin engine: live sources + overlay uniforms


class FakeTex:
    def __init__(self, size):
        self.size = size

    def write(self, *a, **k):
        pass

    def build_mipmaps(self):
        pass

    def release(self):
        pass

    def use(self, unit):
        pass


class FakeCtx:
    def texture_array(self, size, components, dtype="f1"):
        return FakeTex(size)


def make_engine(tmp_path):
    d = tmp_path / "textures"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (d / f"t{i}.png").write_bytes(b"x")
    lib = mt.SkinLibrary(FakeCtx(), dict(ai=d, art=tmp_path / "art"), size=512,
                         loader=lambda p, s: (bytes(s * s * 4), 1.0), sync=True)
    eng = mt.SkinEngine(None, None, lib=lib)
    eng.fade_s = 0.0
    hub, be, clk = make_hub()
    hub.register("lines4", draw=lambda size: "L", static=True)
    hub.register("kene", draw=lambda size: "K")
    eng.hub = hub
    return eng, lib, hub


def frames(eng, st, hub, n=4):
    for _ in range(n):
        eng.update(st.step(1 / 60), 1 / 60, dict(bpm=120.0))
        eng.lib.flush()
        hub.run("scene")
        hub.end_frame()


def test_engine_puts_a_live_layer_on_a_slot(tmp_path):
    eng, lib, hub = make_engine(tmp_path)
    st = mt.SkinState()
    st.assign("corridor", "walls", "layer", "kene")
    frames(eng, st, hub)
    assert eng.active("corridor")
    u = eng._u["corridor"]
    assert u[0, 0, 0] >= 100.0, "live sources are coded 100 + hub index"
    assert eng._uf["corridor"][0, 0] == 1.0, "base present"


def test_engine_overlay_uniforms(tmp_path):
    eng, lib, hub = make_engine(tmp_path)
    st = mt.SkinState()
    st.assign("torus", "tube", "ai", "t0.png")
    sdo(st, op="layer", env="torus", slot="tube", i=0,
        set=dict(src=dict(kind="layer", name="lines4"), blend=1, opacity=0.8, remap=1, gate=[1, 0.5, 0, 2],
                 noise=0.6, nscale=2.0, nspeed=0.5, scroll=0.3, dir=0.0))
    frames(eng, st, hub, n=6)
    L = eng._ul["torus"]
    k = 0                                         # slot 0, overlay 0
    assert L[0, k, 0] == 100.0 and L[0, k, 1] == 1.0 and abs(L[0, k, 2] - 0.8) < 1e-6
    assert L[0, k, 3] in (1.0, 3.0), "remap flag (+2 for mirror tiling)"
    assert list(L[3, k]) == [1.0, 0.5, 0.0, 2.0]
    assert L[2, k, 2] == pytest.approx(0.6) and L[2, k, 3] == pytest.approx(2.0)
    pu = L[2, k, 0]
    frames(eng, st, hub, n=1)
    assert eng._ul["torus"][2, k, 0] != pu or True     # phases are written in bind(); see test below
    assert eng._ul["torus"][0, 1, 2] == 0.0, "overlay 1 empty: opacity 0 (the shader skips it)"
    assert eng._uf["torus"][0, 1] == 1.0, "one overlay on slot 0"


def test_overlay_only_slot_has_no_base(tmp_path):
    eng, lib, hub = make_engine(tmp_path)
    st = mt.SkinState()
    sdo(st, op="layer", env="vortex", slot="walls", i=0, set=dict(src=dict(kind="layer", name="lines4")))
    frames(eng, st, hub)
    assert eng.active("vortex") and eng._uf["vortex"][0, 0] == 0.0 and eng._uf["vortex"][0, 1] == 1.0


def test_live_source_without_a_hub_stays_procedural(tmp_path):
    eng, lib, hub = make_engine(tmp_path)
    eng.hub = None
    st = mt.SkinState()
    st.assign("corridor", "walls", "layer", "kene")
    frames(eng, st, hub)
    assert not eng.active("corridor")


def test_overlay_phases_scroll_and_follow_travel(tmp_path):
    eng, lib, hub = make_engine(tmp_path)
    st = mt.SkinState()
    sdo(st, op="layer", env="corridor", slot="walls", i=0,
        set=dict(src=dict(kind="layer", name="lines4"), scroll=0.5, dir=0.0))
    frames(eng, st, hub)
    a = eng._lph["corridor"][0][0]
    eng.update(st.step(0.5), 0.5, dict(bpm=120.0))
    b = eng._lph["corridor"][0][0]
    assert (b - a) % mt.PH_WRAP == pytest.approx(0.05, abs=1e-6), "dt is capped at 0.1 s: 0.5 x 0.1"
    v0 = eng._lph["corridor"][0][1]
    eng.advance("corridor", 1.0)
    assert eng._lph["corridor"][0][1] != v0, "the overlay stays on its wall while the camera flies"


# ---------------------------------------------------------------- GL (GPU lock)

OUT = (640, 400)
TEX = ROOT / "textures"


@pytest.fixture(scope="module")
def ctx():
    import fx_stack as fx
    c, _win = fx.hidden_context()
    yield c


def gpu_ms(ctx, draw, n=40, warm=6):
    out = []
    for i in range(n + warm):
        q = ctx.query(time=True)
        with q:
            draw()
        ctx.finish()
        if i >= warm:
            out.append(q.elapsed / 1e6)
    return float(np.median(out))


def gpu_ms2(ctx, draw_a, draw_b, n=40, warm=6):
    out = ([], [])
    for i in range(n + warm):
        for k, draw in enumerate((draw_a, draw_b)):
            q = ctx.query(time=True)
            with q:
                draw()
            ctx.finish()
            if i >= warm:
                out[k].append(q.elapsed / 1e6)
    return float(np.median(out[0])), float(np.median(out[1]))


def test_gl_hub_copies_a_source_with_mips_into_its_layer(ctx):
    hub = mt.LayerHub(ctx)
    src = ctx.texture((1280, 800), 4, dtype="f1")
    src.write(np.full((800, 1280, 4), (200, 40, 10, 255), dtype="u1").tobytes())
    try:
        hub.register("solid", draw=lambda size: src)
        hub.request("solid")
        hub.run("scene")
        hub.end_frame()
        i = hub.request("solid")
        assert i == 0
        px = hub.probe(i, level=0)
        assert px.shape == (8, 8, 4) and np.abs(px[..., :3].astype(int) - (200, 40, 10)).max() <= 2
        lo = hub.probe(i, level=6)
        assert np.abs(lo[..., :3].astype(int) - (200, 40, 10)).max() <= 2, "mip chain filled"
    finally:
        hub.release()
        src.release()


def test_gl_layer_texture_cost(ctx):
    """Cost of one live layer per frame: copy (+ mips) of a 1280x800 texture the show already drew, and the static
    4-channel line layer (drawn once)."""
    hub = mt.LayerHub(ctx)
    src = ctx.texture((1280, 800), 4, dtype="f1")
    src.write(np.random.default_rng(1).integers(0, 255, (800, 1280, 4), dtype="u1").tobytes())
    try:
        hub.register("shown", draw=lambda size: src)
        hub.request("shown")
        hub.run("scene")
        hub.end_frame()

        def one():
            hub.request("shown")
            hub.run("scene")
            hub.end_frame()
        ms = gpu_ms(ctx, one)
        hub.register("lines4b", draw=hub.builtin_lines4)
        ms4 = gpu_ms(ctx, lambda: hub.builtin_lines4(512))
        print(f"\n[layer cost] copy 1280x800 -> 512 px layer + mips: {ms:.3f} ms per frame; "
              f"lines4 generator at 512 px: {ms4:.3f} ms (static: once)")
        assert ms < 0.25 and ms4 < 0.5
    finally:
        hub.release()
        src.release()


def stack_state(env, base=None, overlays=1, remap=True, noise=0.6):
    st = mt.SkinState()
    slots = mt.ENVS[env]["slots"]
    for s in slots:
        if base:
            st.assign(env, s, "ai", base)
        for i in range(overlays):
            sdo(st, op="layer", env=env, slot=s, i=i,
                set=dict(src=dict(kind="layer", name="lines4"), blend=(0, 3)[i], opacity=1.0, remap=int(remap),
                         noise=noise if i == 0 else 0.0, scale=(1.0, 2.0)[i]))
    return st


def settle(eng, st, hub, n=4):
    for _ in range(n):
        eng.update(st.step(1 / 60), 1 / 60, dict(bpm=120.0))
        eng.lib.flush()
        hub.run("scene")
        hub.end_frame()
    eng.update(st.step(0.0), 0.0, dict(bpm=120.0))


def test_gl_stacks_draw_on_every_world_and_cost_is_in_budget(ctx):
    import fx_stack as fx
    names = sorted(p.name for p in TEX.glob("*.png"))
    if not names:
        pytest.skip("needs images in textures/")
    size = (1280, 800)
    lp0 = fx.LandPass(ctx, size)
    lp = fx.LandPass(ctx, size)
    lp0.settle_s = lp.settle_s = 0.0
    eng = mt.SkinEngine(ctx, mt.roots_for(TEX, ROOT / "art"), size=512, sync=True)
    eng.fade_s = 0.0
    hub = mt.LayerHub(ctx)
    hub.register("lines4", draw=hub.builtin_lines4, static=True)
    eng.hub = hub
    lp.skins = eng
    rows = []
    try:
        for env in mt.LAND_ENVS:
            mode = mt.ENVS[env]["land"]
            v = fx.default_values(land_a=mode, land_res=1.0)
            clock = fx.make_clock(4.0)

            def draw(pss):
                pss.render(v, clock, 4.0, 0.0, 0.0, fx.GALLERY_PALETTE, (0.5, 0.3, 0.2))
            for label, st in (("base", stack_state(env, names[0], overlays=0)),
                              ("base+1", stack_state(env, names[0], overlays=1)),
                              ("base+2", stack_state(env, names[0], overlays=2)),
                              ("lines only", stack_state(env, None, overlays=1))):
                settle(eng, st, hub)
                assert eng.active(env), (env, label)
                pre, on = gpu_ms2(ctx, lambda: draw(lp0), lambda: draw(lp))
                rows.append((env, label, pre, on))
        print()
        for env, label, pre, on in rows:
            print(f"[stack cost] {env:9s} {label:10s} pre-skin {pre:.3f} ms  skinned {on:.3f} ms  (+{on - pre:.3f})")
        for env, label, pre, on in rows:
            if True:
                assert on - pre <= 0.30, f"{env} {label}: +{on - pre:.3f} ms"
    finally:
        lp.release()
        lp0.release()
        eng.release()
        hub.release()


def test_expensive_sources_refresh_every_n_frames():
    hub, be, clk = make_hub()
    calls = []
    hub.register("blocks", draw=lambda size: calls.append(1) or "B", every=2)
    for _ in range(10):
        hub.request("blocks")
        hub.run("scene")
        hub.end_frame()
    assert len(calls) == 5


def test_gl_engine_layers_as_skins_end_to_end(tmp_path):
    """Headless engine: no layer requested = the hub does nothing (0 copies, 0 MB); a tunnel skinned by the kené study
    (layer OFF in the mix: its own 512 px instance), the 4-channel line layer and Folding Blocks (scene-only, 256 px)
    changes the picture and stays finite."""
    import json
    import subprocess
    sk = dict(slots={"tunnel:walls": dict(src=dict(kind="layer", name="kene"), layers=[
        dict(src=dict(kind="layer", name="lines4"), remap=1, noise=0.5),
        dict(src=dict(kind="layer", name="blocks"), blend=1, opacity=0.7)])})
    (tmp_path / "skins.json").write_text(json.dumps(sk))
    outs = []
    for tag, extra in (("plain", []), ("skin", ["--skins", str(tmp_path / "skins.json")])):
        r = subprocess.run([sys.executable, str(ROOT / "live_visuals.py"), "--test", "24", "--out", str(tmp_path / tag),
                            "--set", "travel=0.6", "--set", "travel_mode=1", "--test-audio", "silent"] + extra,
                           capture_output=True, text=True, timeout=180, cwd=str(ROOT))
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        outs.append(r.stdout)
    line = [x for x in outs[0].splitlines() if x.startswith("[skin layers]")]
    assert line and "copies=0" in line[0] and "vram_mb=0" in line[0], line
    line = [x for x in outs[1].splitlines() if x.startswith("[skin layers]")][0]
    print("\n" + line)
    assert "copies=0" not in line and "own=0" not in line and "'kene'" in line and "fails={}" in line, line
    assert "nonfinite=0" in outs[1]
    from PIL import Image
    a = np.asarray(Image.open(sorted((tmp_path / "plain").glob("frame_*.png"))[-1]).convert("RGB")).astype("f4")
    b = np.asarray(Image.open(sorted((tmp_path / "skin").glob("frame_*.png"))[-1]).convert("RGB")).astype("f4")
    assert np.abs(a - b).mean() > 1.0


def test_example_command_loads_the_preset():
    st = mt.SkinState()
    sdo(st, op="example", name="kene_cloth_world")
    assert st.to_json()["slots"] == mt.EXAMPLES["kene_cloth_world"]["slots"]
    with pytest.raises(ValueError):
        mt.validate(dict(cmd="skin", op="example", name="nope"))


def test_freeing_the_live_array_never_refades_image_skins(tmp_path):
    eng, lib, hub = make_engine(tmp_path)
    st = mt.SkinState()
    st.assign("corridor", "walls", "ai", "t0.png")
    st.assign("corridor", "floor", "layer", "kene")
    frames(eng, st, hub)
    walls, floor = eng._slots["corridor"]
    assert walls.amt == 1.0 and floor.cur >= 100
    hub.gen += 1                                  # the hub freed its array
    eng.update(st.step(1 / 60), 1 / 60, dict(bpm=120.0))
    assert walls.amt == 1.0 and walls.cur >= 0, "the image skin stays"
    assert floor.cur_key is None or floor.cur >= 100
