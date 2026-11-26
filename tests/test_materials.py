"""Environment skins (materials.py): model, commands, persistence, tiling maths, wrap, crossfade, library budget,
governor, generate-a-skin, and (GL, names start with test_gl_) pixel identity, cost, black floor, state restore.

The CPU tests use a fake GL context, so `pytest tests/test_materials.py -k "not gl_"` needs no GPU at all.
"""
import json
import math
import random
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import control  # noqa: E402
import materials as mt  # noqa: E402
import shows  # noqa: E402
from control import Controller  # noqa: E402


# ---------------------------------------------------------------- helpers


class FakeTime:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(tmp_path, skins_on=True, **kw):
    ft = FakeTime()
    kw.setdefault("restore", False)
    kw.setdefault("autosave", True)
    c = Controller(tmp_path, midi=False, ai=False, time_fn=ft, seed=1, **kw)
    c.ft = ft
    c.quantize = "off"
    c.morph_bars = 0.0
    if skins_on and c.skins is not None and not c.skins.on:     # the feature ships OFF; most tests want it on
        c.apply([dict(cmd="skin", op="enable", on=True)])
    return c


def run(c, seconds, dt=1 / 60, bands=(0.6, 0.4, 0.2)):
    for _ in range(int(round(seconds / dt))):
        c.ft.t += dt
        c.update(dt, bands)


def do(c, **cmd):
    rest = [r for r in c.apply([cmd]) if r.get("cmd") != "snapshot"]
    assert rest == []


def sdo(st, **cmd):
    return st.do(mt.validate(dict(cmd, cmd="skin")))


class FakeTex:
    def __init__(self, size):
        self.size, self.writes, self.mips, self.released, self.used = size, [], 0, False, []
        self.repeat_x = self.repeat_y = False
        self.filter = None
        self.anisotropy = 1.0

    def write(self, data, viewport=None):
        self.writes.append((len(data), viewport))

    def build_mipmaps(self):
        self.mips += 1

    def release(self):
        self.released = True

    def use(self, unit):
        self.used.append(unit)


class FakeCtx:
    def __init__(self):
        self.made = []

    def texture_array(self, size, components, dtype="f1"):
        assert components == 4 and dtype == "f1", "skins are RGBA8, never float"
        t = FakeTex(size)
        self.made.append(t)
        return t


def fake_loader(path, size):
    return bytes(size * size * 4), 1.0


def make_lib(tmp_path, n=12, size=512, **kw):
    d = tmp_path / "textures"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"t{i}.png").write_bytes(b"x")
    ctx = FakeCtx()
    lib = mt.SkinLibrary(ctx, dict(ai=d, art=tmp_path / "art"), size=size, loader=fake_loader, sync=True, **kw)
    return lib, ctx


def make_engine(tmp_path, **kw):
    lib, ctx = make_lib(tmp_path, **kw)
    return mt.SkinEngine(None, None, lib=lib), lib, ctx


def frame_with(**slots):
    st = mt.SkinState()
    for key, name in slots.items():
        env, slot = key.split("__")
        st.assign(env, slot, "ai", name)
    return st


# ---------------------------------------------------------------- model + commands


def test_default_is_all_procedural_and_empty():
    st = mt.SkinState()
    assert st.frame == {} and not st.any()
    assert st.to_json() == dict(v=1, slots={}, mods={})
    assert st.step(1 / 60) == {}
    assert len(json.dumps(st.state())) < 60, "no skins must cost next to nothing in the panel state"
    for env, e in mt.ENVS.items():
        # worlds and the tunnel offer >= 3 mappings; a creature / body host maps along its own body (1 is enough)
        assert 1 <= len(e["slots"]) <= mt.MAX_SLOTS and len(e["maps"]) >= (3 if e["host"] in ("land", "scene") else 1)
        for s in e["slots"]:
            assert st.slots[env][s]["src"] is None


def test_commands_are_validated():
    ok = mt.validate(dict(cmd="skin", op="assign", env="corridor", slot="walls", kind="ai", name="tex_1.png"))
    assert ok == dict(cmd="skin", op="assign", env="corridor", slot="walls", kind="ai", name="tex_1.png")
    assert mt.validate(ok) == ok, "validation must be idempotent"
    bad = [
        dict(op="assign", env="nowhere", slot="walls", kind="ai", name="a.png"),
        dict(op="assign", env="corridor", slot="ceiling", kind="ai", name="a.png"),
        dict(op="assign", env="corridor", slot="walls", kind="web", name="a.png"),
        dict(op="assign", env="corridor", slot="walls", kind="ai", name="../secret.png"),
        dict(op="assign", env="corridor", slot="walls", kind="ai", name="C:\\x\\a.png"),
        dict(op="assign", env="corridor", slot="walls", kind="ai", name="a.exe"),
        dict(op="assign", env="corridor", slot="walls", kind="ai", name=""),
        dict(op="set", env="corridor", slot="walls", f="nope", v=1),
        dict(op="set", env="corridor", slot="walls", f="scale", v="big"),
        dict(op="set", env="corridor", slot="walls", f="scale", v=float("nan")),
        dict(op="set", env="corridor", slot="walls", f="tint", v=[1, 2]),
        dict(op="mod", env="corridor", slot="walls", f="map", src="bass", amt=1),
        dict(op="gen", env="corridor", slot="walls", prompt=""),
        dict(op="gen", env="corridor", slot="walls", prompt="x" * 500),
        dict(op="explode"),
        "skin",
    ]
    for c in bad:
        with pytest.raises(ValueError):
            mt.validate(dict(c, cmd="skin") if isinstance(c, dict) else c)
    c = mt.validate(dict(cmd="skin", op="set", env="tunnel", slot="walls", set=dict(scale=99, rot=-999, map=7, mix=2.2)))
    assert c["set"] == dict(scale=8.0, rot=-180.0, map=3, mix=2)
    c = mt.validate(dict(cmd="skin", op="set", env="vortex", slot="walls", f="map", v=9))
    assert c["set"]["map"] == len(mt.DECK_MAPS) - 1


def test_assign_set_clear_reset_and_round_trip():
    st = mt.SkinState()
    sdo(st, op="assign", env="corridor", slot="walls", kind="ai", name="tex_1.png")
    sdo(st, op="set", env="corridor", slot="walls", set=dict(scale=2.5, mix=4, tint=[1, 0.5, 0.25]))
    sdo(st, op="assign", env="tunnel", slot="walls", kind="art", name="my painting (2).jpg")
    assert set(st.frame) == {"corridor", "tunnel"}
    j = st.to_json()
    assert j["slots"]["corridor:walls"] == dict(scale=2.5, mix=4, tint=[1.0, 0.5, 0.25], src=dict(kind="ai", name="tex_1.png"))
    st2 = mt.SkinState()
    st2.load(json.loads(json.dumps(j)))
    assert st2.to_json() == j and st2.frame["corridor"]["walls"]["scale"] == 2.5
    sdo(st, op="clear", env="corridor", slot="walls")
    assert "corridor" not in st.frame and st.slots["corridor"]["walls"]["scale"] == 2.5, "clear keeps the slot's settings"
    sdo(st, op="reset", env="corridor")
    assert st.slots["corridor"]["walls"] == mt.default_slot()
    sdo(st, op="reset")
    assert st.to_json() == dict(v=1, slots={}, mods={})
    sdo(st, op="replace", skins=j)
    assert st.to_json() == j


def test_junk_never_survives_cleaning():
    j = mt.clean_skins(dict(slots={"corridor:walls": dict(src=dict(kind="ai", name="../../x.png"), scale="x", junk=1),
                                   "moon:walls": dict(scale=2), 7: 1, "tunnel:walls": "no"},
                            mods={"corridor:walls:scale": [dict(src="bass", amt=5), "x"], "corridor:walls:map": [dict(src="bass", amt=1)],
                                  "bad": 1}))
    assert j["slots"] == {} and list(j["mods"]) == ["corridor:walls:scale"] and j["mods"]["corridor:walls:scale"][0]["amt"] == 1.0
    assert mt.clean_skins("nonsense") == dict(v=1, slots={}, mods={})


def test_targets_follow_the_stack_naming_style():
    assert mt.parse_target("skin:corridor:walls:scale") == ("corridor", "walls", "scale")
    assert mt.parse_target("skin:corridor:walls:nope") is None and mt.parse_target("stack:1:speed") is None
    assert control.valid_target("skin:tunnel:walls:scroll")
    assert not control.valid_target("skin:tunnel:floor:scroll")
    assert "skin" in control.CONTROL_CMDS


# ---------------------------------------------------------------- controller: commands, persistence, scenes, shows


def test_controller_takes_skin_commands_and_publishes_them(tmp_path):
    c = make(tmp_path)
    assert c.state()["skins"]["slots"] == {}
    do(c, cmd="skin", op="assign", env="torus", slot="tube", kind="ai", name="gen_a.png")
    do(c, cmd="skin", op="set", env="torus", slot="tube", f="scroll", v=0.5)
    run(c, 0.1)
    assert c.state()["skins"]["slots"]["torus:tube"]["src"]["name"] == "gen_a.png"
    assert c.skins.frame["torus"]["tube"]["scroll"] == 0.5
    assert control.meta()["skins"]["envs"][0]["id"] == "corridor"
    with pytest.raises(ValueError):
        control.validate(dict(cmd="skin", op="assign", env="torus", slot="tube", kind="ai", name="a/b.png"))


def test_skins_persist_in_live_state(tmp_path):
    c = make(tmp_path)
    do(c, cmd="skin", op="assign", env="hexdome", slot="domes", kind="art", name="piece.jpg")
    do(c, cmd="skin", op="set", env="hexdome", slot="domes", set=dict(tile=0, bright=1.5))
    c._autosave()
    saved = json.loads((tmp_path / "live_state.json").read_text())["settings"]["skins"]
    assert saved["slots"]["hexdome:domes"]["src"] == dict(kind="art", name="piece.jpg")
    c2 = make(tmp_path, restore=True)
    assert c2.skins.to_json() == c.skins.to_json() and c2.skins.frame["hexdome"]["domes"]["bright"] == 1.5


def test_scenes_store_and_recall_skins_and_old_scenes_leave_them_alone(tmp_path):
    c = make(tmp_path)
    do(c, cmd="skin", op="assign", env="corridor", slot="walls", kind="ai", name="a.png")
    c.scene_save(9)
    assert c.scenes[9]["skins"]["slots"]["corridor:walls"]["src"]["name"] == "a.png"
    do(c, cmd="skin", op="assign", env="corridor", slot="walls", kind="ai", name="b.png")
    c.scene_recall(9, now=True)
    run(c, 0.1)
    assert c.skins.slots["corridor"]["walls"]["src"]["name"] == "a.png"
    do(c, cmd="skin", op="assign", env="corridor", slot="floor", kind="ai", name="b.png")
    c.scene_recall(0, now=True)                       # a factory scene has no "skins" key
    run(c, 0.1)
    assert c.skins.slots["corridor"]["floor"]["src"]["name"] == "b.png"
    assert control._clean_scene(dict(base={}, skins=dict(slots={"x:y": 1})))["skins"] == dict(v=1, slots={}, mods={})


def test_show_files_carry_skins(tmp_path):
    c = make(tmp_path)
    do(c, cmd="skin", op="assign", env="vortex", slot="walls", kind="ai", name="a.png")
    st = shows.clean_settings(c._settings())
    assert st["skins"]["slots"]["vortex:walls"]["src"]["name"] == "a.png"


def test_panic_returns_every_skin_to_procedural_but_keeps_the_settings(tmp_path):
    c = make(tmp_path)
    do(c, cmd="skin", op="assign", env="tunnel", slot="walls", kind="ai", name="a.png")
    do(c, cmd="skin", op="set", env="tunnel", slot="walls", f="scale", v=3)
    c.panic()
    assert not c.skins.any() and c.skins.slots["tunnel"]["walls"]["scale"] == 3.0


def test_panic_cuts_skins_without_the_stack_module_and_stops_the_ai_job(tmp_path):
    """Review #6: independent of stack.py, a cut (not a 0.4 s fade), and an in-flight generation is cancelled."""
    c = make(tmp_path)
    ai = FakeAI()
    ai.cancelled = 0
    ai.cancel = lambda: setattr(ai, "cancelled", ai.cancelled + 1)
    c.ai = ai
    c.stack = None                                       # as if stack.py had failed to import
    do(c, cmd="skin", op="assign", env="corridor", slot="walls", kind="ai", name="t0.png")
    do(c, cmd="skin", op="gen", env="torus", slot="tube", prompt="neon moss")
    run(c, 0.2)
    assert len(ai.jobs) == 1 and c.skins.gen["sent"]
    eng, lib, _ = make_engine(tmp_path / "e")
    for _ in range(60):
        eng.update(c.skins.step(1 / 60), 1 / 60, None)
    assert eng._u["corridor"][0, 0, 3] == 1.0
    cut0 = c.skins.cut
    c.panic()
    assert not c.skins.any() and c.skins.gen is None and ai.cancelled == 1 and c.skins.cut == cut0 + 1
    eng.update(c.skins.step(1 / 60), 1 / 60, None, snap=c.skins.cut != cut0)
    assert eng._u["corridor"][0, 0, 3] == 0.0 and not eng.any_active(), "gone on the very next frame"
    src = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    assert '_cut != sk_st["cut"]' in src


class FakeUniform:
    def __init__(self, n):
        self.array_length, self.data = n, None

    def write(self, a):
        self.data = np.array(a, dtype="f4").copy()


class FakeProg(dict):
    def __init__(self, n=2):
        super().__init__({k: FakeUniform(n) for k in ("skA", "skB", "skC", "skD", "skE")})


def test_the_bound_phase_is_this_frames_phase(tmp_path):
    """Review #5: update -> advance -> bind. The uniform carries the travel of THIS frame (no one-frame lag)."""
    eng, lib, ctx = make_engine(tmp_path)
    st = frame_with(corridor__walls="t0.png")
    steps(eng, st, 40)
    sl = eng._slots["corridor"][0]
    prog = FakeProg()
    for dz in (0.13, 0.31, 0.07):
        eng.update(st.step(1 / 60), 1 / 60, None)
        eng.advance("corridor", dz)
        eng.bind(prog, "corridor")
        assert prog["skD"].data[0, 1] == np.float32(sl.pv) and prog["skD"].data[0, 0] == np.float32(sl.pu)
    assert ctx.made[0].used and ctx.made[0].used[-1] == mt.UNIT


def test_engine_hooks_keep_gl_out_of_the_guard_and_compile_once():
    """Review #7 + #8 (source level: the hooks live inside the engine's main loop)."""
    src = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    assert 'guarded("skins", skins.bind' not in src and 'guarded("skins", skin_scene' not in src
    assert "skins.gl_step(" in src and 'guarded("skins", skins.gl_step' not in src
    i = src.index("def skin_scene():")
    body = src[i:src.index("def skins_ready():", i)]
    assert "scene_skin = False" in body and "if scene_skin is None:" in body and "return scene_skin or None" in body
    assert "skin_scene() or scene_plain" in src, "a failed compile falls back to the plain scene program"
    # update() itself does no GL when asked not to
    assert "def gl_step" in (ROOT / "materials.py").read_text(encoding="utf-8")


def test_update_without_gl_never_touches_the_texture(tmp_path):
    eng, lib, ctx = make_engine(tmp_path)
    st = frame_with(corridor__walls="t0.png")
    for _ in range(30):
        eng.update(st.step(1 / 60), 1 / 60, None, gl=False)
    assert ctx.made == [] and lib.request(("ai", "t0.png")) == -1, "logic only: no allocation, no upload"
    eng.gl_step(False)
    assert len(ctx.made) == 1


def test_mod_matrix_and_midi_reach_skin_fields(tmp_path):
    c = make(tmp_path)
    do(c, cmd="skin", op="assign", env="corridor", slot="walls", kind="ai", name="a.png")
    do(c, cmd="mod", name="skin:corridor:walls:bright", slot=0, src="bass", amt=0.5, rate=1, phase=0)
    assert c.skins.mods["corridor:walls:bright"][0]["src"] == "bass"
    run(c, 0.5, bands=(1.0, 0.0, 0.0))
    assert c.skins.frame["corridor"]["walls"]["bright"] > 1.5, "bass x 0.5 x range 2 on top of 1.0"
    assert c.skins.slots["corridor"]["walls"]["bright"] == 1.0, "modulation never edits the stored value"
    with pytest.raises(ValueError):
        control.validate(dict(cmd="mod", name="skin:corridor:walls:mix", slot=0, src="bass", amt=1))
    c.skins.midi("corridor", "walls", "scale", 1.0)
    assert c.skins.slots["corridor"]["walls"]["scale"] == 8.0
    c.skins.midi("corridor", "walls", "mix", 1.0, pad=True, press=True)
    assert c.skins.slots["corridor"]["walls"]["mix"] == 1


def test_midi_on_stepped_fields_cc_is_absolute_encoder_steps_pad_cycles():
    """Review #3."""
    st = mt.SkinState()
    w = st.slots["corridor"]["walls"]
    w["mix"] = 4
    st.midi("corridor", "walls", "mix", 1.0)             # CC 127 with the knob already at the top: stays there
    assert w["mix"] == 4
    w["mix"] = 2
    st.midi("corridor", "walls", "mix", 0.0)             # CC 0: the bottom, not "one step up"
    assert w["mix"] == 0
    st.midi("corridor", "walls", "mix", 0.5)
    assert w["mix"] == 2
    w["map"] = 1
    st.midi("corridor", "walls", "map", 1 / 127, rel=1)  # relative encoder: one step per click, clamped
    assert w["map"] == 2
    st.midi("corridor", "walls", "map", 1 / 127, rel=1)
    assert w["map"] == 2
    st.midi("corridor", "walls", "map", 127 / 127, rel=-1)
    assert w["map"] == 1
    st.midi("corridor", "walls", "tile", 1.0, pad=True, press=False)
    assert w["tile"] == 1
    st.midi("corridor", "walls", "tile", 1.0, pad=True, press=True)      # a pad steps and wraps
    assert w["tile"] == 0
    # the controller's MIDI path: only a NOTE is a pad
    src = (ROOT / "control.py").read_text(encoding="utf-8")
    assert 'self.skins.midi(t[0], t[1], t[2], 1.0, pad=True, press=edge)' in src
    assert 'self.skins.midi(t[0], t[1], t[2], val / 127.0, rel=rel)' in src


def test_mod_slot_and_source_are_validated():
    """Review #9."""
    base = dict(cmd="skin", op="mod", env="corridor", slot="walls", f="scale", src="bass", amt=0.5)
    assert mt.validate(dict(base, mslot=1))["mslot"] == 1
    for bad in (1.0, True, "1", 2, None):
        with pytest.raises(ValueError):
            mt.validate(dict(base, mslot=bad))
    with pytest.raises(ValueError):
        mt.validate(dict(base, mslot=0, src="not_a_source"))
    st = mt.SkinState()
    sdo(st, **{k: v for k, v in dict(base, mslot=1).items() if k != "cmd"})
    assert st.mods["corridor:walls:scale"][1]["src"] == "bass"
    j = mt.clean_skins(dict(mods={"corridor:walls:scale": [dict(src="not_a_source", amt=1), dict(src="mid", amt=0.2)]}))
    assert j["mods"]["corridor:walls:scale"] == [None, dict(src="mid", amt=0.2, rate=1.0, phase=0.0)], "junk sources never persist"
    st.load(dict(mods={"corridor:walls:scale": [dict(src="nope", amt=1), None]}))
    assert st.mods == {}


# ---------------------------------------------------------------- ships inactive (PR review B21)


def test_skins_ship_inactive_and_switch_on_explicitly(tmp_path):
    c = make(tmp_path, skins_on=False)
    assert c.skins.on is False and c.state()["skins"]["on"] is False
    do(c, cmd="skin", op="assign", env="corridor", slot="walls", kind="ai", name="a.png")
    run(c, 0.1)
    assert c.skins.frame == {} and not c.skins.any(), "off: whatever the slots hold, the engine is given nothing"
    do(c, cmd="skin", op="gen", env="corridor", slot="walls", prompt="p")
    assert c.skins.gen is None and "off" in c.skins.gen_note, "off: no generation job either"
    do(c, cmd="skin", op="enable", on=True)
    assert c.skins.frame["corridor"]["walls"]["src"]["name"] == "a.png", "the slot kept what it held"
    c._autosave()
    assert json.loads((tmp_path / "live_state.json").read_text())["settings"]["skins_on"] is True
    c2 = make(tmp_path, skins_on=False, restore=True)
    assert c2.skins.on is True and c2.skins.any()
    do(c2, cmd="skin", op="enable", on=False)
    assert c2.skins.frame == {} and c2.skins.slots["corridor"]["walls"]["src"]["name"] == "a.png"
    with pytest.raises(ValueError):
        mt.validate(dict(cmd="skin", op="enable", on="yes"))
    assert "skins_on" not in shows.clean_settings(c._settings()), "a show file never switches the feature on by itself"


def test_switching_skins_off_fades_out_and_frees_the_array(tmp_path):
    eng, lib, ctx = make_engine(tmp_path)
    st = frame_with(corridor__walls="t0.png")
    steps(eng, st, 60)
    assert eng.vram_bytes() > 0
    st.enable(False)
    eng.enabled = False
    rows = steps(eng, st, 60)
    assert rows[0][3] > 0.9 and rows[-1][3] == 0.0, "a fade, not a cut"
    assert ctx.made[0].released and eng.vram_bytes() == 0 and eng.idle()
    st.enable(True)
    eng.enabled = True
    rows = steps(eng, st, 60)
    assert rows[-1][3] == 1.0 and len(ctx.made) == 2


def test_engine_compiles_nothing_for_skins_until_they_are_on():
    fx_src = (ROOT / "fx_stack.py").read_text(encoding="utf-8")
    init = fx_src[fx_src.index("class LandPass"):fx_src.index("def skins(self, engine):")]
    assert "skinned_source(" not in init, "LandPass.__init__ compiles only the original programs"
    lv = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    assert lv.count("_land.skins = skins") == 1 and "def skins_ready():" in lv
    body = lv[lv.index("def skins_ready():"):lv.index("# <skins]", lv.index("def skins_ready():"))]
    assert "_land.skins = skins" in body and "skin_scene()" in body
    assert "SkinState(on=False)" in (ROOT / "control.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------- tiling maths


def test_mirror_fold_is_continuous_everywhere_and_has_period_two():
    for u in np.linspace(-7.3, 9.1, 2000):
        assert abs(mt.fold(u + 1e-6, 1) - mt.fold(u, 1)) < 1e-5, "mirror tiling has no jump anywhere"
        assert abs(mt.fold(u + 2.0, 1) - mt.fold(u, 1)) < 1e-9
        assert abs(mt.fold(u + 1.0, 0) - mt.fold(u, 0)) < 1e-9
    assert mt.fold(0.25, 1) == mt.fold(-0.25, 1) == mt.fold(1.75, 1) == 0.25


def test_any_scale_and_rotation_closes_round_a_cylinder():
    """Going once round the axis must move uv by whole periods (even ones for mirror tiling): no seam at the atan cut."""
    rng = random.Random(3)
    for _ in range(3000):
        tile = rng.randint(0, 1)
        m = mt.uv_matrix("cyl", rng.choice([3.0, 4.0, 5.5]), rng.uniform(0.1, 2.0), rng.uniform(0.25, 8.0), rng.uniform(-180, 180), tile)
        z = rng.uniform(-50, 50)
        for th in (-0.5, 0.123):
            u0, v0 = m[0] * th + m[1] * z, m[2] * th + m[3] * z
            u1, v1 = m[0] * (th + 1.0) + m[1] * z, m[2] * (th + 1.0) + m[3] * z
            assert abs(mt.fold(u0, tile) - mt.fold(u1, tile)) < 1e-6 or abs(abs(mt.fold(u0, tile) - mt.fold(u1, tile)) - 1.0) < 1e-6
            assert abs(mt.fold(v0, tile) - mt.fold(v1, tile)) < 1e-6 or abs(abs(mt.fold(v0, tile) - mt.fold(v1, tile)) - 1.0) < 1e-6
        period = 2 if tile else 1
        assert m[0] % period == 0 and m[2] % period == 0 and (m[0], m[2]) != (0.0, 0.0)


def test_plane_matrix_is_scale_times_rotation():
    m = mt.uv_matrix("plane", 1.0, 1.0, 2.0, 90.0, 1)
    assert np.allclose(m, (0.0, -2.0, 2.0, 0.0), atol=1e-9)
    assert np.allclose(mt.uv_matrix("plane", 0.5, 0.5, 1.0, 0.0, 0), (0.5, 0.0, 0.0, 0.5))


# ---------------------------------------------------------------- library: budget, chunking, LRU (fake GL)


def test_library_is_one_rgba8_array_with_bounded_vram(tmp_path):
    lib, ctx = make_lib(tmp_path, n=12, size=1024)
    assert ctx.made == [] and lib.vram_bytes() == 0, "no skin used: no texture, no VRAM"
    for i in range(12):                                 # far more images than layers
        lib.request(("ai", f"t{i}.png"))
        lib.flush()
    assert len(ctx.made) == 1 and ctx.made[0].size == (1024, 1024, mt.MAX_LAYERS), "one array, whatever is requested"
    assert ctx.made[0].repeat_x and ctx.made[0].anisotropy == 4.0
    budget = lib.vram_bytes()
    assert budget == mt.vram_bytes(1024) and budget < 46 * 1024 * 1024
    assert mt.vram_bytes(512) < 12 * 1024 * 1024
    assert lib.status()["resident"] == mt.MAX_LAYERS and lib.evictions == 4
    assert lib.status()["vram_mb"] == round(budget / 1048576, 1)


def test_uploads_are_chunked_and_never_on_a_late_frame(tmp_path):
    """The moderngl-only fallback path (no raw GL): level 0 in strips, then one mip build."""
    lib, ctx = make_lib(tmp_path, size=1024)
    assert lib.request(("ai", "t0.png")) == -1
    for _ in range(50):
        assert lib.tick(late=True) is False
    assert ctx.made == [] and lib.deferred == 50, "a late frame gets no upload work at all (not even the allocation)"
    steps = 0
    while lib.request(("ai", "t0.png")) < 0:
        lib.tick(late=False)
        steps += 1
        assert steps < 20
    tex = ctx.made[0]
    assert all(n <= lib.STRIP_BYTES for n, _ in tex.writes) and len(tex.writes) == 4 and tex.mips == 2   # 1 at allocation
    assert [vp[1] for _, vp in tex.writes] == [0, 256, 512, 768] and all(vp[2] == 0 and vp[5] == 1 for _, vp in tex.writes)
    assert lib.tick(False) is False, "nothing pending: tick does nothing"


class FakeRaw:
    """Stands in for materials.RawGL (glTexImage3D / glTexSubImage3D through ctypes)."""

    def __init__(self):
        self.allocs, self.subs = [], []

    def alloc_levels(self, glo, size, layers):
        self.allocs.append((size, layers))
        return True

    def sub(self, glo, level, y, layer, width, rows, arr, offset):
        assert arr.shape == (width, width, 4) and arr.dtype == np.uint8 and offset == y * width * 4
        self.subs.append((level, y, layer, width, rows))


@pytest.mark.parametrize("size, ticks", [(1024, 6), (512, 2)])
def test_mips_come_from_the_loader_thread_and_are_uploaded_per_level(tmp_path, size, ticks):
    """Review #2: never glGenerateMipmap over all 8 layers. Every level of ONE layer is uploaded, <= 1 MB per frame."""
    raw = FakeRaw()
    lib, ctx = make_lib(tmp_path, size=size, raw=raw)
    lib.request(("ai", "t0.png"))
    per_tick = []
    while lib.request(("ai", "t0.png")) < 0:
        n = len(raw.subs)
        assert lib.tick(False)
        per_tick.append(sum(w * rows * 4 for _, _, _, w, rows in raw.subs[n:]))
        assert len(per_tick) < 20
    tex = ctx.made[0]
    tex.glo = 1
    assert raw.allocs == [(size, mt.MAX_LAYERS)] and tex.mips == 0 and tex.writes == [], "no mip generation on the render thread"
    assert len(per_tick) == ticks and max(per_tick) <= lib.STRIP_BYTES
    levels = sorted({lv for lv, *_ in raw.subs})
    assert levels == list(range(int(np.log2(size)) + 1)), "the whole chain down to 1 x 1"
    for lv in levels:                                    # every row of every level, on the one layer
        rows = sum(r for l2, _, lay, _, r in raw.subs if l2 == lv)
        assert rows == size >> lv and all(lay == 0 for _, _, lay, _, _ in raw.subs)


def test_mip_chain_is_a_box_filter():
    a = np.zeros((4, 4, 4), dtype="u1")
    a[:2, :2] = 200
    chain = mt.mip_chain(a)
    assert [c.shape[0] for c in chain] == [4, 2, 1]
    assert chain[1][0, 0, 0] == 200 and chain[1][1, 1, 0] == 0 and chain[2][0, 0, 0] == 50


def test_a_stale_decode_never_cancels_the_new_request(tmp_path):
    """Review #4a: a result decoded for the old array must not delete the pending entry of the new request."""
    calls = []

    def loader(path, size):
        calls.append((path.name, size))
        return bytes(size * size * 4), 1.0

    d = tmp_path / "textures"
    d.mkdir()
    (d / "t0.png").write_bytes(b"x")
    ctx = FakeCtx()
    lib = mt.SkinLibrary(ctx, dict(ai=d, art=tmp_path / "art"), size=1024, loader=loader, sync=True)
    key = ("ai", "t0.png")
    lib.request(key)                                     # decoded for 1024 (result waits in the queue)
    lib.set_size(512)
    lib.request(key)                                     # asked again for 512
    assert calls == [("t0.png", 1024), ("t0.png", 512)]
    lib.tick(False)                                      # the stale 1024 result arrives first
    assert key in lib._pending, "the 512 request is still pending"
    assert lib.request(key) == -1 and len(calls) == 2, "no duplicate decode"
    lib.flush()
    assert lib.request(key) >= 0 and lib.status()["resident"] == 1 and len(ctx.made) == 1 and ctx.made[0].size[0] == 512


def test_an_image_nobody_wants_any_more_is_not_uploaded(tmp_path):
    """Review #4b: tapping through the strip must not upload (or evict for) pictures that were skipped past."""
    lib, ctx = make_lib(tmp_path, n=4)
    for i in range(3):
        lib.request(("ai", f"t{i}.png"))
    wanted = {("ai", "t2.png")}
    for _ in range(10):
        lib.tick(False, (), wanted)
    assert lib.skipped == 2 and lib.status()["resident"] == 1 and lib.request(("ai", "t2.png")) >= 0
    assert lib.request(("ai", "t0.png")) == -1
    assert len(ctx.made[0].writes) == 1, "only the wanted image reached the GPU"
    # the engine passes its wanted set: a slot that moved on before the decode landed
    eng, lib2, ctx2 = make_engine(tmp_path / "e", size=1024)
    st = frame_with(corridor__walls="t0.png")
    eng.update(st.step(1 / 60), 1 / 60, None, late=True)     # requested, result waits (late frame)
    st.assign("corridor", "walls", "ai", "t1.png")
    steps(eng, st, 40)
    assert lib2.request(("ai", "t0.png")) == -1 and lib2.skipped == 1 and lib2.status()["resident"] == 1


def test_a_permanently_late_show_still_gets_its_skin(tmp_path, monkeypatch):
    lib, ctx = make_lib(tmp_path, size=512)
    now = [1000.0]
    monkeypatch.setattr(mt.time, "monotonic", lambda: now[0])
    lib.request(("ai", "t0.png"))
    assert lib.tick(late=True) is False
    now[0] += lib.STARVE_S + 0.1
    assert lib.tick(late=True) is True


def test_lru_never_evicts_what_is_on_screen(tmp_path):
    lib, _ = make_lib(tmp_path, n=12)
    keys = [("ai", f"t{i}.png") for i in range(12)]
    for k in keys[:8]:
        lib.request(k)
        lib.flush()
    pinned = {keys[0], keys[1]}
    for _ in range(3):
        lib.tick(False, pinned)
        for k in keys[2:8]:                              # 2..7 used more recently than 0 and 1
            lib.request(k)
    lib.request(keys[8])
    lib.flush(pinned)
    assert lib.request(keys[0]) >= 0 and lib.request(keys[1]) >= 0, "pinned layers survive although they are the oldest"
    assert lib.request(keys[8]) >= 0 and lib.request(keys[2]) == -1, "the least recently used free layer went"


def test_missing_or_escaping_files_fail_quietly(tmp_path):
    lib, ctx = make_lib(tmp_path)
    assert lib.request(("ai", "nope.png")) == -1
    lib.flush()
    assert lib.request(("ai", "nope.png")) == -1 and lib.status()["resident"] == 0 and "nope" in lib.last_error
    assert lib.path(("ai", "..")) is None and lib.path(("web", "a.png")) is None


def test_real_decode_gives_square_rgba8(tmp_path):
    from PIL import Image
    Image.new("RGB", (300, 200), (255, 0, 0)).save(tmp_path / "a.png")
    levels, asp = mt.load_image(tmp_path / "a.png", 512)
    assert [a.shape for a in levels] == [(512 >> i, 512 >> i, 4) for i in range(10)], "the whole mip chain, off the render thread"
    assert all(a.dtype == np.uint8 and tuple(a[0, 0]) == (255, 0, 0, 255) for a in levels) and abs(asp - 1.5) < 1e-6
    big = Image.new("RGBA", (4000, 3000), (0, 255, 0, 255))
    big.save(tmp_path / "big.png")
    levels, asp = mt.load_image(tmp_path / "big.png", 512)
    assert levels[0].shape == (512, 512, 4) and tuple(levels[0][5, 5]) == (0, 255, 0, 255)


# ---------------------------------------------------------------- engine side: crossfade, phases, governor


def steps(eng, st, n, dt=1 / 60, late=False, **kw):
    out = []
    for _ in range(n):
        eng.update(st.step(dt), dt, dict(bpm=120.0), late, **kw)
        out.append(eng._u["corridor"][0, 0].copy())
    return out


def test_nothing_skinned_does_no_work(tmp_path):
    eng, lib, ctx = make_engine(tmp_path)
    st = mt.SkinState()
    t0 = time.perf_counter()
    for _ in range(20000):
        eng.update(st.frame, 1 / 60)
    assert (time.perf_counter() - t0) / 20000 < 5e-6, "the all-procedural frame costs a dict check"
    assert not eng.any_active() and eng.binds == 0
    assert ctx.made == [] and eng.vram_bytes() == 0 and eng.status()["vram_mb"] == 0.0, "never used: no texture array, zero VRAM"


def test_a_skin_fades_in_then_swaps_by_crossfade_without_a_dip(tmp_path):
    eng, lib, _ = make_engine(tmp_path)
    st = frame_with(corridor__walls="t0.png")
    rows = steps(eng, st, 60)
    amts = [r[3] for r in rows]
    assert amts[0] == 0.0 and amts[-1] == 1.0 and eng.active("corridor")
    assert max(b - a for a, b in zip(amts, amts[1:])) <= (1 / 60) / mt.FADE_S + 1e-6, "no jump: at most dt / FADE_S per frame"
    st.assign("corridor", "walls", "ai", "t1.png")
    rows = steps(eng, st, 60)
    assert all(r[3] == 1.0 for r in rows), "a swap never dips through the procedural look"
    xf = [r[2] for r in rows]
    assert max(xf) > 0.9 and xf[-1] == 0.0 and max(abs(b - a) for a, b in zip(xf, xf[1:]) if b) <= (1 / 60) / mt.FADE_S + 1e-6
    assert rows[-1][0] == lib.request(("ai", "t1.png"))
    sdo(st, op="clear", env="corridor", slot="walls")
    rows = steps(eng, st, 40)
    assert rows[0][3] > 0.9 and rows[-1][3] == 0.0 and not eng.active("corridor"), "a cleared slot fades out"
    assert sorted(set(round(r[3], 3) for r in rows), reverse=True)[1] > 0.9


def test_a_slow_load_shows_the_old_skin_until_the_new_one_is_ready(tmp_path):
    eng, lib, _ = make_engine(tmp_path, size=1024)
    st = frame_with(corridor__walls="t0.png")
    steps(eng, st, 60)
    st.assign("corridor", "walls", "ai", "t1.png")
    rows = steps(eng, st, 30, late=True)                 # late frames: the upload waits
    assert all(r[3] == 1.0 and r[2] == 0.0 for r in rows) and lib.request(("ai", "t1.png")) == -1
    rows = steps(eng, st, 60)
    assert rows[-1][0] == lib.request(("ai", "t1.png")) >= 0 and all(r[3] == 1.0 for r in rows)


def test_phases_wrap_at_two_and_follow_travel(tmp_path):
    eng, lib, _ = make_engine(tmp_path)
    st = frame_with(corridor__walls="t0.png")
    sdo(st, op="set", env="corridor", slot="walls", set=dict(scroll=2.0, dir=90.0, tile=1))
    steps(eng, st, 30)
    sl = eng._slots["corridor"][0]
    for _ in range(4000):                                # more than an hour of scroll + flight
        eng.update(st.step(1.0), 0.1, dict(bpm=120.0))
        eng.advance("corridor", 37.123)
        assert 0.0 <= sl.pu < mt.PH_WRAP and 0.0 <= sl.pv < mt.PH_WRAP
    # a phase and the same phase + one wrap sample the same texel (GLSL fold == mt.fold)
    for p in np.linspace(0, 2, 41):
        for tile in (0, 1):
            assert abs(mt.fold(0.37 + p, tile) - mt.fold(0.37 + p + mt.PH_WRAP, tile)) < 1e-9
    # travel follow: the uv of a point fixed in the world does not change when the camera moves by dz
    m, pv = sl.m, sl.pv
    zr = 5.0
    eng.advance("corridor", 0.75)
    uv_before = mt.fold(m[3] * zr + pv, 1)
    uv_after = mt.fold(m[3] * (zr - 0.75) + sl.pv, 1)
    assert abs(uv_before - uv_after) < 1e-6


def test_beat_sync_scrolls_in_tiles_per_beat(tmp_path):
    eng, lib, _ = make_engine(tmp_path)
    st = frame_with(vortex__walls="t0.png")
    sdo(st, op="set", env="vortex", slot="walls", set=dict(scroll=0.25, dir=0.0, sync=1))
    eng.update(st.step(0.0), 0.0, dict(bpm=120.0))
    lib.flush()
    sl = eng._slots["vortex"][0]
    p0 = sl.pu
    for _ in range(60):                                  # 1 s at 120 bpm = 2 beats = 0.5 tile
        eng.update(st.step(1 / 60), 1 / 60, dict(bpm=120.0))
    assert abs(((sl.pu - p0) % 2.0) - 0.5) < 1e-6


def test_nothing_is_retained_per_frame_while_a_skin_runs(tmp_path):
    """Retained memory only: a skinned frame still creates short-lived floats / one small tuple (documented)."""
    eng, lib, _ = make_engine(tmp_path)
    st = frame_with(corridor__walls="t0.png", corridor__floor="t1.png")
    sdo(st, op="set", env="corridor", slot="walls", set=dict(scroll=0.5))
    steps(eng, st, 80)
    tracemalloc.start()
    a = tracemalloc.take_snapshot()
    for _ in range(600):
        eng.update(st.step(1 / 60), 1 / 60, None)
        eng.advance("corridor", 0.02)
    b = tracemalloc.take_snapshot()
    tracemalloc.stop()
    grown = sum(x.size_diff for x in b.compare_to(a, "filename") if "materials.py" in str(x.traceback))
    assert grown < 4096, f"{grown} bytes kept after 600 frames"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_governed(tmp_path):
    lib, ctx = make_lib(tmp_path, size=1024)
    clk = Clock()
    eng = mt.SkinEngine(None, None, lib=lib, clock=clk)
    return eng, lib, ctx, clk


def test_governor_decision_lowers_resolution_then_drops_to_procedural(tmp_path):
    eng, lib, ctx, clk = make_governed(tmp_path)
    st = frame_with(corridor__walls="t0.png")
    steps(eng, st, 60)
    full = eng.vram_bytes()
    eng.govern(1)                                        # the governor APPLIED a reduction
    rows = steps(eng, st, 120)
    assert lib.size == 512 and eng.vram_bytes() < full / 3.5 and rows[-1][3] == 1.0, "512 px, skin back on"
    amts = [r[3] for r in rows]
    assert max(abs(b - a) for a, b in zip(amts, amts[1:])) <= (1 / 60) / mt.FADE_S + 1e-6, "the change is a fade, never a cut"
    assert ctx.made[0].released and len(ctx.made) == 2
    eng.govern(2)                                        # its last-resort step
    rows = steps(eng, st, 60)
    assert rows[-1][3] == 0.0 and not eng.any_active() and eng.status()["procedural"]
    eng.govern(0)                                        # restored: skins come back at once, still small
    rows = steps(eng, st, 80)
    assert rows[-1][3] == 1.0 and lib.size == 512
    clk.t += mt.UPSIZE_CALM_S + 1
    eng.govern(0)                                        # the big array only after a long calm
    rows = steps(eng, st, 160)
    assert lib.size == 1024 and rows[-1][3] == 1.0


def test_no_reload_storm_when_the_governor_decision_flips(tmp_path):
    """Review #1: a decision that comes and goes costs ONE shrink; nothing is reloaded again until a long calm."""
    eng, lib, ctx, clk = make_governed(tmp_path)
    st = frame_with(corridor__walls="t0.png")
    steps(eng, st, 60)
    for i in range(40):                                  # 0 / 1 / 0 / 1 ... every half second for 20 s
        eng.govern(i % 2)
        steps(eng, st, 30)
        clk.t += 0.5
    assert len(ctx.made) == 2 and lib.size == 512, "one reallocation in total"
    assert lib.allocs == 2 and lib.evictions == 0


def test_governor_skin_quality_follows_applied_steps_not_the_raw_level():
    import governor
    g = governor.Governor(sensors=None, mode="off", log=lambda *a: None)
    g.level = 2                                          # raw strain, but the governor is OFF: it decided nothing
    assert g.skin_quality() == 0
    g.mode = "suggest"
    assert g.skin_quality() == 0, "a suggestion that was not applied changes nothing"
    res = next(s for s in g.steps if s.cat == 0)
    assert g._apply(res.id, "perf")
    assert g.skin_quality() == 1
    floor = next(s for s in g.steps if s.cat >= 3)
    assert g._apply(floor.id, "perf")
    assert g.skin_quality() == 2
    src = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    assert "_q = gov.skin_quality()" in src and "skins.govern(_q)" in src
    assert "skins.govern(_lvl)" not in src and "skins.govern(gov.level" not in src, "never the raw level"


# ---------------------------------------------------------------- generate a skin


class FakeAI:
    def __init__(self):
        self.st = dict(available=True, busy=False, queue=0, pending=0, done=[dict(id=1, name="old.png", prompt="x")])
        self.jobs = []

    def status(self):
        return dict(self.st)

    def submit(self, job):
        self.jobs.append(job)

    def pop_shown(self):
        return []


def test_generate_a_skin_waits_for_calm_and_an_idle_worker_then_assigns():
    st, ai = mt.SkinState(), FakeAI()
    sdo(st, op="gen", env="corridor", slot="walls", prompt="glowing coral lattice")
    assert st.state()["gen"]["env"] == "corridor"
    assert st.gen_tick(0.0, ai.status(), ai.submit, strained=True) is None and ai.jobs == [], "never while the governor reports strain"
    ai.st["busy"] = True
    st.gen_tick(1.0, ai.status(), ai.submit)
    ai.st.update(busy=False, queue=2)
    st.gen_tick(2.0, ai.status(), ai.submit)
    assert ai.jobs == [], "lowest priority: only when the worker has nothing else to do"
    ai.st["queue"] = 0
    st.gen_tick(3.0, ai.status(), ai.submit)
    assert len(ai.jobs) == 1
    job = ai.jobs[0]
    assert job["show"] is False and job["n"] == 1 and job["init"] is None and job["prompt"].startswith("glowing coral lattice")
    assert "seamless" in job["prompt"] and job["tag"] == "skin1"
    st.gen_tick(4.0, ai.status(), ai.submit)
    assert len(ai.jobs) == 1 and not st.any()
    ai.st["done"] = ai.st["done"] + [dict(id=2, name="gen_other.png", prompt="something else"),
                                     dict(id=3, name="gen_new.png", prompt=job["prompt"][:80])]
    assert st.gen_tick(5.0, ai.status(), ai.submit) == "gen_new.png"
    assert st.slots["corridor"]["walls"]["src"] == dict(kind="ai", name="gen_new.png") and st.gen is None


def test_generate_matches_by_job_tag_not_by_prompt():
    """Review #10: a manual AI job with the same prompt must not be taken for the skin."""
    st, ai = mt.SkinState(), FakeAI()
    ai.st["loaded"] = True
    results = {}
    sdo(st, op="gen", env="vortex", slot="walls", prompt="x" * 200)
    st.gen_tick(0.0, ai.status(), ai.submit, result=results.get)
    job = ai.jobs[0]
    ai.st["done"] = ai.st["done"] + [dict(id=7, name="gen_manual.png", prompt=job["prompt"][:80])]     # same prompt prefix, another job
    assert st.gen_tick(1.0, ai.status(), ai.submit, result=results.get) is None and not st.any()
    results[job["tag"]] = "gen_mine.png"
    assert st.gen_tick(2.0, ai.status(), ai.submit, result=results.get) == "gen_mine.png"
    assert st.slots["vortex"]["walls"]["src"]["name"] == "gen_mine.png"


def test_generate_never_overwrites_a_slot_the_performer_changed_meanwhile():
    st, ai = mt.SkinState(), FakeAI()
    results = {}
    sdo(st, op="gen", env="vortex", slot="walls", prompt="p")
    st.gen_tick(0.0, ai.status(), ai.submit, result=results.get)
    st.assign("vortex", "walls", "art", "my choice.jpg")          # the performer picked something while it rendered
    results[ai.jobs[0]["tag"]] = "gen_late.png"
    assert st.gen_tick(1.0, ai.status(), ai.submit, result=results.get) is None
    assert st.slots["vortex"]["walls"]["src"] == dict(kind="art", name="my choice.jpg") and st.gen is None
    assert "left alone" in st.gen_note


def test_generate_does_not_cold_load_the_model_without_free_vram():
    st, ai = mt.SkinState(), FakeAI()
    ai.st["loaded"] = False
    sdo(st, op="gen", env="vortex", slot="walls", prompt="p")
    st.gen_tick(0.0, ai.status(), ai.submit, vram_free_mb=900)
    assert ai.jobs == [] and "VRAM" in st.gen_note, "one click must not load 2.5 GB onto a full card"
    st.gen_tick(1.0, ai.status(), ai.submit, strained=True, vram_free_mb=3500)
    assert ai.jobs == [], "calm is re-checked at the moment of sending"
    st.gen_tick(2.0, ai.status(), ai.submit, vram_free_mb=3500)
    assert len(ai.jobs) == 1 and "loading" in st.gen_note
    st2 = mt.SkinState()
    sdo(st2, op="gen", env="vortex", slot="walls", prompt="p")
    st2.gen_tick(0.0, ai.status(), lambda j: None, vram_free_mb=900)
    st2.gen_tick(mt.GEN_WAIT_S + 1, ai.status(), lambda j: None, vram_free_mb=900)
    assert st2.gen is None, "it does not wait for ever"


def test_ai_link_matches_tagged_jobs_by_worker_id():
    import ai_link
    link = ai_link.AIClient([], start=False)
    sent = []

    def http(path, body=None, timeout=None):
        if path == "/job":
            sent.append(dict(body))
            return dict(id=41)
        return dict(done=[dict(id=40, name="other.png", prompt="p"), dict(id=41, name="mine.png", prompt="p")], queue=[])

    link._http = http
    link._send_job(dict(prompt="p", show=False, tag="skin3"))
    assert "tag" not in sent[0], "the tag is the link's own business, the worker never sees it"
    assert link.tagged("skin3") is None
    link.poll_once()
    assert link.tagged("skin3") == "mine.png" and link.tagged("skin3") is None


def test_generate_gives_up_without_a_worker_and_can_be_cancelled():
    st, ai = mt.SkinState(), FakeAI()
    ai.st["available"] = False
    sdo(st, op="gen", env="tunnel", slot="walls", prompt="p")
    st.gen_tick(0.0, ai.status(), ai.submit)
    st.gen_tick(30.0, ai.status(), ai.submit)
    assert st.gen is None and "nothing" in st.gen_note
    sdo(st, op="gen", env="tunnel", slot="walls", prompt="p")
    sdo(st, op="gen_cancel")
    assert st.gen is None


def test_controller_drives_generation_through_its_ai_link(tmp_path):
    c = make(tmp_path)
    ai = FakeAI()
    c.ai = ai
    do(c, cmd="skin", op="gen", env="torus", slot="tube", prompt="neon moss")
    c.gov_level = 1
    run(c, 0.2)
    assert ai.jobs == []
    c.gov_level = 0
    run(c, 0.2)
    assert len(ai.jobs) == 1
    ai.st["done"] = ai.st["done"] + [dict(id=9, name="gen_moss.png", prompt=ai.jobs[0]["prompt"][:80])]
    run(c, 0.2)
    assert c.skins.slots["torus"]["tube"]["src"]["name"] == "gen_moss.png"


# ---------------------------------------------------------------- shader sources (no GL)


def test_procedural_sources_are_untouched_and_hooks_still_fit():
    import fx_stack as fx
    for i, fs in enumerate(fx.LAND_FS):
        assert "texSkin" not in fs and "skA" not in fs, "the original programs carry no skin code at all"
        sk = fx.skinned_source(i)
        assert sk.count("uniform sampler2DArray texSkin,") == 1 and sk.count("skMix(") == 2
        anchor, add, _ = fx.SKIN_HOOKS[i]
        back = (sk.replace(mt.GLSL, "").replace(add, "").replace(fx.SKIN_DECL, "").replace(fx.SKIN_POST, "")
                .replace("col += skDim*palC(", "col += palC("))
        assert back == fs, "skinned = original + chunk + declarations + in-loop recorder + post-loop block + haze dimmer"
        loop = fs[fs.index("for (int i"):fs.index(fx.GLOW_ANCHOR)]
        assert "skMix(" not in add and "skTap(" not in add, "nothing heavy inside the march loop: it only records the hit"
    # 4 = (image | live array) x (plain | mirrored sampler): ONE of them runs per fetch (uniform branch); a slot still
    # costs one fetch per pixel (+ one only while a swap fades, + one per overlay layer)
    assert mt.GLSL.count("textureGrad(") == 4
    assert "sampler2DArray" in mt.GLSL and mt.UNIT not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21)


# ====================================================================== GL tests (need the GPU; run under .locks/gpu_suite)

OUT = (640, 400)
TEX = ROOT / "textures"


@pytest.fixture(scope="module")
def ctx():
    import fx_stack as fx
    c, _win = fx.hidden_context()
    yield c


@pytest.fixture()
def rig(ctx):
    import fx_stack as fx
    lp = fx.LandPass(ctx, OUT)
    lp.settle_s = 0.0
    eng = mt.SkinEngine(ctx, mt.roots_for(TEX, ROOT / "art"), size=512, sync=True)
    eng.fade_s = 0.0
    yield fx, lp, eng
    lp.release()
    eng.release()


def tex_names(n=3):
    names = sorted(p.name for p in TEX.glob("*.png"))
    if len(names) < n:
        pytest.skip("needs images in textures/")
    return names[:n]


def gl_shot(fx, lp, frames=1, t0=4.0, **over):
    return fx.render_frames(lp, fx.default_values(**over), t0=t0, frames=frames, bands=(0.5, 0.3, 0.2))


def settle(eng, st, n=4):
    for _ in range(n):
        eng.update(st.step(1 / 60), 1 / 60, dict(bpm=120.0))
        eng.lib.flush()
    eng.update(st.step(0.0), 0.0, dict(bpm=120.0))


def test_gl_default_is_pixel_identical_and_uses_the_original_programs(rig):
    fx, lp, eng = rig
    lp2 = fx.LandPass(lp.ctx, OUT)
    lp2.settle_s = 0.0
    lp.skins = eng                                      # engine attached, nothing assigned
    st = mt.SkinState()
    try:
        for mode in range(4):
            eng.update(st.step(1 / 60), 1 / 60)
            a = gl_shot(fx, lp, land_a=mode)
            b = gl_shot(fx, lp2, land_a=mode)
            assert np.array_equal(a, b), f"world {mode}: all-procedural must be the pre-skin picture, bit for bit"
            assert lp.skinned_decks == 0
        assert eng.binds == 0 and eng.lib.binds == 0, "no texture bind, no uniform write while every slot is procedural"
    finally:
        lp2.release()


def test_gl_every_world_takes_a_skin_in_every_mapping_and_mix(rig):
    fx, lp, eng = rig
    lp.skins = eng
    name = tex_names(1)[0]
    for mode, env in enumerate(mt.LAND_ENVS):
        base = gl_shot(fx, lp, land_a=mode)
        for slot in mt.ENVS[env]["slots"]:
            st = mt.SkinState()
            st.assign(env, slot, "ai", name)
            for mp in range(3):
                for mix in range(len(mt.MIX_NAMES)):
                    sdo(st, op="set", env=env, slot=slot, set=dict(map=mp, mix=mix))
                    settle(eng, st)
                    img = gl_shot(fx, lp, land_a=mode)
                    assert lp.skinned_decks == 1 and np.isfinite(img).all()
                    assert np.abs(img.astype("f4") - base.astype("f4")).mean() > 0.15, (env, slot, mp, mix)
        settle(eng, mt.SkinState())


def test_gl_black_floor_and_no_blowout(rig):
    """A skin never lights the room more than the world's own look does: black stays black (dark paint is keyed out),
    nothing clips to white, even at brightness 2."""
    fx, lp, eng = rig
    lp.skins = eng
    for mode, env in enumerate(mt.LAND_ENVS):
        settle(eng, mt.SkinState())
        base = gl_shot(fx, lp, land_a=mode)[..., :3].astype("f4") / 255.0
        base_black = (base.max(axis=2) < 0.08).mean()
        for name in tex_names(3):
            st = mt.SkinState()
            for slot in mt.ENVS[env]["slots"]:
                st.assign(env, slot, "ai", name)
            for bright in (2.0, 1.0):
                for slot in mt.ENVS[env]["slots"]:
                    sdo(st, op="set", env=env, slot=slot, set=dict(bright=bright))
                settle(eng, st)
                img = gl_shot(fx, lp, land_a=mode)[..., :3].astype("f4") / 255.0
                lum = img.max(axis=2)
                assert (lum > 0.97).mean() < 0.02, f"{env} {name}: blown out"
                assert img.mean() < max(0.45, base.mean() * 1.5), f"{env} {name}: brighter than the world itself"
            assert img.mean() < base.mean() * 1.5 + 0.02, f"{env} {name}: at brightness 1 a skin lights the room about as much as the world did"
            assert (lum < 0.08).mean() > min(0.04, base_black * 0.4), f"{env} {name}: the black of the procedural look is gone"


def test_gl_wrap_and_travel_have_no_pop(rig):
    fx, lp, eng = rig
    lp.skins = eng
    name = tex_names(1)[0]
    for mode, env in enumerate(mt.LAND_ENVS):
        st = mt.SkinState()
        slot = mt.ENVS[env]["slots"][0]
        st.assign(env, slot, "ai", name)
        sdo(st, op="set", env=env, slot=slot, set=dict(rot=33.0, scale=1.7))
        settle(eng, st)
        sl = eng._slots[env][0]
        gl_shot(fx, lp, land_a=mode)                     # the pass's first frame carries a start-up travel step: get it out of the way
        sl.pu, sl.pv = 1.9990, 0.0005
        eng.update(st.step(0.0), 0.0)
        a = gl_shot(fx, lp, land_a=mode).astype("f4")
        sl.pu, sl.pv = (1.9990 + 0.002) % 2.0, (0.0005 - 0.001) % 2.0        # both cross the wrap
        eng.update(st.step(0.0), 0.0)
        b = gl_shot(fx, lp, land_a=mode).astype("f4")
        sl.pu, sl.pv = 0.7000, 0.9000                                        # the same step far from any wrap
        eng.update(st.step(0.0), 0.0)
        c = gl_shot(fx, lp, land_a=mode).astype("f4")
        sl.pu, sl.pv = 0.7020, 0.8990
        eng.update(st.step(0.0), 0.0)
        d = gl_shot(fx, lp, land_a=mode).astype("f4")
        step, plain = np.abs(a - b).mean(), np.abs(c - d).mean()
        assert step < 1.6 * plain + 0.4, f"{env}: crossing the phase wrap ({step:.2f}) must look like any other tiny step ({plain:.2f})"


def test_gl_no_seam_at_the_angle_cut(rig):
    """The corridor's atan cut is on the left (angle = +-pi). With analytic gradients and whole tiles round the axis the
    two sides of the cut must be as alike as any neighbouring columns."""
    fx, lp, eng = rig
    lp.skins = eng
    st = mt.SkinState()
    st.assign("vortex", "walls", "ai", tex_names(1)[0])
    sdo(st, op="set", env="vortex", slot="walls", set=dict(mix=0, rot=20.0, scale=1.3))
    settle(eng, st)
    lp.debug_raw = True
    try:
        img = gl_shot(fx, lp, land_a=2, land_twist=0.0, cam_roll=0.0, land_cam=0)[..., :3].astype("f4")
    finally:
        lp.debug_raw = False
    d = np.abs(np.diff(img, axis=1)).mean(axis=(0, 2))             # column-to-column change
    assert d.max() < 12.0 * (np.median(d) + 0.5), "one column that differs far more than the rest = a seam"
    dv = np.abs(np.diff(img, axis=0)).mean(axis=(1, 2))
    assert dv.max() < 12.0 * (np.median(dv) + 0.5)


def test_gl_swap_crossfades_without_a_hitch(rig):
    fx, lp, eng = rig
    lp.skins = eng
    eng.fade_s = mt.FADE_S
    eng.lib.sync = False                                 # the real path: decode on the worker thread, strips per frame
    a, b = tex_names(2)
    st = mt.SkinState()
    st.assign("corridor", "walls", "ai", a)
    v = fx.default_values(land_a=0)
    frames, worst = [], 0.0

    def one(i):
        nonlocal worst
        t0 = time.perf_counter()
        eng.update(st.step(1 / 60), 1 / 60, dict(bpm=120.0), late=False)
        worst = max(worst, (time.perf_counter() - t0) * 1000.0)
        tex = lp.render(v, fx.make_clock(4.0), 4.0, 0.0, 0.0, fx.GALLERY_PALETTE, (0.5, 0.3, 0.2))
        return fx.read_texture(tex).astype("f4")

    for i in range(200):
        one(i)
        if eng._u["corridor"][0, 0, 3] >= 1.0:
            break
        time.sleep(0.005)
    assert eng._u["corridor"][0, 0, 3] >= 1.0
    st.assign("corridor", "walls", "ai", b)
    prev = one(0)
    jumps = []
    for i in range(240):
        cur = one(i)
        jumps.append(np.abs(cur - prev).mean())
        prev = cur
        if eng._slots["corridor"][0].cur_key == ("ai", b):
            break
        time.sleep(0.004)
    assert eng._slots["corridor"][0].cur_key == ("ai", b)
    assert max(jumps) < 14.0, f"largest frame-to-frame change {max(jumps):.1f}/255: the swap must be a fade"
    assert worst < 8.0, f"the slowest skin update took {worst:.1f} ms (decode is off-thread, uploads are strips)"


def test_gl_state_is_restored(rig):
    import moderngl
    fx, lp, eng = rig
    lp.skins = eng
    ctx = lp.ctx
    st = mt.SkinState()
    st.assign("torus", "tube", "ai", tex_names(1)[0])
    settle(eng, st)
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((32, 32), 4)])
    fbo.use()
    ctx.viewport = (1, 2, 30, 29)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
    probe = ctx.texture((4, 4), 4, data=bytes([200] * 64))
    probe.use(0)
    lp.render(fx.default_values(land_a=1), fx.make_clock(1.0), 1.0, 0.0, 1 / 60, fx.GALLERY_PALETTE, (0, 0, 0))
    assert lp.skinned_decks == 1
    assert ctx.fbo is fbo and ctx.viewport == (1, 2, 30, 29)
    from math_layers import _gl_lib, _GL_BLEND, _GL_BLEND_ENUMS, gl_get_int
    gl = _gl_lib()
    assert gl.glIsEnabled(_GL_BLEND)
    assert [gl_get_int(gl, e) for e in _GL_BLEND_ENUMS][:2] == [moderngl.SRC_ALPHA, moderngl.ONE]
    ctx.disable(moderngl.BLEND)
    ctx.screen.use()
    fbo.release()
    probe.release()


def gpu_ms2(ctx, draw_a, draw_b, n=40, warm=6):
    """Median GPU times of two draws measured alternately (ms)."""
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


def gpu_ms(ctx, draw, n=40, warm=6):
    """Median GPU time of draw() in ms (timer queries)."""
    out = []
    for i in range(n + warm):
        q = ctx.query(time=True)
        with q:
            draw()
        ctx.finish()
        if i >= warm:
            out.append(q.elapsed / 1e6)
    return float(np.median(out))


def test_gl_cost_budget(ctx):
    """<= 0.3 ms extra per skinned world at 1280x800 (ONE fetch per slot per pixel), zero when procedural."""
    import fx_stack as fx
    size = (1280, 800)
    lp, lp0 = fx.LandPass(ctx, size), fx.LandPass(ctx, size)
    eng = mt.SkinEngine(ctx, mt.roots_for(TEX, ROOT / "art"), size=512, sync=True)
    eng.fade_s = 0.0
    lp.skins = eng
    name = tex_names(1)[0]
    report = []
    try:
        for res in (0.5, 1.0):
            for mode, env in enumerate(mt.LAND_ENVS):
                v = fx.default_values(land_a=mode, land_res=res)
                clock = fx.make_clock(4.0)

                def draw(p):
                    p.render(v, clock, 4.0, 0.0, 0.0, fx.GALLERY_PALETTE, (0.5, 0.3, 0.2))

                settle(eng, mt.SkinState())
                # interleaved, so thermal drift hits both alike: the pre-skin build (no engine attached) and the
                # same pass with the engine attached and every slot procedural
                pre, off = gpu_ms2(ctx, lambda: draw(lp0), lambda: draw(lp))
                st = mt.SkinState()
                for slot in mt.ENVS[env]["slots"]:
                    st.assign(env, slot, "ai", name)
                settle(eng, st)
                pre2, on = gpu_ms2(ctx, lambda: draw(lp0), lambda: draw(lp))
                on += pre - pre2                                  # same baseline as the first pair
                report.append((res, env, pre, off, on))
        print()
        for res, env, pre, off, on in report:
            print(f"[skin cost] land_res {res:.1f} {env:9s} pre-skin {pre:.3f} ms  procedural {off:.3f} ms  skinned {on:.3f} ms  (+{on - off:.3f})")
        for res, env, pre, off, on in report:
            assert abs(off - pre) < max(0.06, 0.05 * pre), f"{env}: procedural must cost what the pre-skin build costs"
            assert on - off <= 0.30, f"{env} at land_res {res}: a skin costs {on - off:.3f} ms (> 0.3)"
    finally:
        lp.release()
        lp0.release()
        eng.release()


# ====================================================================== panel (headless browser, no GPU needed)


def test_panel_skins_card_assigns_by_click_and_by_drag(tmp_path):
    pw_api = pytest.importorskip("playwright.sync_api")
    import socket
    import threading
    from PIL import Image
    from remote_control import RemoteControl
    (tmp_path / "tex").mkdir()
    (tmp_path / "art").mkdir()
    for n, col in (("gen_a.png", (200, 30, 90)), ("tex_b.png", (20, 160, 200))):
        Image.new("RGB", (64, 64), col).save(tmp_path / "tex" / n)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art", tex_dir=tmp_path / "tex")
    rc.start()
    ctl = control.Controller(tmp_path, midi=False, ai=False, art_dir=tmp_path / "art", restore=False)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            ctl.apply(rc.drain())
            ctl.update(1 / 60, [0.5, 0.3, 0.1])
            rc.publish_state(dict(ctl.state(), corners=[[0, 0], [1, 0], [1, 1], [0, 1]], res=[1280, 800], fps=60,
                                  skin_lib=dict(size=512, layers=8, resident=1, loading=0, vram_mb=10.7, error="")))
            time.sleep(1 / 60)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    errors = []
    try:
        with pw_api.sync_playwright() as pw:
            br = None
            for channel in ("msedge", "chrome", None):
                try:
                    br = pw.chromium.launch(channel=channel, headless=True) if channel else pw.chromium.launch(headless=True)
                    break
                except Exception:
                    continue
            if br is None:
                pytest.skip("no Chromium-family browser available")
            page = br.new_page(viewport=dict(width=1600, height=1600))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/?k=tok")
            page.wait_for_function("window.__panel && window.__skins", timeout=10000)
            page.click('#tabs button[data-tab="land"]')
            # ships OFF: one switch and one honest line; no slots, no strip, no generate box, no /sources request
            page.wait_for_selector("#skins-land .sksw", timeout=8000)
            assert not ctl.skins.on and page.inner_text("#skins-land .sksw") == "SKINS: OFF"
            assert page.locator("#skins-land .sksl:visible").count() == 0 and page.locator("#skins-land .skgen:visible").count() == 0
            assert page.locator("#skins-land .skstrip .skt").count() == 0
            page.click("#skins-land .sksw")
            page.wait_for_timeout(300)
            assert ctl.skins.on
            page.wait_for_selector("#skins-land .skstrip .skt", timeout=8000)
            assert page.locator("#skins-land .sksl").count() == 6, "corridor 2 + torus 2 + vortex 1 + hexdome 1"
            # touch path: tap a slot, then an image
            page.click('#skins-land .sksl[data-env="corridor"][data-slot="walls"] .th')
            page.click('#skins-land .skstrip .skt[title="gen_a.png"]')
            page.wait_for_timeout(400)
            assert ctl.skins.slots["corridor"]["walls"]["src"] == dict(kind="ai", name="gen_a.png")
            # drag path
            a = page.locator('#skins-land .skstrip .skt[title="tex_b.png"]').bounding_box()
            b = page.locator('#skins-land .sksl[data-env="torus"][data-slot="chambers"]').bounding_box()
            page.mouse.move(a["x"] + 30, a["y"] + 30)
            page.mouse.down()
            page.mouse.move(a["x"] + 50, a["y"] + 20, steps=4)
            page.mouse.move(b["x"] + 60, b["y"] + 60, steps=8)
            page.wait_for_timeout(100)
            page.mouse.up()
            page.wait_for_timeout(400)
            assert ctl.skins.slots["torus"]["chambers"]["src"] == dict(kind="ai", name="tex_b.png")
            # stepped + numeric controls are learnable targets and drive the engine state
            assert page.locator('#skins-land [data-mt="skin:torus:chambers:scale"]').count() == 1
            page.click('#skins-land .skctl .skb button:has-text("Multiply")')
            page.wait_for_timeout(300)
            assert ctl.skins.slots["torus"]["chambers"]["mix"] == 1
            # skin STACK: a live layer on top of the selected slot, its blend, colour by channel
            sels = page.locator("#skins-land .skctl select")
            sels.nth(0).select_option("layer:lines4")
            page.wait_for_timeout(300)
            lay = ctl.skins.slots["torus"]["chambers"]["layers"][0]
            assert lay is not None and lay["src"] == dict(kind="layer", name="lines4")
            sels.nth(1).select_option("1")
            page.click('#skins-land .skctl button:has-text("COLOUR BY CHANNEL") >> nth=0')
            page.wait_for_timeout(400)
            lay = ctl.skins.slots["torus"]["chambers"]["layers"][0]
            assert lay["blend"] == 1 and lay["remap"] == 1
            page.click('#skins-land button:has-text("EXAMPLE WORLD")')
            page.wait_for_timeout(400)
            assert ctl.skins.slots["vortex"]["walls"]["layers"][0]["src"] == dict(kind="layer", name="lines4")
            # generate a skin: lands in the model as a pending job for the selected slot
            page.fill("#skgen-land", "glowing coral lattice")
            page.click('#skins-land .skgen button:has-text("GENERATE A SKIN")')
            page.wait_for_timeout(300)
            assert ctl.skins.gen is not None and ctl.skins.gen["slot"] == "chambers"
            # back to procedural
            page.click('#skins-land .sksl[data-env="corridor"][data-slot="walls"] button')
            page.wait_for_timeout(300)
            assert ctl.skins.slots["corridor"]["walls"]["src"] is None
            page.click('#tabs button[data-tab="motion"]')
            assert page.locator("#skins-motion .sksl").count() == 1
            assert page.evaluate("document.documentElement.scrollWidth - window.innerWidth") <= 2
            out = ROOT / "test_frames" / "skins"
            out.mkdir(parents=True, exist_ok=True)
            page.click('#tabs button[data-tab="land"]')
            page.wait_for_timeout(300)
            page.locator("#skins-land").screenshot(path=str(out / "panel_skins.png"))
            br.close()
        assert errors == []
    finally:
        stop.set()
        th.join(timeout=2)
        ctl.close()
        rc.stop()


def test_gl_scene_shader_without_a_skin_is_the_pre_skin_shader(ctx):
    """The engine draws SCENE_FS_PLAIN (no skin code at all) unless a skin is on the travel tunnel; SCENE_FS (with the
    block) at amount 0 gives the same picture. Prints what the skipped block would have cost had it been left in."""
    import moderngl
    import live_visuals as lv
    plain, fs = lv.SCENE_FS_PLAIN, lv.SCENE_FS
    assert "texSkin" not in plain and "skA" not in plain and "skins>" not in plain and "SKIN_GLSL" not in plain
    assert fs.count("// [skins>") == 1 and fs.count("// <skins]") == 1 and "sampler2DArray texSkin" in fs
    cutout = fs[:fs.index("  // [skins>")] + fs[fs.index("  // <skins]") + len("  // <skins]"):]
    assert cutout.replace(mt.glsl(1, "palette"), "").split() == plain.split(), "plain = the scene minus the skin block, nothing else"
    src_engine = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    assert "scene_prog = ctx.program(vertex_shader=VS, fragment_shader=SCENE_FS_PLAIN)" in src_engine
    size = (1280, 800)
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
    fbo = ctx.framebuffer(color_attachments=[ctx.texture(size, 4, dtype="f2"), ctx.texture(size, 4, dtype="f2")])
    sets = dict(p_symmetry=1.0, p_folds=8.0, p_l_kene=1.0, p_l_beads=0.6, p_l_lines=0.5, p_master=1.0, aspect=1.6, pxh=1 / 800,
                p_density=0.5, p_saturation=1.0, limiter=1.0, coreOn=1.0, p_travel_mode=1.0, t=3.0, tK=0.3, bass=0.4)
    vaos, progs = [], []
    for src in (plain, fs):
        prog = ctx.program(vertex_shader=lv.VS, fragment_shader=src)
        for k, v in sets.items():
            if k in prog:
                prog[k].value = v
        if "pal" in prog:
            prog["pal"].write(np.array([(0, .85, .8), (1, .05, .55), (1, .7, .05), (.28, .1, 1), (.1, .9, .3)], dtype="f4").tobytes())
        progs.append(prog)
        vaos.append(ctx.vertex_array(prog, [(quad, "2f", "in_pos")]))
    fbo.use()
    ctx.viewport = (0, 0) + size
    ctx.disable(moderngl.BLEND)
    times, imgs = [[], []], []
    for it in range(30):                                 # interleaved, so drift / throttling hits both alike
        for i, vao in enumerate(vaos):
            q = ctx.query(time=True)
            with q:
                vao.render(moderngl.TRIANGLE_STRIP)
            ctx.finish()
            if it >= 5:
                times[i].append(q.elapsed / 1e6)
    for vao in vaos:
        vao.render(moderngl.TRIANGLE_STRIP)
        imgs.append(np.frombuffer(fbo.read(components=3, attachment=0, dtype="f4"), dtype="f4").copy())
    ctx.screen.use()
    for o in vaos + progs:
        o.release()
    print(f"\n[skin cost] scene shader 1280x800: plain (what the show runs) {np.median(times[0]):.3f} ms; "
          f"with the tunnel block compiled in but skipped {np.median(times[1]):.3f} ms")
    assert imgs[0].max() > 0.05, "the test picture must not be black"
    assert np.array_equal(imgs[0], imgs[1]), "amount 0: the skinned program draws the plain picture bit for bit"


def test_gl_engine_tunnel_skin_end_to_end(tmp_path):
    """The real engine, headless: no skin = no bind and no upload; a skin on the tunnel changes the picture, stays finite."""
    import subprocess
    name = tex_names(1)[0]
    (tmp_path / "skins.json").write_text(json.dumps(dict(slots={"tunnel:walls": dict(src=dict(kind="ai", name=name), mix=4)})))
    outs = []
    for tag, extra in (("plain", []), ("skin", ["--skins", str(tmp_path / "skins.json")])):
        r = subprocess.run([sys.executable, str(ROOT / "live_visuals.py"), "--test", "24", "--out", str(tmp_path / tag),
                            "--set", "travel=0.6", "--set", "travel_mode=1", "--test-audio", "silent"] + extra,
                           capture_output=True, text=True, timeout=120, cwd=str(ROOT))
        assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
        outs.append(r.stdout)
    assert "[skins] binds=0 uploads=0" in outs[0], "no skin: the engine never binds or uploads a skin texture"
    assert "ready=0 arrays=0" in outs[0] and "vram_mb=0.0" in outs[0], "inactive: no skinned programs, no texture array"
    assert "nonfinite=0" in outs[1] and "[skins] binds=0" not in outs[1] and "uploads=0" not in outs[1]
    assert "ready=1 arrays=1" in outs[1]
    from PIL import Image
    a = np.asarray(Image.open(sorted((tmp_path / "plain").glob("frame_*.png"))[-1]).convert("RGB")).astype("f4")
    b = np.asarray(Image.open(sorted((tmp_path / "skin").glob("frame_*.png"))[-1]).convert("RGB")).astype("f4")
    assert np.abs(a - b).mean() > 1.0



def test_gl_uploads_never_rebuild_mips_and_every_step_is_small(ctx):
    """Review #2 on the real GPU: mips come from the loader thread, each level goes up with glTexSubImage3D, no
    glGenerateMipmap after a load, and no single upload step is long (1024 px, glFinish included)."""
    name = tex_names(1)[0]
    lib = mt.SkinLibrary(ctx, mt.roots_for(TEX, ROOT / "art"), size=1024, sync=True)
    try:
        assert lib.tex is None and lib.vram_bytes() == 0, "allocated on first use only"
        lib.request(("ai", name))
        worst, n = 0.0, 0
        while lib.request(("ai", name)) < 0:
            t0 = time.perf_counter()
            lib.tick(False)
            ctx.finish()
            if n:                                        # the first step also allocates the array (once per show)
                worst = max(worst, (time.perf_counter() - t0) * 1000.0)
            n += 1
            assert n < 30
        if lib.raw is None:
            pytest.skip("raw GL uploads unavailable here: the moderngl fallback is in use")
        assert lib.mips == 0, "glGenerateMipmap must never run for a loaded skin"
        assert n == 6 and worst < 3.0, f"{n} steps, worst {worst:.2f} ms"
        print(f"\n[skin cost] 1024 px skin upload: {n} frames, worst step {worst:.2f} ms (no mip rebuild)")
    finally:
        lib.release()


def test_gl_per_level_uploads_match_the_gl_generated_mips(ctx):
    """The ctypes path puts every level into the right level / layer: a far, minified view (high mips) looks like the
    same view with GL-generated mips."""
    import fx_stack as fx
    name = tex_names(2)[1]
    shots = []
    for raw in (None, False):
        lib = mt.SkinLibrary(ctx, mt.roots_for(TEX, ROOT / "art"), size=512, sync=True, raw=raw)
        eng = mt.SkinEngine(None, None, lib=lib)
        eng.fade_s = 0.0
        lp = fx.LandPass(ctx, OUT)
        lp.settle_s = 0.0
        lp.skins = eng
        try:
            st = mt.SkinState()
            st.assign("vortex", "walls", "ai", name)
            sdo(st, op="set", env="vortex", slot="walls", set=dict(scale=4.0, mix=0))
            settle(eng, st)
            if raw is None and lib.raw is None:
                pytest.skip("raw GL uploads unavailable here")
            shots.append(gl_shot(fx, lp, land_a=2, land_res=0.25).astype("f4"))
        finally:
            lp.release()
            eng.release()
    assert shots[0].max() > 30
    assert np.abs(shots[0] - shots[1]).mean() < 3.0, "box-filtered CPU mips and GL mips must agree closely"


def test_gl_cost_budget_triplanar_two_images_and_1024(ctx):
    """Review #15: the 0.3 ms budget also holds for the expensive cases: triplanar (3 fetches), two DIFFERENT images
    on a two-slot world, and the 1024 px array."""
    import fx_stack as fx
    size = (1280, 800)
    a, b = tex_names(2)
    lp, lp0 = fx.LandPass(ctx, size), fx.LandPass(ctx, size)
    eng = mt.SkinEngine(ctx, mt.roots_for(TEX, ROOT / "art"), size=1024, sync=True)
    eng.fade_s = 0.0
    lp.skins = eng
    rows = []
    try:
        for mode, env in ((0, "corridor"), (1, "torus")):
            v = fx.default_values(land_a=mode, land_res=1.0)
            clock = fx.make_clock(4.0)

            def draw(pss):
                pss.render(v, clock, 4.0, 0.0, 0.0, fx.GALLERY_PALETTE, (0.5, 0.3, 0.2))

            for mp in (0, 1):
                st = mt.SkinState()
                for slot, nm in zip(mt.ENVS[env]["slots"], (a, b)):
                    st.assign(env, slot, "ai", nm)
                    sdo(st, op="set", env=env, slot=slot, set=dict(map=mp))
                settle(eng, st)
                pre, on = gpu_ms2(ctx, lambda: draw(lp0), lambda: draw(lp))
                rows.append((env, mp, pre, on))
        print()
        for env, mp, pre, on in rows:
            print(f"[skin cost] 1024 px, two images, {env:9s} {'triplanar' if mp else 'surface  '} pre-skin {pre:.3f} ms  skinned {on:.3f} ms  (+{on - pre:.3f})")
        for env, mp, pre, on in rows:
            assert on - pre <= 0.30, f"{env} map {mp}: +{on - pre:.3f} ms"
    finally:
        lp.release()
        lp0.release()
        eng.release()
