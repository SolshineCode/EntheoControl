"""PROMPT mode spike (prompt_mode.py): op parsing, expansion, safety validation, scoring. CPU only, no model needed
except the last test (skipped when models/llm has no GGUF or the CPU llama-server copy is missing)."""
import json

import pytest

import prompt_mode as pm
from params import PARAMS, TRIGGERS


@pytest.fixture(scope="module")
def ctx():
    return pm.make_ctx()


def run(ops, ctx, themes=pm.THEMES):
    return pm.validate([pm.parse_op(o) for o in ops], ctx, themes)


def test_parse_ops():
    assert pm.parse_op("speed--") == {"kind": "step", "p": "speed", "sign": -1, "n": 2}
    assert pm.parse_op("zoom+++")["n"] == 3
    assert pm.parse_op("palette=Emerald") == {"kind": "pick", "p": "palette", "opt": "Emerald"}
    assert pm.parse_op("being=Jaguar spirit")["opt"] == "Jaguar spirit"
    assert pm.parse_op("l_text=off") == {"kind": "level", "p": "l_text", "level": "off"}
    assert pm.parse_op("zoom~bass") == {"kind": "mod", "p": "zoom", "band": "bass"}
    assert pm.parse_op("theme 2") == {"kind": "theme", "n": 2}
    assert pm.parse_op("solo l_stack l_lines")["keep"] == ["l_stack", "l_lines"]
    assert pm.parse_op("say:HELLO") == {"kind": "say", "text": "HELLO"}
    assert pm.parse_op("refuse") == {"kind": "refuse"}
    assert pm.parse_op("rm -rf /") is None and pm.parse_op(3) is None


def test_grammar_only_admits_known_names():
    names = pm.catalog("core")
    g = pm.grammar(names, 5)
    assert all(n in PARAMS for n in names) and not set(names) & set(TRIGGERS)
    assert '"strobe"' not in g and '"flash"' not in g and '"blackout"' not in g
    assert '"Emerald"' in g and '"Jaguar spirit"' in g and "[1-5]" in g
    assert "master" not in g.split("modp ::=")[1].split("\n")[0]        # master cannot follow the sound


def test_retrieval_picks_relevant_params():
    n = pm.catalog("retrieval", "more kene, less beads")
    assert "l_kene_shipibo" in n and "l_beads" in n and len(n) <= 30
    assert "l_stack" in pm.catalog("retrieval", "drop everything to just the stripes")


def test_relative_moves_are_validated_and_clamped(ctx):
    r = run(["speed---", "speed---", "zoom++"], ctx)
    sets = {c["name"]: c["v"] for c in r["commands"] if c["cmd"] == "set"}
    assert 0.0 <= sets["speed"] < ctx["base"]["speed"] and sets["zoom"] > ctx["base"]["zoom"]
    assert not pm.unsafe(r["commands"], ctx)


def test_pick_being_brings_layer_in(ctx):
    r = run(["being=Jaguar spirit"], ctx)
    sets = {c["name"]: c["v"] for c in r["commands"] if c["cmd"] == "set"}
    assert sets["being"] == 2.0 and sets["l_beings"] >= 0.5


def test_unknown_option_rejected(ctx):
    r = run(["being=Unicorn"], ctx)
    assert not r["commands"] and r["rejected"]


def test_blackout_and_master_floor(ctx):
    r = run(["master=off"], ctx)
    assert all(c["v"] >= 0.15 for c in r["commands"] if c["cmd"] == "set" and c["name"] == "master")
    r = run(["solo l_video"], ctx)          # l_video at 0 -> brought in; everything else off: still visible
    assert not pm.unsafe(r["commands"], ctx)


def test_trigger_ops_never_pass(ctx):
    r = run(["strobe+++", "flash=max", "blackout=max"], ctx)
    assert not r["commands"] and len(r["rejected"]) == 3


def test_chase_clamped(ctx):
    r = run(["chase_mode=Counter", "chase_rate=max", "chase=max"], ctx)    # counter chase: 4 flashes a sweep
    assert not pm.unsafe(r["commands"], ctx)
    assert any("chase_rate" in n for n in r["notes"])


def test_mod_on_layer_or_master_refused(ctx):
    r = run(["master~bass"], ctx)
    assert not r["commands"]
    r = run(["zoom~bass"], ctx)
    assert r["commands"][0]["cmd"] == "mod" and r["commands"][0]["src"] == "bass"


def test_theme_then_modifier_is_relative_to_theme(ctx):
    r = run(["theme 2", "speed-"], ctx)
    assert r["commands"][0] == {"cmd": "theme", "op": "restore", "id": "t2", "bars": 2.0}
    sp = next(c["v"] for c in r["commands"] if c.get("name") == "speed")
    assert sp < pm.THEMES[1]["values"]["speed"] and sp > ctx["base"]["speed"]      # slower than look 2, not than now
    assert run(["theme 9"], ctx)["rejected"]


def test_text_op(ctx):
    r = run(["say:WELCOME HOME"], ctx)
    assert any(c["cmd"] == "text" and c["text"] == "WELCOME HOME" for c in r["commands"])


def test_density_budget_never_grows_a_busy_look(ctx):
    r = run(["l_swarm=high", "l_fractal=high", "l_land=high"], ctx)
    assert not pm._over(r["commands"], ctx) and any("density budget" in n for n in r["notes"])
    sets = {c["name"]: c["v"] for c in r["commands"] if c["cmd"] == "set"}
    assert sets.get("l_fractal", 0) > 0.5 and sets.get("l_land", 0) > 0.5       # what was asked for stays


def test_density_budget_caps_filters_added_at_once(ctx):
    r = run(["flt_neon+++", "flt_bloom+++", "flt_oil+++", "flt_dream+++"], ctx)
    assert not pm._over(r["commands"], ctx)


def test_score():
    it = [("dir", "speed", -1), ("dir", "travel", 1)]
    assert pm.score(["speed-", "travel+|zoom+"], it)["exact"] == 1.0
    assert pm.score(["speed-"], it)["exact"] == 0.0 and pm.score(["speed-"], it)["dir_recall"] == 1.0
    assert pm.score(["speed-", "?travel+"], it)["exact"] == 1.0
    assert pm.score(["refuse|chase+"], [("refuse", None, None)])["exact"] == 1.0
    assert pm.score(["*~bass"], [("mod", "zoom", "bass")])["exact"] == 1.0
    assert pm.score(["theme:2", "speed-"], [("theme", None, 2), ("dir", "speed", 1)])["dir_recall"] == 0.5


def test_eval_set_gold_is_well_formed():
    assert len(pm.EVAL) >= 60
    for phrase, gold in pm.EVAL:
        for g in gold:
            for a in g.lstrip("?").split("|"):
                m = a.split("=")[0].split("~")[0].rstrip("+-*")
                assert a in ("refuse", "say") or a.startswith(("solo:", "theme:")) or a.startswith("*") or m in PARAMS, a
                if "=" in a:
                    assert a.split("=", 1)[1] in pm.options(m), a


def test_fingerprint_finds_similar_look():
    from PIL import Image
    red = Image.new("RGB", (80, 60), (220, 30, 30))
    green = Image.new("RGB", (80, 60), (20, 200, 60))
    half = Image.new("RGB", (80, 60), (0, 0, 0))
    half.paste((210, 40, 40), (0, 0, 40, 60))
    looks = [{"fp": pm.fingerprint(g)} for g in (green, red)]
    i, cos = pm.nearest_look(pm.fingerprint(half), looks)
    assert i == 1 and -1.0 <= cos <= 1.0


LLM = sorted(pm.LLM_DIR.glob("LFM2-350M*.gguf")) if pm.LLM_DIR.exists() else []


@pytest.mark.skipif(not LLM or not pm.LLM_BIN.exists(), reason="no local model / CPU llama-server")
def test_live_model_reply_always_parses(ctx):
    llm = pm.LocalLLM(LLM[0], threads=4, port=8793)
    try:
        for ph in ("faster and emerald", "strobe like crazy", "back to the second one"):
            r = pm.phrase_to_commands(llm, ph, ctx, pm.THEMES)
            assert isinstance(json.loads(r["raw"]), list)
            assert not pm.unsafe(r["commands"], ctx)
    finally:
        llm.close()
