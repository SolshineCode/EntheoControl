"""WALL LOOK LOOP: Gemini Neon (via agy) iteratively improves the projection-mapped look by LOOKING AT THE REAL WALL
through the AUTO MAP camera (user 2026-11-26: "have agy iteratively improve it visually using this look with the
projector and camera").

Each round: camera picture of the wall (AUTO MAP live view) + the mapped surfaces + current settings -> agy (Gemini Neon)
-> score + a few setting changes -> applied through the normal panel command API (server-side validation, flash
guard stays on) -> settle -> next picture. If a round scores lower than the best so far, the best settings come back.

Safety: only whitelisted params (no flash/strobe/invert/freeze/camera/video layers) and surface look fields
(content / src / rect / opacity / brightness / glow / color); values are clamped to the registry ranges; at most
MAX_CHANGES per round. Pictures of the room stay local (.wall_loop/, gitignored) and only go to agy's sandbox.

Run (show running, camera on, surfaces mapped):
    PYTHONUTF8=1 .venv/Scripts/python.exe -u tools/wall_look_loop.py --rounds 5 [--goal "..."]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import params as P  # noqa: E402

BASE = "http://127.0.0.1:8765"
NOTES = ROOT / ".wall_loop" / "NOTES.md"   # persistent local notebook across runs (user 2026-11-26): lessons + updates
NOTES_FEED = 1500                          # last N chars of the notebook are given back to agy each round
MAX_CHANGES = 8
SETTLE_S = 8.0
BLOCK = {"flash", "strobe", "invert", "freeze", "master", "l_camera", "l_video", "rec", "img_auto"}
BLOCK_PREFIX = ("crowd_", "cam_", "rec_", "midi")
SURF_FIELDS = ("content", "src", "rect", "opacity", "brightness", "glow", "color", "enabled", "layers")
NO_SKIN_SRC = ("video", "camera")            # never skin surfaces with people / video sources from the loop
CORE = ("speed", "folds", "hue", "hue_speed", "saturation", "contrast", "zoom", "spin", "fb", "glow", "bright")
DEFAULT_GOAL = ("Make this look stunning as a projection-mapped psychedelic piece IN THIS ROOM: each mapped surface should "
                "read as its own distinct, moving, beautiful thing on the real object (walls, hanging cloth, plant, pot), "
                "with strong contrast on the physical surfaces, edges that sit on the objects, no clutter or "
                "overcrowding, pleasing motion speed, and a coherent overall composition.")


def token():
    return (ROOT / "show_preview4" / ".remote_token").read_text().strip() if (ROOT / "show_preview4" / ".remote_token").exists() \
        else (ROOT / ".remote_token").read_text().strip()


K = token()


def post(c):
    r = urllib.request.Request(f"{BASE}/cmd?k={K}", data=json.dumps(c).encode(), method="POST",
                               headers={"Content-Type": "application/json", "X-Token": K, "Origin": BASE})
    return urllib.request.urlopen(r, timeout=10).read().decode()


def state():
    r = urllib.request.Request(f"{BASE}/state?k={K}", headers={"X-Token": K})
    return json.loads(urllib.request.urlopen(r, timeout=10).read())


def picture(path):
    for _ in range(6):
        r = urllib.request.Request(f"{BASE}/automap-image?k={K}&i={time.time()}", headers={"X-Token": K})
        try:
            b = urllib.request.urlopen(r, timeout=10).read()
            if b[:2] == b"\xff\xd8":
                path.write_bytes(b)
                return True
        except Exception:
            pass
        time.sleep(1.5)
    return False


def allowed(name):
    return name in P.PARAMS and name not in BLOCK and not name.startswith(BLOCK_PREFIX)


def snapshot(s):
    vals = {k: v for k, v in (s.get("values") or {}).items() if allowed(k)}
    # state keeps the picture source as source.mode / source.rect; the "set" command calls them src / rect
    surfs = {x["id"]: {f: (x.get("source") or {}).get({"src": "mode"}.get(f, f)) if f in ("src", "rect") else x.get(f)
                       for f in SURF_FIELDS
                       if ((x.get("source") or {}).get({"src": "mode"}.get(f, f)) if f in ("src", "rect") else x.get(f)) is not None}
             for x in (s.get("surfaces") or {}).get("surfaces", [])}
    return {"values": vals, "surfaces": surfs}


def apply(cmds, log):
    done = []
    for c in cmds[:MAX_CHANGES]:
        try:
            if c.get("cmd") == "set" and allowed(str(c.get("name"))):
                lo, hi = P.PARAMS[c["name"]][0], P.PARAMS[c["name"]][1]
                v = min(hi, max(lo, float(c["v"])))
                post({"cmd": "set", "name": c["name"], "v": v})
                done.append(f"{c['name']}={v:.3g}")
            elif c.get("cmd") == "surf" and isinstance(c.get("id"), int):
                o = {"cmd": "surf", "op": "set", "id": c["id"]}
                o.update({f: c[f] for f in SURF_FIELDS if f in c})
                if isinstance(o.get("layers"), list):          # per-surface skin stack (surfaces.py content "layers")
                    o["layers"] = [L for L in o["layers"][:3] if isinstance(L, dict) and L.get("src") not in NO_SKIN_SRC]
                    o["content"] = "layers"
                if any(f in c for f in SURF_FIELDS):
                    r = post(o)
                    done.append(f"surf{c['id']}:{','.join(f for f in SURF_FIELDS if f in c)}" + ("" if '"ok": true' in r else "(refused)"))
            else:
                done.append(f"skipped {json.dumps(c)[:60]}")
        except Exception as ex:
            done.append(f"error {type(ex).__name__}")
    log.write(f"  applied: {done}\n")
    return done


def restore(snap):
    for k, v in snap["values"].items():
        try:
            post({"cmd": "set", "name": k, "v": v})
        except Exception:
            pass
    for i, f in snap["surfaces"].items():
        try:
            post(dict({"cmd": "surf", "op": "set", "id": i}, **f))
        except Exception:
            pass


def param_table(vals):
    rows = []
    for k, spec in P.PARAMS.items():
        if allowed(k) and (k in CORE or k.startswith("l_") or abs(vals.get(k, spec[2]) - spec[2]) > 1e-6):
            rows.append(f"{k} | {spec[3]} | {spec[4]} | {spec[0]}..{spec[1]} | now {vals.get(k, spec[2]):.3g}")
    return "\n".join(rows)


TASK = """# WALL LOOK LOOP, round {r} of {n}

You are the visual director of a live projection-mapping show. `wall.jpg` in this folder is a REAL camera photo of the
wall (greyscale webcam: judge brightness, contrast, structure, motion blur, placement; colours are invisible).
{prev}
GOAL: {goal}

Mapped surfaces (id | name | current look; content main=live show, solid=colour, glow=edge glow, black=off;
src full=whole show / rect=a window [x,y,w,h] 0..1 of the show / span=one image across all):
{surfs}

SKINS (per-surface layers): a surface can carry its OWN stack of up to 3 moving layers instead of the show:
{{"cmd":"surf","id":<id>,"layers":[{{"src":"<source>","on":true,"opacity":0..1,"blend":"normal|add|screen|multiply","scale":<~0.25..4>,"scroll":<speed>,"dir":<degrees>,"hue":<shift>}}, ...]}}
sources: kene (Kené study), beadstudy, blocks (Folding Blocks), beads, lines (sacred lines), beings, fractal, land (3D deck),
stack (math look), art, ai, text, lines4, kene4, main (the show itself). EVERY ROUND, test skinning at least one mapped
surface with a design that suits its real object (e.g. flowing kené on the cloth, beads/lines along the plant, fractal on a
wall), and judge on the photo whether it reads better than the show crop. If the engine refuses "layers" (older build),
note it and fall back to crops.

Global settings you may change (name | group | label | range | current):
{table}

Your notebook from earlier rounds and runs (what worked, what didn't, how this system behaves):
{notes}

Rules: at most {mx} changes per round; prefer a few bold, meaningful changes over many tiny ones; do not overcrowd
(fewer, stronger layers beat many faint ones); keep motion smooth; never try to create flashing.

Open and LOOK at wall.jpg first. Then write a file `RESULT.json` in this folder with exactly:
{{"score": <0-10 how good the wall looks now>, "say": "<two sentences: what you see on each surface, what you change and why>",
 "commands": [{{"cmd":"set","name":"<param>","v":<number>}}, {{"cmd":"surf","id":<id>,"content":"main","src":"rect","rect":[x,y,w,h]}}, ...],
 "notes": {{"lessons": ["<what you learned about how a setting/surface behaves on the real wall>"],
            "bugs": ["<anything that behaved wrongly or surprisingly>"],
            "feature_requests": ["<a control, mode or code change that would let you (or the performer) do better>"],
            "handling": ["<tips for running this loop / this system better next time>"]}}}}
Keep notes short and concrete; only write what this round actually taught you (empty lists are fine).
Use "commands": [] if it already looks great. Also print the same JSON as the last thing you output.
"""


def ask_agy(d, prompt_task, log):
    (d / "TASK.md").write_text(prompt_task, encoding="utf-8")
    env = dict(os.environ, PYTHONUTF8="1")
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "agy_neon_runner.py"), "--idle", "240", "--timeout", "900",
                        "--cwd", str(d)], input="Read TASK.md in this folder and execute it fully.",
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=1000)
    (d / "agy_out.txt").write_text(p.stdout + "\n--- stderr\n" + p.stderr, encoding="utf-8")
    src = (d / "RESULT.json").read_text(encoding="utf-8") if (d / "RESULT.json").exists() else p.stdout
    for m in reversed(list(re.finditer(r"\{.*\"score\".*\}", src, re.S))):
        txt = m.group(0)
        for end in range(len(txt), 0, -1):          # trim trailing junk after the JSON
            if txt[end - 1] == "}":
                try:
                    return json.loads(txt[:end])
                except Exception:
                    continue
    log.write("  agy: no parsable RESULT\n")
    return None


def notes_text():
    try:
        return NOTES.read_text(encoding="utf-8")[-NOTES_FEED:] or "(empty: first run)"
    except FileNotFoundError:
        return "(empty: first run)"


def note(run, r, score, res, applied=None):
    """Append this round to the persistent notebook (local only)."""
    n = res.get("notes") if isinstance(res.get("notes"), dict) else {}
    lines = [f"\n## {time.strftime('%Y-%m-%d %H:%M')} run {run.name} round {r}: score {score}",
             f"- said: {str(res.get('say', ''))[:400]}"]
    if applied:
        lines.append(f"- applied: {', '.join(applied)[:400]}")
    for key in ("lessons", "bugs", "feature_requests", "handling"):
        for item in (n.get(key) or [])[:5]:
            lines.append(f"- {key.upper()}: {str(item)[:300]}")
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    if not NOTES.exists():
        NOTES.write_text("# WALL LOOK LOOP notebook (local, gitignored)\nagy/Gemini Neon's running notes from the camera "
                         "loop: lessons, bugs, feature requests, handling tips. Fed back each round; the coordinator turns "
                         "BUGS / FEATURE_REQUESTS into HANDOFF tasks.\n", encoding="utf-8")
    with open(NOTES, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--goal", default=DEFAULT_GOAL)
    a = ap.parse_args()
    run = ROOT / ".wall_loop" / time.strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    log = open(run / "log.txt", "a", encoding="utf-8", buffering=1)
    post({"cmd": "camera", "op": "on"})
    post({"cmd": "automap", "op": "watch", "on": True})
    time.sleep(3)
    best, best_score, prev = snapshot(state()), -1.0, ""
    for r in range(1, a.rounds + 1):
        d = run / f"round{r}"
        d.mkdir()
        post({"cmd": "automap", "op": "watch", "on": True})   # renew the live view (its own 15-min stop)
        time.sleep(2.5)
        if not picture(d / "wall.jpg"):
            log.write("no camera picture: stopping\n")
            break
        s = state()
        cur = snapshot(s)
        surfs = "\n".join(f"{x['id']} | {x.get('name')} | " + json.dumps({f: x.get(f) for f in SURF_FIELDS if x.get(f) is not None})
                          for x in (s.get("surfaces") or {}).get("surfaces", []))
        res = ask_agy(d, TASK.format(r=r, n=a.rounds, goal=a.goal, surfs=surfs, mx=MAX_CHANGES,
                                     table=param_table(s.get("values") or {}), prev=prev, notes=notes_text()), log)
        if not res:
            continue
        score = float(res.get("score", 0))
        log.write(f"round {r}: score {score} | {res.get('say', '')}\n")
        print(f"round {r}: score {score} | {res.get('say', '')}", flush=True)
        if score < best_score - 0.5:                 # this round's look is worse than the best: go back, then refine
            restore(best)
            note(run, r, score, res, ["(made it worse: best settings restored)"])
            log.write(f"  worse than best ({best_score}): restored best settings\n")
            prev = f"Your last changes made it WORSE (score {score} < {best_score}); they were undone. Try something different."
            time.sleep(SETTLE_S)
            continue
        if score >= best_score:
            best, best_score = cur, score
        cmds = res.get("commands") or []
        if not cmds:
            note(run, r, score, res, ["(satisfied: no changes)"])
            log.write("  agy is satisfied: done\n")
            break
        done = apply(cmds, log)
        note(run, r, score, res, done)
        prev = f"Previous round you scored it {score} and applied: {', '.join(done)}. Compare and refine."
        time.sleep(SETTLE_S)
    d = run / "final"
    d.mkdir(exist_ok=True)
    picture(d / "wall.jpg")
    log.write(f"best score {best_score}\n")
    print(f"done: best score {best_score}; log {run / 'log.txt'}", flush=True)


if __name__ == "__main__":
    main()
