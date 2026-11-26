"""Impact audit: how much does each param change the picture?

Renders every params.py entry at min and max (everything else at its default) headlessly through
live_visuals.py --batch and reports the mean absolute pixel difference against the default picture, in percent
of full scale. Params that only act inside another feature are measured inside it (CONTEXT below: e.g. the
serpent faders with serpents on, the art faders with the art layer up). Nothing is shown on a monitor.

    .venv\\Scripts\\python.exe impact_audit.py [--out test_frames/impact] [--frames 100] [--tag before]

Writes <out>/impact_<tag>.json and prints the table, weakest first. Under 2 % reads as "does nothing" from
across a room.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from params import PARAMS, TRIGGERS  # noqa: E402

# param -> the settings it needs around it to be visible at all
CONTEXT = {
    "flow_dir": dict(symmetry=0.0), "folds": {}, "luma_key": dict(l_video=1.0),
    "travel_mode": dict(travel=0.6), "tunnel_curve": dict(travel=0.6, travel_mode=1.0), "depth_fog": dict(travel=0.6, travel_mode=1.0),
    "spin_twist": dict(spin=0.5, travel=0.6),
    "serpent_mode": dict(serpent=1.0), "serpent_count": dict(serpent=1.0), "serpent_segments": dict(serpent=1.0),
    "serpent_speed": dict(serpent=1.0), "serpent_wiggle": dict(serpent=1.0), "serpent_size": dict(serpent=1.0),
    "chase_width": dict(chase=1.0), "chase_rate": dict(chase=1.0), "chase_mode": dict(chase=1.0),
    "ripple": "ripple", "ripple_decay": "ripple",
    "art_source": dict(l_art=1.0, art_use=2.0), "art_use": dict(l_art=1.0),
    "art_combine": dict(l_art=1.0, art_source=1.0, art_mix_stencil=1.0, art_mix_kaleido=1.0),
    **{f"art_mix_{m}": dict(l_art=1.0, art_source=1.0, art_combine=1.0)
       for m in ("relief", "stencil", "kaleido", "tunnel", "displace", "melt")},
    "swarm_vortex": dict(l_swarm=1.0), "swarm_turb": dict(l_swarm=1.0), "swarm_trail": dict(l_swarm=1.0), "swarm_tint": dict(l_swarm=1.0),
}
GROUP_CONTEXT = {"fractal": dict(l_fractal=1.0), "art": dict(l_art=1.0), "land": dict(l_land=1.0)}
SKIP = {"master", "blackout"}                       # trivially total


def context(name):
    c = CONTEXT.get(name)
    if c is None:
        c = GROUP_CONTEXT.get(PARAMS[name][3], {})
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "test_frames" / "impact"))
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--render", default="640x400")
    ap.add_argument("--art", default=None, help="art folder (default: procedural test paintings written under --out)")
    ap.add_argument("--only", default=None, help="comma-separated param names")
    a = ap.parse_args()
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    art = a.art
    if art is None:
        import art_layer
        art = out.parent / "art"
        art_layer.write_test_art(art)
    names = [n for n in PARAMS if n not in SKIP and (not a.only or n in a.only.split(","))]
    jobs, plan = [dict(name="warmup", sets=dict(l_art=1.0, l_video=1.0), frames=600)], {}     # lets the art decoder finish
    ripple = {"40": [["ripple", 0.5, 0.5]], "70": [["ripple", 0.3, 0.6]]}
    for n in names:
        lo, hi, dflt = PARAMS[n][:3]
        ctx = context(n)
        script = ripple if ctx == "ripple" else {}
        sets = {} if ctx == "ripple" else {k: v for k, v in ctx.items() if k in PARAMS}
        key = json.dumps([sets, script], sort_keys=True)
        base = f"base{abs(hash(key)) % 10 ** 8}"
        if base not in {j["name"] for j in jobs}:
            jobs.append(dict(name=base, sets=sets, frames=a.frames, script=script))
        plan[n] = dict(base=base, ends={})
        for tag, v in (("min", lo), ("mid", (lo + hi) / 2), ("max", hi)):
            if v == dflt and n not in sets:
                continue
            if n in TRIGGERS:                                  # held from frame 60, like a finger on the pad
                job = dict(name=f"{n}_{tag}", sets=sets, frames=a.frames, script={**script, "60": [["trigger", n, bool(v)]]})
            else:
                job = dict(name=f"{n}_{tag}", sets={**sets, n: v}, frames=a.frames, script=script)
            jobs.append(job)
            plan[n]["ends"][tag] = job["name"]
    spec = out / "jobs.json"
    spec.write_text(json.dumps(jobs))
    r = subprocess.run([sys.executable, str(HERE / "live_visuals.py"), "--batch", str(spec), "--out", str(out), "--render", a.render,
                        "--art", str(art), "--videos", str(out.parent / "video")], capture_output=True, text=True, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(r.stderr[-3000:])
    img = lambda nm: np.asarray(Image.open(out / ("".join(ch if ch.isalnum() else "_" for ch in nm) + ".png"))).astype(np.float32)
    rows = []
    for n, pl in plan.items():
        b = img(pl["base"])
        d = {tag: float(np.abs(img(j) - b).mean() / 255.0 * 100.0) for tag, j in pl["ends"].items()}
        rows.append(dict(name=n, group=PARAMS[n][3], min=d.get("min"), mid=d.get("mid"), max=d.get("max"), best=max(d.values()) if d else 0.0))
    rows.sort(key=lambda x: x["best"])
    (out.parent / f"impact_{a.tag}.json").write_text(json.dumps(rows, indent=1))
    fmt = lambda v: "   -  " if v is None else f"{v:6.2f}"
    print(f"{'param':18s} {'group':8s} {'min %':>6s} {'mid %':>6s} {'max %':>6s}")
    for x in rows:
        print(f"{x['name']:18s} {x['group']:8s} {fmt(x['min'])} {fmt(x['mid'])} {fmt(x['max'])}{'   <-- weak' if x['best'] < 2.0 else ''}")
    print(f"[impact] {len(rows)} params, {sum(x['best'] < 2.0 for x in rows)} under 2 % -> {out.parent / f'impact_{a.tag}.json'}")


if __name__ == "__main__":
    main()
