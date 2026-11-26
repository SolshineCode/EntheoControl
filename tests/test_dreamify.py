"""DREAMIFY: real DeepDream as an offline ai_worker job. Fakes only (no torch, no GPU), except test_real_dream_small,
which needs DREAM_REAL=1 (one small real GoogLeNet run; take .locks/gpu_suite first)."""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import ai_worker  # noqa: E402
import dreamify  # noqa: E402


# ------------------------------------------------------------------ pure helpers
def test_presets_cover_the_four_looks():
    assert set(dreamify.PRESETS) == {"dogslugs", "eyes", "architecture", "psychedelic swirl"}
    assert set(dreamify.PRESETS["dogslugs"]["layers"]) == {"inception4c", "inception4d"}
    low = dreamify.PRESETS["architecture"]["layers"]
    assert all(n.startswith("inception3") or n == "inception4a" for n in low)
    assert dreamify.PRESETS["psychedelic swirl"]["octaves"] > max(
        p["octaves"] for k, p in dreamify.PRESETS.items() if k != "psychedelic swirl")


def test_octave_sizes_end_at_full_size_and_shrink_geometrically():
    s = dreamify.octave_sizes(768, 480, 4, 1.4)
    assert s[-1] == (768, 480) and len(s) == 4
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(s, s[1:]))
    tiny = dreamify.octave_sizes(100, 80, 8, 1.5)
    assert all(min(x) >= dreamify.MIN_SIDE for x in tiny[:-1]) and tiny[-1] == (100, 80)


def test_tiles_cover_every_pixel():
    for w, h, t in ((768, 480, 512), (300, 200, 512), (1000, 1000, 384)):
        boxes = dreamify.tile_boxes(w, h, t)
        for x0, y0, x1, y1 in boxes:
            assert x1 - x0 <= t and y1 - y0 <= t
        for y in list(range(0, h, 7)) + [h - 1]:
            for x in list(range(0, w, 7)) + [w - 1]:
                assert sum(1 for x0, y0, x1, y1 in boxes if x0 <= x < x1 and y0 <= y < y1) == 1


def test_fit_size_caps_long_side():
    assert dreamify.fit_size(2000, 1000, 768) == (768, 384)
    assert dreamify.fit_size(300, 200, 768) == (300, 200)


def test_iterations_grow_with_strength():
    assert dreamify.iterations(0.2) < dreamify.iterations(1.0)
    assert dreamify.iterations(0.0) >= 2


@pytest.fixture
def dirs(tmp_path):
    art, tex = tmp_path / "art", tmp_path / "textures"
    art.mkdir()
    tex.mkdir()
    Image.new("RGB", (96, 64), (200, 40, 90)).save(art / "My piece.png")
    Image.new("RGB", (64, 64), (20, 140, 200)).save(tex / "gen_20261125_1_2.png")
    return art, tex


def test_validate_accepts_library_names_and_rejects_paths(dirs):
    art, tex = dirs
    j = dreamify.validate({"name": "My piece.png", "source": "art", "strength": 0.6, "octaves": 5, "preset": "eyes"},
                          art, tex)
    assert j["kind"] == "dream" and j["octaves"] == 5 and j["preset"] == "eyes" and j["n"] == 1
    j = dreamify.validate({"name": "gen_20261125_1_2.png", "source": "ai"}, art, tex)
    assert j["preset"] == "dogslugs" and j["octaves"] == dreamify.PRESETS["dogslugs"]["octaves"]
    bad = [{"name": "../x.png", "source": "art"}, {"name": "My piece.png", "source": "ai"},
           {"name": "nope.png", "source": "art"}, {"name": "My piece.png", "source": "web"},
           {"name": "My piece.png", "source": "art", "preset": "dogs"},
           {"name": "My piece.png", "source": "art", "strength": 3},
           {"name": "My piece.png", "source": "art", "octaves": 0}, {"name": "x.depth.png", "source": "art"},
           {"name": "C:\\Windows\\a.png", "source": "art"}, "x"]
    for b in bad:
        with pytest.raises(ValueError):
            dreamify.validate(b, art, tex)


def test_output_names_land_in_the_right_library():
    d = dreamify.out_name("art", "My piece.png", "psychedelic swirl", "20261125_120000")
    assert d == "My_piece_dream_psychedelic_swirl_20261125_120000.png"
    a = dreamify.out_name("ai", "gen_20261125_1_2.png", "eyes", "20261125_120000")
    assert a.startswith("gen_dream_") and a.endswith(".png")
    import remote_control
    assert remote_control.TEX_NAME.match(a)


# ------------------------------------------------------------------ worker integration with fakes
class FakeDreamer:
    calls = []

    def __init__(self, device):
        self.device = device

    def dream(self, img, preset, strength, octaves, max_side, progress, cancelled):
        FakeDreamer.calls.append(dict(device=self.device, max_side=max_side, preset=preset, octaves=octaves))
        total = octaves * 3
        for i in range(total):
            if cancelled():
                return None
            progress(i // 3, i + 1, total)
            time.sleep(0.01)
        return img.convert("RGB").resize(dreamify.fit_size(*img.size, max_side))

    def release(self):
        pass


class FakePipes:
    base = "fake"

    def __init__(self):
        self.offloads = self.restores = 0

    def offload(self):
        self.offloads += 1

    def restore(self):
        self.restores += 1


def _worker(dirs, free_mb, pipes=None, slow=False):
    art, tex = dirs
    FakeDreamer.calls = []
    w = ai_worker.Worker(tex, tex / "snap.png", "tok", port=0, pipes_factory=lambda: pipes or FakePipes(),
                         art_dir=art, dream_factory=FakeDreamer, vram_free=lambda: free_mb)
    return w


def _wait(w, pred, t=10.0):
    end = time.time() + t
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_dream_job_writes_atomically_with_metadata_and_reports_done(dirs):
    art, tex = dirs
    w = _worker(dirs, 3000)
    w.start()
    try:
        jid = w.submit(dreamify.validate({"name": "My piece.png", "source": "art", "preset": "eyes"}, art, tex))
        assert _wait(w, lambda: any(d.get("kind") == "dream" for d in w.status()["done"]))
        d = [d for d in w.status()["done"] if d.get("kind") == "dream"][0]
        assert d["id"] == jid and (art / d["name"]).is_file() and d["source"] == "art"
        with Image.open(art / d["name"]) as im:
            assert im.text["dream_preset"] == "eyes" and im.text["dream_source"] == "My piece.png"
            assert "googlenet" in im.text["dream_model"].lower()
        assert not list(art.glob("*.tmp")) and w.status()["dream"] is None
        assert FakeDreamer.calls[0]["device"] == "cuda" and FakeDreamer.calls[0]["max_side"] == 768
    finally:
        w.stop()


def test_ai_source_goes_to_textures_as_gen_dream(dirs):
    art, tex = dirs
    w = _worker(dirs, 3000)
    w.start()
    try:
        w.submit(dreamify.validate({"name": "gen_20261125_1_2.png", "source": "ai"}, art, tex))
        assert _wait(w, lambda: list(tex.glob("gen_dream_*.png")))
    finally:
        w.stop()


def test_progress_is_reported_while_running(dirs):
    art, tex = dirs
    w = _worker(dirs, 3000)
    seen = []
    w.start()
    try:
        w.submit(dreamify.validate({"name": "My piece.png", "source": "art", "octaves": 8}, art, tex))
        end = time.time() + 5
        while time.time() < end and not any(d.get("kind") == "dream" for d in w.status()["done"]):
            s = w.status()["dream"]
            if s:
                seen.append(s)
            time.sleep(0.01)
        assert seen and all(0.0 <= s["pct"] <= 100.0 for s in seen)
        assert {"octave", "octaves", "device", "name", "preset"} <= set(seen[-1])
    finally:
        w.stop()


def test_low_vram_offloads_sd_turbo_then_restores(dirs):
    art, tex = dirs
    pipes = FakePipes()
    free = {"mb": 500}

    class W(ai_worker.Worker):
        pass

    w = ai_worker.Worker(tex, tex / "s.png", "tok", port=0, pipes_factory=lambda: pipes, art_dir=art,
                         dream_factory=FakeDreamer, vram_free=lambda: free["mb"])
    FakeDreamer.calls = []
    orig = pipes.offload

    def off():
        orig()
        free["mb"] = 2600                 # moving SD-Turbo to RAM frees ~2 GB
    pipes.offload = off
    w.start()
    try:
        w.q.put(("warmup", None, -1))      # SD-Turbo is resident
        w.submit(dreamify.validate({"name": "My piece.png", "source": "art"}, art, tex))
        assert _wait(w, lambda: pipes.restores == 1)
        assert pipes.offloads == 1 and FakeDreamer.calls[0]["device"] == "cuda"
    finally:
        w.stop()


def test_no_vram_falls_back_to_cpu_small(dirs):
    art, tex = dirs
    w = _worker(dirs, 200)
    w.start()
    try:
        w.submit(dreamify.validate({"name": "My piece.png", "source": "art"}, art, tex))
        assert _wait(w, lambda: FakeDreamer.calls)
        assert FakeDreamer.calls[0]["device"] == "cpu" and FakeDreamer.calls[0]["max_side"] <= 384
    finally:
        w.stop()


def test_no_cuda_at_all_runs_on_cpu(dirs):
    w = _worker(dirs, None)
    art, tex = dirs
    w.start()
    try:
        w.submit(dreamify.validate({"name": "My piece.png", "source": "art"}, art, tex))
        assert _wait(w, lambda: FakeDreamer.calls)
        assert FakeDreamer.calls[0]["device"] == "cpu"
    finally:
        w.stop()


def test_cancel_stops_a_running_dream_without_writing(dirs):
    art, tex = dirs
    w = _worker(dirs, 3000)
    w.start()
    try:
        w.submit(dreamify.validate({"name": "My piece.png", "source": "art", "octaves": 8}, art, tex))
        assert _wait(w, lambda: w.status()["dream"] is not None)
        w.cancel_dream()
        assert _wait(w, lambda: w.status()["dream"] is None and not w.status()["busy"])
        time.sleep(0.1)
        assert not list(art.glob("*dream*")) and "cancel" in (w.status()["error"] or "").lower()
    finally:
        w.stop()


def test_http_dreamify_endpoint(dirs):
    art, tex = dirs
    w = _worker(dirs, 3000)
    w.start()
    try:
        def post(path, body, tok="tok"):
            req = urllib.request.Request(f"http://127.0.0.1:{w.port}{path}", data=json.dumps(body).encode(),
                                         headers={"X-Token": tok, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=3) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as ex:
                return ex.code, json.loads(ex.read())
        code, out = post("/dreamify", {"name": "My piece.png", "source": "art", "preset": "architecture"})
        assert code == 200 and "id" in out
        assert post("/dreamify", {"name": "../../x.png", "source": "art"})[0] == 400
        assert post("/dreamify", {"name": "My piece.png", "source": "art"}, tok="bad")[0] == 403
        assert post("/dream-cancel", {})[0] == 200
    finally:
        w.stop()


def test_remote_control_validates_and_proxies_dreamify():
    import remote_control
    ok = remote_control.validate_dream_request({"name": "My piece.png", "source": "art", "preset": "eyes",
                                                "strength": 0.5, "octaves": 4, "junk": 1})
    assert ok == {"name": "My piece.png", "source": "art", "preset": "eyes", "strength": 0.5, "octaves": 4}
    for bad in ({"name": "a/b.png", "source": "art"}, {"name": "x.png", "source": "zzz"}, {"source": "art"}):
        with pytest.raises(ValueError):
            remote_control.validate_dream_request(bad)


# ------------------------------------------------------------------ one real run (opt-in)
@pytest.mark.skipif(os.environ.get("DREAM_REAL") != "1", reason="set DREAM_REAL=1 (GPU lock!)")
def test_real_dream_small(tmp_path):
    img = Image.open(HERE / "test_frames" / "gallery" / "03_5_layers_mandala_all.png").convert("RGB")
    d = dreamify.RealDream("cuda")
    try:
        out = d.dream(img, "dogslugs", 0.4, 3, 384, lambda *a: None, lambda: False)
    finally:
        d.release()
    assert out.size[0] == 384
    out.save(tmp_path / "real.png")
