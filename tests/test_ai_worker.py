import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ai_worker as aw  # noqa: E402


class FakePipes:
    """Stands in for diffusers: returns solid images, records calls."""

    def __init__(self, delay=0.0, fail=False):
        self.calls, self.delay, self.fail = [], delay, fail

    def txt2img(self, prompt, negative, seed, size):
        self.calls.append(("txt2img", prompt, negative, seed))
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("CUDA out of memory (fake)")
        return Image.new("RGB", (size, size), (200, 50, 120))

    def img2img(self, prompt, negative, seed, init, strength):
        self.calls.append(("img2img", prompt, negative, seed, strength, init.size))
        return Image.new("RGB", init.size, (20, 180, 90))

    def depth(self, img):
        self.calls.append(("depth", img.size))
        import numpy as np
        h, w = img.size[1], img.size[0]
        return np.tile(np.linspace(0.0, 1.0, w, dtype="float32"), (h, 1))   # float depth, 1 = near


@pytest.fixture
def worker(tmp_path):
    fake = FakePipes()
    w = aw.Worker(out_dir=tmp_path / "textures", snapshot=tmp_path / "snap.png", token="tok",
                  port=0, pipes_factory=lambda: fake, art_dir=tmp_path / "art")
    w.start()
    w.fake = fake
    yield w
    w.stop()


def req(w, path, body=None, token="tok"):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"http://127.0.0.1:{w.port}{path}", data=data, method="POST" if data else "GET",
                               headers={"X-Token": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=5) as resp:
        return resp.status, json.loads(resp.read() or b"{}")


def wait_done(w, n, timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        if len(w.status()["done"]) >= n:
            return
        time.sleep(0.05)
    raise AssertionError(f"timeout: {w.status()}")


def test_rejects_missing_or_bad_token(worker):
    for tok in ("", "nope"):
        with pytest.raises(urllib.error.HTTPError) as e:
            req(worker, "/status", token=tok)
        assert e.value.code == 403


def test_txt2img_job_writes_png_atomically_with_metadata(worker):
    code, body = req(worker, "/job", {"prompt": "kene lattice neon", "n": 2, "seed": 7})
    assert code == 200 and body["id"]
    wait_done(worker, 2)
    files = sorted((worker.out_dir).glob("gen_*.png"))
    assert len(files) == 2
    assert not list(worker.out_dir.glob("*.tmp*")), "temp files left behind"
    im = Image.open(files[0])
    assert im.info.get("prompt") == "kene lattice neon"
    seeds = [c[3] for c in worker.fake.calls]
    assert seeds == [7, 8]
    st = req(worker, "/status")[1]
    assert st["loaded"] and not st["busy"] and st["done"][-1]["prompt"] == "kene lattice neon"


@pytest.mark.parametrize("bad", [
    {"prompt": ""}, {"prompt": "x" * 501}, {"prompt": "ok", "n": 0}, {"prompt": "ok", "n": 9},
    {"prompt": "ok", "strength": 1.5}, {"prompt": "ok", "seed": "abc"}, {"prompt": 5}, ["not", "a", "dict"],
    {"prompt": "ok", "init": "/etc/passwd"},
])
def test_invalid_jobs_rejected(worker, bad):
    with pytest.raises(urllib.error.HTTPError) as e:
        req(worker, "/job", bad)
    assert e.value.code == 400


def test_snapshot_img2img_uses_snapshot_file(worker):
    Image.new("RGB", (768, 480), (0, 0, 255)).save(worker.snapshot)
    req(worker, "/job", {"prompt": "melt it", "init": "snapshot", "strength": 0.45, "seed": 1})
    wait_done(worker, 1)
    kind, prompt, _, seed, strength, size = worker.fake.calls[0]
    assert kind == "img2img" and strength == 0.45
    assert size[0] % 8 == 0 and size[1] % 8 == 0 and max(size) <= 768


def test_snapshot_missing_reports_error_and_keeps_running(worker):
    req(worker, "/job", {"prompt": "melt it", "init": "snapshot"})
    t = time.time()
    while not worker.status()["error"] and time.time() - t < 5:
        time.sleep(0.05)
    assert "snapshot" in worker.status()["error"]
    req(worker, "/job", {"prompt": "still alive", "n": 1})
    wait_done(worker, 1)


def test_pipeline_failure_is_reported_not_fatal(tmp_path):
    fake = FakePipes(fail=True)
    w = aw.Worker(out_dir=tmp_path / "t", snapshot=tmp_path / "s.png", token="tok", port=0, pipes_factory=lambda: fake)
    w.start()
    try:
        req(w, "/job", {"prompt": "boom"})
        t = time.time()
        while not w.status()["error"] and time.time() - t < 5:
            time.sleep(0.05)
        assert "out of memory" in w.status()["error"]
        t = time.time()
        while w.status()["busy"] and time.time() - t < 60:   # OOM cleanup (torch import) may take a while
            time.sleep(0.1)
        assert not w.status()["busy"]
    finally:
        w.stop()


def test_cancel_clears_queue(tmp_path):
    fake = FakePipes(delay=0.3)
    w = aw.Worker(out_dir=tmp_path / "t", snapshot=tmp_path / "s.png", token="tok", port=0, pipes_factory=lambda: fake)
    w.start()
    try:
        for i in range(4):
            req(w, "/job", {"prompt": f"p{i}", "n": 2})
        req(w, "/cancel", {})
        time.sleep(1.0)
        assert len(fake.calls) <= 2          # at most the in-flight image(s) of the first job
        assert w.status()["queue"] == []
    finally:
        w.stop()


def test_queue_limit(tmp_path):
    fake = FakePipes(delay=0.5)
    w = aw.Worker(out_dir=tmp_path / "t", snapshot=tmp_path / "s.png", token="tok", port=0, pipes_factory=lambda: fake)
    w.start()
    try:
        codes = []
        for i in range(aw.MAX_QUEUE + 3):
            try:
                codes.append(req(w, "/job", {"prompt": f"p{i}"})[0])
            except urllib.error.HTTPError as e:
                codes.append(e.code)
        assert 503 in codes
    finally:
        w.stop()


def test_token_file_roundtrip(tmp_path):
    f = tmp_path / ".remote_token"
    assert aw.read_token(f) is None
    f.write_text("abc123\n")
    assert aw.read_token(f) == "abc123"


# ---------------------------------------------------------------- depth maps for uploaded art

def wait_for(pred, timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_depth_job_writes_16bit_depth_png_next_to_art(worker):
    worker.art_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (300, 200), (90, 40, 160)).save(worker.art_dir / "piece_one.png")
    code, body = req(worker, "/depth", {"name": "piece_one.png"})
    assert code == 200 and body["id"]
    out = worker.art_dir / "piece_one.depth.png"
    assert wait_for(out.exists)
    d = Image.open(out)
    assert d.size == (300, 200) and d.mode in ("I;16", "I", "I;16B")
    import numpy as np
    a = np.asarray(d)
    assert a[:, 0].mean() < 1000 and a[:, -1].mean() > 64000       # normalized 0..65535, near = white
    assert not list(worker.art_dir.glob("*.tmp*"))
    assert wait_for(lambda: any(x.get("kind") == "depth" for x in worker.status()["done"]))


@pytest.mark.parametrize("name", ["../secret.png", "nope.png", "a/b.png", "x.depth.png", "", 5, "piece.exe"])
def test_depth_rejects_bad_names(worker, name):
    worker.art_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(urllib.error.HTTPError) as e:
        req(worker, "/depth", {"name": name})
    assert e.value.code == 400


def test_depth_does_not_need_diffusion_model_loaded_first(tmp_path):
    """Depth jobs work even if the diffusion model failed to load (separate, lazy)."""
    class DepthOnly:
        def depth(self, img):
            import numpy as np
            return np.zeros((img.size[1], img.size[0]), "float32")
    w = aw.Worker(out_dir=tmp_path / "t", snapshot=tmp_path / "s.png", token="tok", port=0,
                  pipes_factory=lambda: (_ for _ in ()).throw(RuntimeError("no diffusion")),
                  depth_factory=lambda: DepthOnly(), art_dir=tmp_path / "art")
    w.start()
    try:
        w.art_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64)).save(w.art_dir / "p.png")
        req(w, "/depth", {"name": "p.png"})
        assert wait_for((w.art_dir / "p.depth.png").exists)
    finally:
        w.stop()


def test_depth_respects_exif_rotation(worker):
    worker.art_dir.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (300, 200), (90, 40, 160))
    exif = Image.Exif(); exif[0x0112] = 6                       # rotate 90 CW on display
    im.save(worker.art_dir / "phone.jpg", exif=exif.tobytes())
    req(worker, "/depth", {"name": "phone.jpg"})
    out = worker.art_dir / "phone.depth.png"
    assert wait_for(out.exists)
    assert Image.open(out).size == (200, 300)                  # depth matches the displayed orientation
