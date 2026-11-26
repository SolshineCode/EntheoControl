"""recorder.py: every path derived from meta.json is untrusted (Neon PR1 B2 Recorder.recover, PR2 B9 finalize_parts)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import recorder as rec  # noqa: E402

BS = chr(92)
EVIL = ["../../pwned.mp4", ".." + BS + ".." + BS + "pwned.mp4", "/abs/pwned.mp4", "C:" + BS + "Windows" + BS + "pwned.mp4",
        "C:pwned.mp4", "sub/dir.mp4", "..mp4", "a..b.mp4", "live_visuals.py", "show.mp4.py", "con" + chr(0) + ".mp4", "",
        "x" * 300 + ".mp4", "nul:.mp4", BS + BS + "server" + BS + "share" + BS + "x.mp4", ".hidden.mp4", "semi;colon.mp4",
        "pipe|.mp4", 5, None, ["a.mp4"], {"n": 1}, True]


def make_parts(tmp_path, name, **meta):
    folder = tmp_path / "recordings"
    parts = folder / ".parts" / "show_20261125_200000"
    parts.mkdir(parents=True)
    (parts / "video.mp4").write_bytes(b"v" * 64)
    (parts / "meta.json").write_text(json.dumps(dict(meta, name=name)), encoding="utf-8")
    return folder, parts


def fake_mux(written):
    def run(ffmpeg, popen, video, pcm, out, logf, **kw):
        written.append(Path(out))
        Path(out).write_bytes(b"muxed")
        return True
    return run


@pytest.mark.parametrize("name", EVIL)
def test_safe_output_name_never_leaves_the_folder(tmp_path, name):
    out = rec.safe_output_name(name, "show_1.mp4", tmp_path)
    assert out == "show_1.mp4"
    assert (tmp_path / out).resolve().parent == tmp_path.resolve()


def test_safe_output_name_keeps_ordinary_names_and_survives_a_bad_fallback(tmp_path):
    for good in ("show_20261125_200000.mp4", "My set (2).mp4", "replay-01.MP4"):
        assert rec.safe_output_name(good, "x.mp4", tmp_path) == good
    assert rec.safe_output_name("../a.mp4", "../../b.mp4", tmp_path) == "recovered.mp4"


@pytest.mark.parametrize("name", EVIL)
def test_finalize_parts_writes_only_inside_recordings(tmp_path, monkeypatch, name):
    folder, parts = make_parts(tmp_path, name, fps="thirty", frames=[1], sealed=True)      # junk numbers too
    victim = tmp_path / "pwned.mp4"
    written = []
    monkeypatch.setattr(rec, "run_mux", fake_mux(written))
    status, msg, secs = rec.finalize_parts(parts, "ffmpeg", disk_free=lambda p: 1 << 40, log=lambda *a: None)
    assert status == "ok", msg
    assert len(written) == 1 and written[0].resolve().parent == folder.resolve()
    assert written[0].name == "show_20261125_200000.mp4"                               # the parts folder's own name
    assert not victim.exists() and not list(tmp_path.glob("*.mp4")) and not list(tmp_path.parent.glob("pwned*"))


def test_recorder_recover_on_startup_cannot_be_steered_outside(tmp_path, monkeypatch):
    folder, parts = make_parts(tmp_path, "../../startup_pwned.mp4", fps=30, frames=30)
    written = []
    monkeypatch.setattr(rec, "run_mux", fake_mux(written))
    r = rec.Recorder(folder, ffmpeg="ffmpeg") if "ffmpeg" in rec.Recorder.__init__.__code__.co_varnames else rec.Recorder(folder)
    r.ffmpeg = "ffmpeg"
    r.disk_free = lambda p: 1 << 40
    try:
        r.recover(wait=True)
    finally:
        getattr(r, "close", lambda: None)()
    assert written and all(w.resolve().parent == folder.resolve() for w in written)
    assert not (tmp_path.parent / "startup_pwned.mp4").exists() and not (tmp_path / "startup_pwned.mp4").exists()
