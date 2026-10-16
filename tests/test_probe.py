from pathlib import Path

import pytest

from blackboard.probe import probe, ProbeError

SAMPLE = Path("samples/test1.mp4")
needs_sample = pytest.mark.skipif(not SAMPLE.exists(), reason="sample clip not found")


@needs_sample
def test_basic_properties():
    info = probe(SAMPLE)
    assert info.codec == "h264"
    assert (info.width, info.height) == (1920, 1008)
    assert info.resolution == "1920x1008"
    assert info.fps == pytest.approx(29.97, abs=0.01)
    assert info.frame_count == 347
    assert info.has_audio is True


@needs_sample
def test_cover_art_stream_is_ignored():
    info = probe(SAMPLE)
    assert (info.width, info.height) != (1280, 720)


@needs_sample
def test_gop_adds_up_to_frame_count():
    info = probe(SAMPLE)
    assert info.keyframe_count * info.avg_gop == pytest.approx(info.frame_count)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ProbeError):
        probe(tmp_path / "nope.mp4")


def test_zero_byte_file_raises(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(ProbeError):
        probe(empty)


def test_text_file_renamed_mp4_raises(tmp_path):
    fake = tmp_path / "fake.mp4"
    fake.write_text("this is not a video")
    with pytest.raises(ProbeError):
        probe(fake)