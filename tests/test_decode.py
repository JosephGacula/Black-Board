from pathlib import Path

import numpy as np
import pytest

from blackboard.decode import read_segment, read_frame, DecodeError

SAMPLE = Path("samples/test1.mp4")
needs_sample = pytest.mark.skipif(not SAMPLE.exists(), reason="sample clip not found")


@needs_sample
def test_segment_returns_the_requested_frames():
    frames = list(read_segment(SAMPLE, 10, 19))
    assert [f.index for f in frames] == list(range(10, 20))


@needs_sample
def test_frame_shape_matches_video():
    frame = read_frame(SAMPLE, 0)
    assert frame.image.shape == (1008, 1920, 3)


@needs_sample
def test_single_frame_segment():
    frames = list(read_segment(SAMPLE, 5, 5))
    assert len(frames) == 1


@needs_sample
def test_step_skips_frames():
    frames = list(read_segment(SAMPLE, 0, 20, step=5))
    assert [f.index for f in frames] == [0, 5, 10, 15, 20]


@needs_sample
def test_first_and_last_frame():
    assert read_frame(SAMPLE, 0).index == 0
    assert read_frame(SAMPLE, 346).index == 346   # clip has 347 frames


@needs_sample
def test_seeking_matches_sequential_decoding():
    # Frame 250 needs decoding from keyframe 190. It should be identical
    # to frame 250 reached by decoding straight through from frame 0.
    last = None
    for frame in read_segment(SAMPLE, 0, 250):
        last = frame
    assert np.array_equal(last.image, read_frame(SAMPLE, 250).image)


@needs_sample
def test_start_after_end_raises():
    with pytest.raises(DecodeError):
        list(read_segment(SAMPLE, 20, 10))


@needs_sample
def test_frame_past_the_end_raises():
    with pytest.raises(DecodeError):
        list(read_segment(SAMPLE, 0, 100000))


def test_non_video_file_raises(tmp_path):
    fake = tmp_path / "fake.mp4"
    fake.write_text("this is not a video")
    with pytest.raises(DecodeError):
        list(read_segment(fake, 0, 1))