"""Tests for the audio details that probe() reports.

The test videos are made on the fly with FFmpeg (a test pattern plus a
generated tone), so we know exactly what audio each one should contain.
"""
import shutil
import subprocess

import pytest

from blackboard.probe import AV_TOLERANCE, VideoInfo, audio_fields, probe

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)

VIDEO = ["-f", "lavfi", "-i", "testsrc=duration={d}:size=320x240:rate=30"]
TONE = ["-f", "lavfi", "-i", "sine=frequency={f}:duration={d}:sample_rate=44100"]
ENCODE = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"]


def make_video(path, video_secs, tones, extra=()):
    """Make an mp4 with `video_secs` of video and one audio track per
    (frequency, seconds) pair in `tones`. No tones means no audio."""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    cmd += [a.format(d=video_secs) for a in VIDEO]
    for freq, secs in tones:
        cmd += [a.format(f=freq, d=secs) for a in TONE]
    cmd += ["-map", "0:v"]
    for i in range(len(tones)):
        cmd += ["-map", f"{i + 1}:a"]
    cmd += ENCODE + list(extra) + [str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    folder = tmp_path_factory.mktemp("audio_clips")
    return {
        "stereo": make_video(folder / "stereo.mp4", 2, [(440, 2)],
                             ["-ac", "2", "-b:a", "128k"]),
        "mono": make_video(folder / "mono.mp4", 2, [(440, 2)],
                           ["-ac", "1", "-ar", "48000"]),
        "two_tracks": make_video(folder / "two.mp4", 2, [(440, 2), (880, 2)]),
        "short_audio": make_video(folder / "short.mp4", 3, [(440, 1)]),
        "silent": make_video(folder / "silent.mp4", 2, []),
    }


# ---------- real files ----------

@needs_ffmpeg
def test_stereo_details(clips):
    info = probe(clips["stereo"])
    assert info.has_audio is True
    assert info.audio_codec == "aac"
    assert info.audio_sample_rate == 44100
    assert info.audio_channels == 2
    assert 100_000 <= info.audio_bitrate <= 160_000   # asked for 128 kbps
    assert info.audio_track_count == 1


@needs_ffmpeg
def test_mono_and_sample_rate(clips):
    info = probe(clips["mono"])
    assert info.audio_channels == 1
    assert info.audio_sample_rate == 48000


@needs_ffmpeg
def test_matching_length(clips):
    info = probe(clips["stereo"])
    assert info.audio_matches_video is True
    assert abs(info.audio_length_diff) <= AV_TOLERANCE


@needs_ffmpeg
def test_audio_shorter_than_video(clips):
    info = probe(clips["short_audio"])          # 1 s of audio, 3 s of video
    assert info.audio_matches_video is False
    assert info.audio_length_diff == pytest.approx(-2.0, abs=0.2)


@needs_ffmpeg
def test_two_audio_tracks(clips):
    info = probe(clips["two_tracks"])
    assert info.audio_track_count == 2
    assert info.audio_codec == "aac"            # details of the first track


@needs_ffmpeg
def test_no_audio(clips):
    info = probe(clips["silent"])
    assert info.has_audio is False
    assert info.audio_track_count == 0
    assert info.audio_codec is None
    assert info.audio_sample_rate is None
    assert info.audio_channels is None
    assert info.audio_bitrate is None
    assert info.audio_length_diff is None
    assert info.audio_matches_video is None


# ---------- no files needed ----------

def test_audio_fields_handles_missing_values():
    # Some containers leave out the bitrate or duration of an audio track
    fields = audio_fields([{"codec_type": "audio", "codec_name": "opus"}])
    assert fields["audio_codec"] == "opus"
    assert fields["audio_sample_rate"] is None
    assert fields["audio_bitrate"] is None
    assert fields["audio_duration"] is None
    assert fields["audio_track_count"] == 1


def test_audio_fields_converts_text_numbers():
    # ffprobe gives numbers as strings
    fields = audio_fields([{
        "codec_type": "audio", "codec_name": "aac", "sample_rate": "48000",
        "channels": 2, "bit_rate": "130000", "duration": "11.5",
    }])
    assert fields["audio_sample_rate"] == 48000
    assert fields["audio_bitrate"] == 130000
    assert fields["audio_duration"] == 11.5


def test_audio_fields_ignores_video_streams():
    assert audio_fields([{"codec_type": "video"}]) == {"audio_track_count": 0}


def make_info(duration, audio_duration):
    return VideoInfo(codec="h264", width=1, height=1, fps=30.0,
                     duration=duration, has_audio=True,
                     audio_duration=audio_duration)


def test_length_match_tolerance_edges():
    assert make_info(10.0, 10.0 + AV_TOLERANCE).audio_matches_video is True
    assert make_info(10.0, 10.0 - AV_TOLERANCE).audio_matches_video is True
    assert make_info(10.0, 10.0 + AV_TOLERANCE + 0.05).audio_matches_video is False


def test_length_match_unknown():
    assert make_info(10.0, None).audio_matches_video is None
    assert make_info(0.0, 5.0).audio_matches_video is None