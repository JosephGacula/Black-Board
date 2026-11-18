"""Tests for the platform checks: does a video suit where it will be posted?"""
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from blackboard.platforms import (
    DEFAULT_KEYS, FAIL, GB, MB, OK, PLACEMENTS, PLACEMENTS_BY_KEY, WARN, Clip,
    check_file_size, check_file_type, check_frame_rate, check_length,
    check_places, check_placement, check_resolution, check_shape, clip_from,
    clip_summary, display_size, duration_text, read_display_geometry,
    shape_name, size_text,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not installed",
)


def clip(width=1080, height=1920, fps=30.0, duration=30.0, size=20 * MB, ext="mp4"):
    return Clip(width, height, fps, duration, size, ext)


def place(key):
    return PLACEMENTS_BY_KEY[key]


# ---------- the table itself ----------

def test_keys_are_unique_and_the_defaults_exist():
    keys = [p.key for p in PLACEMENTS]
    assert len(keys) == len(set(keys))
    assert all(k in keys for k in DEFAULT_KEYS)


def test_every_ideal_shape_is_inside_the_accepted_range():
    for p in PLACEMENTS:
        for top, bottom in p.ideal:
            assert p.accept_min <= top / bottom <= p.accept_max, p.key


def test_limits_are_in_a_sensible_order():
    for p in PLACEMENTS:
        if p.length_soft and p.length_hard:
            assert p.length_soft <= p.length_hard, p.key
        if p.size_soft and p.size_hard:
            assert p.size_soft <= p.size_hard, p.key
        if p.length_min and p.length_hard:
            assert p.length_min < p.length_hard, p.key
        if p.min_short:
            assert p.min_short <= p.good_short, p.key


# ---------- words for numbers ----------

@pytest.mark.parametrize("ratio, name", [
    (16 / 9, "16:9"), (9 / 16, "9:16"), (1.0, "1:1"), (0.8, "4:5"),
    (1920 / 1008, "1.91:1"), (1.37, "1.37:1"), (0.6, "1:1.67"),
])
def test_shape_names(ratio, name):
    assert shape_name(ratio) == name


@pytest.mark.parametrize("seconds, text", [
    (0.5, "0.5 s"), (45, "45 s"), (60, "1:00"), (185, "3:05"),
    (3600, "1 h 00 min"), (3725, "1 h 02 min"),
])
def test_duration_text(seconds, text):
    assert duration_text(seconds) == text


def test_size_text():
    assert size_text(512 * 1000) == "512 KB"
    assert size_text(38 * MB) == "38 MB"
    assert size_text(int(2.5 * GB)) == "2.5 GB"


def test_summary_line():
    text = clip_summary(clip(1080, 1920, 29.97, 45, 38 * MB))
    assert text == "1080x1920 (9:16), 29.97 fps, 45 s, 38 MB"
    assert "MB" not in clip_summary(clip(size=None))


# ---------- the picture shape ----------

def test_the_right_shape_is_ok():
    result = check_shape(clip(1080, 1920), place("ig_reels"))
    assert result.status == OK and "9:16" in result.message


def test_a_slightly_off_shape_still_counts_as_the_right_one():
    assert check_shape(clip(1080, 1900), place("tiktok")).status == OK


def test_landscape_video_on_a_vertical_place_is_a_poor_fit_with_numbers():
    result = check_shape(clip(1920, 1080), place("ig_reels"))
    assert result.status == FAIL
    assert "fills 32 % of the screen" in result.message
    assert "crops away 68 %" in result.message
    assert "9:16" in result.message and "16:9" in result.message


def test_vertical_video_on_youtube_is_a_poor_fit():
    result = check_shape(clip(1080, 1920), place("youtube"))
    assert result.status == FAIL and "32 %" in result.message


def test_square_video_on_reels_works_with_a_catch():
    result = check_shape(clip(1080, 1080), place("ig_reels"))
    assert result.status == WARN and "56 %" in result.message


def test_a_place_that_accepts_several_shapes_takes_the_nearest():
    feed = place("ig_feed")
    for width, height in [(1080, 1350), (1080, 1080), (1920, 1008)]:
        assert check_shape(clip(width, height), feed).status == OK, (width, height)


def test_the_nearest_shape_is_named_in_the_message():
    result = check_shape(clip(1080, 1920), place("ig_feed"))
    assert "4:5" in result.message


def test_a_shape_outside_the_accepted_range_says_what_happens():
    result = check_shape(clip(1920, 1080), place("shorts"))
    assert result.status == FAIL
    assert place("shorts").outside in result.message
    result = check_shape(clip(1080, 1920), place("ig_feed"))
    assert result.status == FAIL and place("ig_feed").outside in result.message


def test_thin_bars_are_not_worth_a_warning():
    # 2:1 on a 16:9 place fills 89 % of the screen
    result = check_shape(clip(2000, 1000), place("youtube"))
    assert result.status == OK and "thin bars" in result.message


def test_four_by_three_on_youtube_warns():
    assert check_shape(clip(1440, 1080), place("youtube")).status == WARN


# ---------- resolution ----------

def test_resolution_uses_the_short_side():
    assert check_resolution(clip(1080, 1920), place("ig_reels")).status == OK
    assert check_resolution(clip(1920, 1080), place("youtube")).status == OK


def test_small_but_allowed_resolution_warns():
    result = check_resolution(clip(720, 1280), place("ig_reels"))
    assert result.status == WARN and "1080" in result.message


def test_below_the_minimum_warns_but_never_fails():
    result = check_resolution(clip(480, 854), place("ig_reels"))
    assert result.status == WARN and "720" in result.message and "soft" in result.message
    assert check_resolution(clip(1, 2), place("ig_reels")).status == WARN


def test_a_vertical_1080_tall_export_is_a_caveat_not_a_failure():
    """608x1080, the 9:16 picture cut out of a 4K file, posts fine on Reels."""
    report = check_placement(clip(608, 1080), place("ig_reels"))
    assert report.verdict == WARN
    assert [c.name for c in report.problems] == ["Resolution"]


def test_a_place_without_a_minimum_only_warns():
    assert check_resolution(clip(360, 640), place("shorts")).status == WARN


# ---------- length ----------

@pytest.mark.parametrize("seconds, status", [
    (2, FAIL), (3, OK), (180, OK), (181, WARN), (900, WARN), (901, FAIL),
])
def test_reels_length_edges(seconds, status):
    assert check_length(clip(duration=seconds), place("ig_reels")).status == status


def test_shorts_length_has_a_soft_and_a_hard_limit():
    shorts = place("shorts")
    assert check_length(clip(duration=60), shorts).status == OK
    assert check_length(clip(duration=61), shorts).status == WARN
    assert check_length(clip(duration=181), shorts).status == FAIL


def test_length_message_names_the_limit():
    result = check_length(clip(duration=95), place("ig_stories"))
    assert result.status == FAIL and "1:00" in result.message
    result = check_length(clip(duration=30), place("ig_stories"))
    assert result.status == WARN and "15 s" in result.message


# ---------- frame rate ----------

@pytest.mark.parametrize("fps, status", [(15, WARN), (22.9, WARN), (23, OK),
                                          (29.97, OK), (60, OK), (61, WARN)])
def test_frame_rate_edges(fps, status):
    assert check_frame_rate(clip(fps=fps), place("youtube")).status == status


# ---------- file size and type ----------

def test_file_size_levels():
    tiktok = place("tiktok")
    assert check_file_size(clip(size=100 * MB), tiktok).status == OK
    assert check_file_size(clip(size=300 * MB), tiktok).status == WARN
    assert check_file_size(clip(size=11 * GB), tiktok).status == FAIL


def test_unknown_file_size_is_not_checked():
    assert check_file_size(clip(size=None), place("tiktok")) is None


def test_file_type_checks_only_places_that_care():
    assert check_file_type(clip(ext="mkv"), place("ig_reels")).status == WARN
    assert check_file_type(clip(ext="mov"), place("ig_reels")).status == OK
    assert check_file_type(clip(ext="mkv"), place("youtube")) is None
    assert check_file_type(clip(ext=""), place("ig_reels")) is None


# ---------- the verdict and the list ----------

def test_the_verdict_is_the_worst_check():
    great = check_placement(clip(), place("ig_reels"))
    assert great.verdict == OK and great.verdict_text == "Great fit" and not great.problems
    caveat = check_placement(clip(duration=200), place("ig_reels"))
    assert caveat.verdict == WARN and caveat.verdict_text == "Works, with caveats"
    poor = check_placement(clip(1920, 1080, duration=200), place("ig_reels"))
    assert poor.verdict == FAIL and poor.verdict_text == "Poor fit"


def test_problems_are_listed_worst_first():
    report = check_placement(clip(1920, 1080, duration=200, ext="mkv"), place("ig_reels"))
    statuses = [c.status for c in report.problems]
    assert statuses[0] == FAIL and statuses == sorted(statuses, key=lambda s: s != FAIL)


def test_unknown_checks_are_left_out():
    report = check_placement(clip(size=None, ext=""), place("ig_reels"))
    assert [c.name for c in report.checks] == ["Picture shape", "Resolution",
                                               "Length", "Frame rate"]


def test_places_come_best_fit_first():
    reports = check_places(clip(1080, 1920))
    verdicts = [r.verdict for r in reports]
    assert verdicts == sorted(verdicts, key=lambda v: {OK: 0, WARN: 1, FAIL: 2}[v])
    assert reports[0].verdict == OK and reports[-1].verdict == FAIL
    assert len(reports) == len(PLACEMENTS)


def test_a_clean_report_beats_one_with_a_caveat():
    # both fit Reels' shape; the 200 s one has a length caveat
    ok = check_placement(clip(), place("ig_reels"))
    caveat = check_placement(clip(duration=200), place("ig_reels"))
    assert ok.score < caveat.score


def test_choosing_places_ignores_unknown_keys():
    reports = check_places(clip(), ["tiktok", "nowhere", "youtube"])
    assert sorted(r.placement.key for r in reports) == ["tiktok", "youtube"]
    assert check_places(clip(), []) == []


def test_ties_keep_the_order_of_the_table():
    reports = check_places(clip(1080, 1920), ["ig_reels", "tiktok", "shorts"])
    assert [r.placement.key for r in reports] == ["shorts", "ig_reels", "tiktok"]


# ---------- the size a viewer sees ----------

def test_rotation_swaps_width_and_height():
    assert display_size(1920, 1080, rotation=90) == (1080, 1920)
    assert display_size(1920, 1080, rotation=-90) == (1080, 1920)
    assert display_size(1920, 1080, rotation=270) == (1080, 1920)
    assert display_size(1920, 1080, rotation=180) == (1920, 1080)
    assert display_size(1920, 1080) == (1920, 1080)


def test_pixel_shape_widens_the_picture():
    assert display_size(1440, 1080, sar=4 / 3) == (1920, 1080)
    assert display_size(1440, 1080, sar=0) == (1440, 1080)


@needs_ffmpeg
def test_geometry_is_read_from_a_real_file(tmp_path):
    plain = tmp_path / "plain.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=30:duration=1", "-pix_fmt", "yuv420p",
                    str(plain)], check=True)
    assert read_display_geometry(plain) == (320, 180)

    turned = tmp_path / "turned.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i",
                    str(plain), "-c", "copy", str(turned)], check=True)
    assert read_display_geometry(turned) == (180, 320)

    wide = tmp_path / "wide.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(plain), "-vf", "setsar=4/3",
                    "-pix_fmt", "yuv420p", str(wide)], check=True)
    assert read_display_geometry(wide) == (427, 180)


@needs_ffmpeg
def test_a_phone_style_vertical_video_is_judged_as_vertical(tmp_path):
    plain = tmp_path / "plain.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=1280x720:rate=30:duration=1", "-pix_fmt", "yuv420p",
                    str(plain)], check=True)
    turned = tmp_path / "turned.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i",
                    str(plain), "-c", "copy", str(turned)], check=True)
    info = SimpleNamespace(width=1280, height=720, fps=30.0, duration=1.0)
    facts = clip_from(turned, info)
    assert (facts.width, facts.height) == (720, 1280)
    assert facts.extension == "mp4" and facts.size == turned.stat().st_size


def test_unreadable_geometry_falls_back_to_the_probe(tmp_path):
    missing = tmp_path / "gone.mov"
    assert read_display_geometry(missing) is None
    info = SimpleNamespace(width=640, height=360, fps=24.0, duration=5.0)
    facts = clip_from(missing, info)
    assert (facts.width, facts.height, facts.size, facts.extension) == (640, 360, None, "mov")
    assert facts.fps == 24.0 and facts.duration == 5.0


# ---------- black bars baked into the picture ----------

import numpy as np

import blackboard.platforms as platforms_module
from blackboard.decode import DecodeError, Frame
from blackboard.platforms import (
    BARS_MATTER, Bars, check_bars, find_bars, sample_frame_numbers, with_bars,
)


def pillarbox(width=1920, height=1080, picture=608, value=120, noise=True):
    """A frame with a vertical picture in the middle and black on both sides."""
    rng = np.random.default_rng(1)
    image = np.zeros((height, width, 3), np.uint8)
    left = (width - picture) // 2
    inner = np.full((height, picture, 3), value, np.uint8)
    if noise:
        inner = np.clip(inner + rng.integers(-20, 20, inner.shape), 30, 255).astype(np.uint8)
    image[:, left:left + picture] = inner
    return image


def letterbox(width=1920, height=1080, picture=800, value=120):
    return np.transpose(pillarbox(height, width, picture, value), (1, 0, 2)).copy()


def serve(monkeypatch, frames):
    """Make find_bars read these frames, one per call, instead of a file."""
    queue = list(frames)

    def fake(path, number):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return Frame(index=number, time=0.0, image=item)

    monkeypatch.setattr(platforms_module, "read_frame", fake)


def test_sample_frames_are_spread_across_the_video():
    numbers = sample_frame_numbers(900, 9)
    assert numbers == [50, 150, 250, 350, 450, 550, 650, 750, 850]
    assert sample_frame_numbers(3, 9) == [0, 1, 2]
    assert sample_frame_numbers(1, 9) == [0]
    assert sample_frame_numbers(0, 9) == []
    assert all(0 <= n < 900 for n in sample_frame_numbers(900, 9))


def test_bars_add_up_and_matter_only_when_big_enough():
    assert Bars(0, 0, 0.2, 0.2).horizontal == pytest.approx(0.4)
    assert Bars(0.1, 0.05, 0, 0).vertical == pytest.approx(0.15)
    assert Bars(0, 0, 0.2, 0.2).matter and Bars(0.1, 0.1, 0, 0).matter
    assert not Bars(0, 0, 0.01, 0.01).matter and not Bars(0, 0, 0, 0).matter
    assert Bars(0, 0, BARS_MATTER / 2, BARS_MATTER / 2).matter       # exactly at the edge


def test_the_picture_size_leaves_the_bars_out():
    plain = Clip(1920, 1080, 30, 10)
    assert (plain.picture_width, plain.picture_height) == (1920, 1080) and not plain.has_bars
    barred = Clip(1920, 1080, 30, 10, bars=Bars(0, 0, 0.342, 0.342))
    assert (barred.picture_width, barred.picture_height) == (607, 1080) and barred.has_bars
    wide = Clip(1080, 1920, 30, 10, bars=Bars(0.2, 0.2, 0, 0))
    assert (wide.picture_width, wide.picture_height) == (1080, 1152)
    thin = Clip(1920, 1080, 30, 10, bars=Bars(0, 0, 0.005, 0.005))
    assert (thin.picture_width, thin.picture_height) == (1920, 1080) and not thin.has_bars


def test_finding_pillarbox_bars(monkeypatch):
    serve(monkeypatch, [pillarbox()] * 9)
    bars = find_bars("x.mp4", 900)
    assert bars.left == pytest.approx(0.3396, abs=0.01) and bars.right == pytest.approx(0.3396, abs=0.01)
    assert bars.top == 0 and bars.bottom == 0
    assert bars.frame_ratio == pytest.approx(16 / 9)


def test_finding_letterbox_bars(monkeypatch):
    serve(monkeypatch, [letterbox()] * 9)
    bars = find_bars("x.mp4", 900)
    assert bars.top == pytest.approx(0.1296, abs=0.01) and bars.bottom == pytest.approx(0.1296, abs=0.01)
    assert bars.left == 0 and bars.right == 0


def test_a_picture_without_bars_has_none(monkeypatch):
    rng = np.random.default_rng(2)
    full = rng.integers(60, 200, (1080, 1920, 3)).astype(np.uint8)
    serve(monkeypatch, [full] * 9)
    bars = find_bars("x.mp4", 900)
    assert not bars.matter


def test_a_few_dark_scenes_do_not_make_bars(monkeypatch):
    # three frames are almost black with a bright spot, which can look like bars
    dark = np.zeros((1080, 1920, 3), np.uint8)
    dark[400:600, 800:1100] = 200
    full = np.random.default_rng(3).integers(60, 200, (1080, 1920, 3)).astype(np.uint8)
    serve(monkeypatch, [dark, full, full, dark, full, full, dark, full, full])
    assert not find_bars("x.mp4", 900).matter


def test_bars_that_are_in_most_frames_count_even_if_a_few_frames_lack_them(monkeypatch):
    full = np.random.default_rng(3).integers(60, 200, (1080, 1920, 3)).astype(np.uint8)
    serve(monkeypatch, [pillarbox()] * 6 + [full] * 3)
    assert find_bars("x.mp4", 900).matter


def test_unreadable_frames_are_skipped(monkeypatch):
    serve(monkeypatch, [DecodeError("no"), pillarbox(), OSError("no"), pillarbox(),
                        pillarbox(), pillarbox(), DecodeError("no"), pillarbox(), pillarbox()])
    assert find_bars("x.mp4", 900).matter


def test_too_few_readable_frames_gives_no_answer(monkeypatch):
    serve(monkeypatch, [pillarbox(), pillarbox()] + [DecodeError("no")] * 7)
    assert find_bars("x.mp4", 900) is None


def test_the_search_can_be_stopped(monkeypatch):
    serve(monkeypatch, [pillarbox()] * 9)
    assert find_bars("x.mp4", 900, should_stop=lambda: True) is None


def test_only_the_frames_that_exist_are_read(monkeypatch):
    seen = []

    def fake(path, number):
        seen.append(number)
        return Frame(index=number, time=0.0, image=pillarbox())

    monkeypatch.setattr(platforms_module, "read_frame", fake)
    find_bars("x.mp4", 5)
    assert seen == [0, 1, 2, 3, 4]


def test_bars_are_applied_when_the_frame_has_the_clips_shape():
    clip = Clip(3840, 2160, 60, 42)
    bars = Bars(0, 0, 0.34, 0.34, 16 / 9)
    assert with_bars(clip, bars).bars == bars


def test_a_turned_frame_swaps_the_bars():
    clip = Clip(1080, 1920, 30, 10)                       # as displayed, vertical
    bars = Bars(0.1, 0.2, 0.0, 0.0, 1920 / 1080)         # measured in a landscape frame
    turned = with_bars(clip, bars).bars
    assert (turned.left, turned.right, turned.top, turned.bottom) == (0.1, 0.2, 0.0, 0.0)
    assert turned.frame_ratio == pytest.approx(1080 / 1920)


def test_bars_from_a_frame_of_another_shape_are_ignored():
    clip = Clip(1920, 1080, 30, 10)
    assert with_bars(clip, Bars(0, 0, 0.3, 0.3, 1.0)).bars is None
    assert with_bars(clip, None) is clip
    assert with_bars(clip, Bars(0, 0, 0.3, 0.3, 0.0)) is clip


def test_the_judgement_uses_the_picture_inside_the_bars():
    """The reported bug: a vertical video in a 4K landscape file made for Reels."""
    file = Clip(3840, 2160, 60.0, 42.2, 387 * MB, "mp4", bars=Bars(0, 0, 0.342, 0.342, 16 / 9))
    reels = check_placement(file, place("ig_reels"))
    shape = reels.checks[0]
    assert shape.status == OK and "inside the black bars" in shape.message
    assert [c.name for c in reels.checks][:2] == ["Picture shape", "Black bars"]
    assert reels.verdict == WARN                           # only the bars are a problem
    assert [c.name for c in reels.problems] == ["Black bars"]
    youtube = check_placement(file, place("youtube"))
    assert youtube.checks[0].status == FAIL                # a vertical picture on YouTube
    assert "inside the black bars" in youtube.checks[0].message


def test_the_bars_warning_says_how_big_and_what_to_do():
    file = Clip(1920, 1080, 30, 10, bars=Bars(0.1, 0.1, 0.3, 0.3, 16 / 9))
    result = check_bars(file, place("tiktok"))
    assert result.status == WARN
    assert "60 % of the width" in result.message and "20 % of the height" in result.message
    assert "crop them off" in result.message and "TikTok" in result.message
    assert "768x864" in result.message


def test_no_bars_warning_without_bars():
    assert check_bars(Clip(1920, 1080, 30, 10), place("tiktok")) is None
    assert check_bars(Clip(1920, 1080, 30, 10, bars=Bars(0, 0, 0.01, 0.01, 1.78)), place("tiktok")) is None


def test_resolution_is_measured_on_the_picture_not_the_file():
    # the file is 4K, but the picture inside the bars is only 540 px wide
    file = Clip(1920, 1080, 30, 10, bars=Bars(0, 0, 0.36, 0.36, 16 / 9))
    result = check_resolution(file, place("ig_reels"))
    assert result.status == WARN and "538 px" in result.message and "720" in result.message
    # the whole file is 1080 px on its short side, which alone would be fine
    assert check_resolution(Clip(1920, 1080, 30, 10), place("ig_reels")).status == OK


def test_the_summary_mentions_the_picture_inside_the_bars():
    file = Clip(3840, 2160, 60.0, 42.2, 387 * MB, "mp4", bars=Bars(0, 0, 0.342, 0.342, 16 / 9))
    text = clip_summary(file)
    assert text.startswith("3840x2160 (16:9), picture inside black bars 1213x2160 (9:16), 60.00 fps")
    assert "black bars" not in clip_summary(Clip(3840, 2160, 60.0, 42.2))


@needs_ffmpeg
def test_bars_are_found_in_a_real_file(tmp_path):
    path = tmp_path / "pillar.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=0x808080:size=202x360:rate=30:duration=2,noise=alls=25:allf=t",
                    "-vf", "pad=640:360:219:0:black", "-pix_fmt", "yuv420p", str(path)], check=True)
    bars = find_bars(path, 60)
    assert bars.left == pytest.approx(219 / 640, abs=0.02)
    assert bars.right == pytest.approx(219 / 640, abs=0.02)
    assert bars.top == 0 and bars.bottom == 0
    file = with_bars(Clip(640, 360, 30, 2), bars)
    assert abs(file.picture_width - 202) <= 14 and file.picture_height == 360