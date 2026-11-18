"""Tests for cut detection and the shot list in shots.py.

Most tests use tiny made-up frames whose right answer we know. The ones that
need FFmpeg build real videos with cuts at known frames.
"""
import shutil
import subprocess

import numpy as np
import pytest

from blackboard.decode import Frame, read_segment
from blackboard.shots import (
    CUT_RATIO, MIN_CUT_SCORE, MIN_SHOT_FRAMES, NEIGHBOR_FRAMES, SIGNATURE_WIDTH,
    ChangeMeter, Shot, ShotSummary, change_score, change_scores, detect_shots,
    find_cuts, shots_from_cuts, signature, summarize_shots,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)

RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)     # B, G, R order


def solid(bgr, height=36, width=64):
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = bgr
    return image


def sig(bgr, **size):
    return signature(solid(bgr, **size))


def frames_of(colors_and_counts, first=0, grain=0, seed=1):
    """Made-up Frame objects: frames_of([(RED, 10), (BLUE, 10)])."""
    rng = np.random.default_rng(seed)
    frames, index = [], first
    for color, count in colors_and_counts:
        for _ in range(count):
            image = solid(color).astype(int)
            if grain:
                image += rng.integers(-grain, grain + 1, image.shape)
            frames.append(Frame(index=index, time=index / 30,
                                image=np.clip(image, 0, 255).astype(np.uint8)))
            index += 1
    return frames


# ---------- signature ----------

def test_signature_is_a_small_float_picture_with_three_numbers_per_pixel():
    result = signature(solid(RED, height=360, width=640))
    assert result.dtype == np.float32
    assert result.shape == (36, SIGNATURE_WIDTH, 3)


def test_signature_shrinks_a_medium_width_picture_too():
    assert signature(solid(RED, height=100, width=200)).shape == (32, SIGNATURE_WIDTH, 3)


def test_signature_never_enlarges_small_pictures():
    assert signature(solid(RED, height=9, width=16)).shape == (9, 16, 3)


def test_signature_keeps_at_least_one_row():
    assert signature(solid(RED, height=1, width=1000)).shape == (1, SIGNATURE_WIDTH, 3)


def test_signature_rejects_bad_images():
    for bad in (np.zeros((4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8),
                np.zeros((0, 4, 3), np.uint8), [[1, 2, 3]]):
        with pytest.raises(ValueError):
            signature(bad)


def test_red_at_either_end_of_the_color_wheel_is_the_same_color():
    # OpenCV hue 0 and hue 179 are both red
    almost_red = solid((10, 0, 255))     # a little blue: hue near 358 degrees
    pure_red = solid((0, 0, 255))
    assert change_score(signature(almost_red), signature(pure_red)) < 10


# ---------- change_score ----------

def test_identical_pictures_score_zero():
    assert change_score(sig(RED), sig(RED)) == 0


def test_black_next_to_white_scores_100():
    assert change_score(sig((0, 0, 0)), sig((255, 255, 255))) == pytest.approx(100)


def test_the_score_is_the_same_either_way_round():
    a, b = sig((60, 120, 90)), sig((90, 90, 90))
    assert change_score(a, b) == pytest.approx(change_score(b, a))


def test_the_score_stays_between_0_and_100():
    colors = [(0, 0, 0), (255, 255, 255), RED, GREEN, BLUE, (30, 90, 200), (128, 128, 128)]
    for a in colors:
        for b in colors:
            assert 0 <= change_score(sig(a), sig(b)) <= 100


def test_a_tiny_change_scores_low():
    assert change_score(sig((100, 100, 100)), sig((104, 100, 100))) < 6


def test_a_brightness_change_scores_in_proportion():
    small = change_score(sig((90, 90, 90)), sig((100, 100, 100)))
    big = change_score(sig((90, 90, 90)), sig((180, 180, 180)))
    assert small < big


def test_a_change_of_color_counts_even_at_the_same_brightness():
    assert change_score(sig(RED), sig(BLUE)) > 40


def test_a_half_changed_picture_scores_about_half_as_much():
    whole = change_score(sig((0, 0, 0)), sig((255, 255, 255)))
    half_picture = solid((0, 0, 0))
    half_picture[:, 32:] = 255
    half = change_score(signature(solid((0, 0, 0))), signature(half_picture))
    assert half == pytest.approx(whole / 2, abs=2)


def test_pictures_of_different_sizes_can_still_be_compared():
    assert change_score(sig(RED, height=36, width=64),
                        sig(RED, height=18, width=32)) == pytest.approx(0, abs=1)


def test_film_grain_hardly_changes_the_score():
    clean = solid((100, 120, 140), height=360, width=640)
    rng = np.random.default_rng(3)
    grainy = np.clip(clean.astype(int) + rng.integers(-12, 13, clean.shape), 0, 255)
    assert change_score(signature(clean), signature(grainy.astype(np.uint8))) < 4


# ---------- ChangeMeter / change_scores ----------

def test_the_first_frame_scores_zero():
    assert ChangeMeter().add(solid(RED)) == 0


def test_each_frame_is_compared_with_the_one_before():
    meter = ChangeMeter()
    meter.add(solid(RED))
    assert meter.add(solid(RED)) == 0
    assert meter.add(solid(BLUE)) == pytest.approx(change_score(sig(RED), sig(BLUE)))
    assert meter.add(solid(BLUE)) == 0


def test_change_scores_gives_one_score_per_image():
    scores = change_scores([solid(RED), solid(RED), solid(BLUE)])
    assert len(scores) == 3 and scores[0] == 0 and scores[1] == 0 and scores[2] > 40


def test_change_scores_of_nothing_is_empty():
    assert change_scores([]) == []


# ---------- find_cuts ----------

def quiet(count, level=1.0):
    return [level] * count


def test_a_quiet_run_has_no_cuts():
    assert find_cuts(quiet(50)) == []


def test_no_scores_means_no_cuts():
    assert find_cuts([]) == []
    assert find_cuts([0.0]) == []


def test_a_clear_spike_is_a_cut():
    scores = quiet(30)
    scores[12] = 60
    assert find_cuts(scores) == [12]


def test_the_first_position_is_never_a_cut():
    scores = [90.0] + quiet(20)
    assert find_cuts(scores) == []


def test_a_spike_below_the_minimum_is_ignored():
    scores = quiet(30, level=0.1)
    scores[10] = MIN_CUT_SCORE - 0.5
    assert find_cuts(scores) == []


def test_a_spike_exactly_at_the_minimum_counts():
    scores = quiet(30, level=0.1)
    scores[10] = MIN_CUT_SCORE
    assert find_cuts(scores) == [10]


def test_constant_high_activity_is_not_a_cut():
    # A fast camera move: every frame changes a lot, none stands out
    assert find_cuts([30.0] * 40) == []


def test_a_spike_must_stand_out_from_busy_neighbors():
    busy = [25.0] * 30
    busy[15] = 25.0 * CUT_RATIO - 1        # not enough
    assert find_cuts(busy) == []
    busy[15] = 25.0 * CUT_RATIO + 1        # enough
    assert find_cuts(busy) == [15]


def test_two_far_apart_cuts_are_both_found():
    scores = quiet(60)
    scores[15] = 70
    scores[45] = 55
    assert find_cuts(scores) == [15, 45]


def test_a_cut_right_next_to_another_keeps_only_the_stronger():
    scores = quiet(40)
    scores[20] = 50
    scores[21] = 80                        # the flash's second spike is stronger
    assert find_cuts(scores) == [21]
    scores[21], scores[20] = 50, 80
    assert find_cuts(scores) == [20]


def test_cuts_exactly_the_minimum_gap_apart_are_both_kept():
    scores = quiet(40)
    scores[10] = 60
    scores[10 + MIN_SHOT_FRAMES] = 60
    assert find_cuts(scores) == [10, 10 + MIN_SHOT_FRAMES]


def test_cuts_near_the_start_and_end_are_found():
    scores = quiet(30)
    scores[1] = 60
    scores[29] = 60
    assert find_cuts(scores) == [1, 29]


def test_the_minimum_score_can_be_changed():
    scores = quiet(30)
    scores[10] = 20
    assert find_cuts(scores, min_score=25) == []
    assert find_cuts(scores, min_score=15) == [10]


def test_the_ratio_can_be_changed():
    scores = [8.0] * 30
    scores[10] = 20                        # 2.5 times its neighbors
    assert find_cuts(scores, ratio=2.4, min_score=5) == [10]
    assert find_cuts(scores, ratio=2.6, min_score=5) == []


def test_the_neighbor_window_can_be_changed():
    # The frames right next to the spike are busy, the ones farther out are quiet
    scores = quiet(30)
    for position in (8, 9, 11, 12):
        scores[position] = 9.0
    scores[10] = 20
    assert find_cuts(scores, neighbors=2) == []        # sees only the busy ones
    assert find_cuts(scores, neighbors=5) == [10]      # sees the quiet ones too


def test_the_minimum_gap_can_be_changed():
    scores = quiet(30)
    scores[10] = 60
    scores[12] = 70
    assert find_cuts(scores, min_gap=3) == [12]
    assert find_cuts(scores, min_gap=2) == [10, 12]


def test_find_cuts_accepts_a_numpy_array():
    scores = np.array(quiet(30))
    scores[7] = 60
    assert find_cuts(scores) == [7]


def test_the_neighbor_window_is_what_counts():
    # Busy frames far outside the window do not hide a cut
    scores = [30.0] * 10 + quiet(30) + [30.0] * 10
    scores[25] = 80
    assert NEIGHBOR_FRAMES < 14
    assert find_cuts(scores) == [25]


# ---------- shots_from_cuts ----------

def test_no_cuts_is_one_shot_covering_everything():
    shots = shots_from_cuts([], 0, 99, fps=25)
    assert shots == [Shot(number=1, start=0, end=99, frames=100, seconds=4.0)]


def test_two_cuts_make_three_shots():
    shots = shots_from_cuts([30, 90], 0, 119, fps=30)
    assert [(s.number, s.start, s.end, s.frames) for s in shots] == [
        (1, 0, 29, 30), (2, 30, 89, 60), (3, 90, 119, 30)]
    assert [s.seconds for s in shots] == pytest.approx([1.0, 2.0, 1.0])


def test_shots_leave_no_gaps_and_add_up_to_the_segment():
    shots = shots_from_cuts([5, 17, 18, 40], 0, 59, fps=24)
    assert sum(s.frames for s in shots) == 60
    for before, after in zip(shots, shots[1:]):
        assert after.start == before.end + 1


def test_a_segment_that_does_not_start_at_zero():
    shots = shots_from_cuts([150], 100, 199, fps=50)
    assert (shots[0].start, shots[0].end) == (100, 149)
    assert (shots[1].start, shots[1].end) == (150, 199)
    assert shots[0].seconds == pytest.approx(1.0)


def test_cuts_can_come_in_any_order_and_repeat():
    assert shots_from_cuts([90, 30, 30], 0, 119, 30) == shots_from_cuts([30, 90], 0, 119, 30)


def test_a_one_frame_segment_is_one_shot():
    shots = shots_from_cuts([], 7, 7, fps=30)
    assert len(shots) == 1 and shots[0].frames == 1


def test_a_shot_can_be_a_single_frame():
    shots = shots_from_cuts([10, 11], 0, 20, fps=30)
    assert shots[1].frames == 1 and shots[1].start == shots[1].end == 10


@pytest.mark.parametrize("cuts", [[0], [-3], [100], [250]])
def test_cuts_outside_the_segment_are_rejected(cuts):
    with pytest.raises(ValueError):
        shots_from_cuts(cuts, 0, 99, fps=30)


def test_a_cut_at_the_first_frame_of_an_offset_segment_is_rejected():
    with pytest.raises(ValueError):
        shots_from_cuts([100], 100, 199, fps=30)


def test_the_last_frame_can_start_a_shot():
    shots = shots_from_cuts([99], 0, 99, fps=30)
    assert shots[-1].frames == 1


@pytest.mark.parametrize("fps", [0, -24])
def test_bad_frame_rates_are_rejected(fps):
    with pytest.raises(ValueError):
        shots_from_cuts([], 0, 10, fps)


def test_a_segment_that_ends_before_it_starts_is_rejected():
    with pytest.raises(ValueError):
        shots_from_cuts([], 10, 9, 30)


# ---------- summarize_shots ----------

def make_shots(seconds_list, fps=10):
    cuts, frame = [], 0
    for seconds in seconds_list[:-1]:
        frame += round(seconds * fps)
        cuts.append(frame)
    total = round(sum(seconds_list) * fps)
    return shots_from_cuts(cuts, 0, total - 1, fps)


def test_summary_of_shot_lengths():
    summary = summarize_shots(make_shots([2.0, 4.0, 6.0, 8.0]))
    assert summary.count == 4
    assert summary.average == pytest.approx(5.0)
    assert summary.median == pytest.approx(5.0)
    assert summary.shortest == pytest.approx(2.0)
    assert summary.longest == pytest.approx(8.0)


def test_median_is_not_pulled_by_one_long_shot():
    summary = summarize_shots(make_shots([2.0, 2.0, 2.0, 30.0]))
    assert summary.median == pytest.approx(2.0)
    assert summary.average == pytest.approx(9.0)


def test_cutting_pace_is_cuts_per_minute():
    # 3 shots in 30 seconds = 2 cuts in half a minute = 4 cuts a minute
    summary = summarize_shots(make_shots([10.0, 10.0, 10.0]))
    assert summary.cuts_per_minute == pytest.approx(4.0)


def test_one_shot_has_no_cuts():
    summary = summarize_shots(make_shots([12.0]))
    assert summary.count == 1 and summary.cuts_per_minute == 0
    assert summary.average == summary.median == summary.shortest == summary.longest


def test_summary_of_no_shots_is_rejected():
    with pytest.raises(ValueError):
        summarize_shots([])


def test_summary_is_a_plain_record():
    assert isinstance(summarize_shots(make_shots([1.0, 2.0])), ShotSummary)


# ---------- detect_shots ----------

def test_detect_shots_finds_the_cuts_between_colors():
    shots, scores = detect_shots(frames_of([(RED, 10), (BLUE, 12), (GREEN, 8)]), fps=30)
    assert [(s.start, s.end) for s in shots] == [(0, 9), (10, 21), (22, 29)]
    assert len(scores) == 30


def test_the_scores_peak_at_the_cuts():
    _, scores = detect_shots(frames_of([(RED, 10), (BLUE, 10)]), fps=30)
    assert scores[10] > 40
    assert max(s for i, s in enumerate(scores) if i != 10) == 0


def test_detect_shots_uses_the_real_frame_numbers():
    shots, _ = detect_shots(frames_of([(RED, 10), (BLUE, 10)], first=500), fps=25)
    assert [(s.start, s.end) for s in shots] == [(500, 509), (510, 519)]
    assert shots[0].seconds == pytest.approx(10 / 25)


def test_detect_shots_with_grain_still_finds_only_the_real_cut():
    shots, _ = detect_shots(
        frames_of([(RED, 15), (BLUE, 15)], grain=15), fps=30)
    assert [(s.start, s.end) for s in shots] == [(0, 14), (15, 29)]


def test_a_slow_fade_is_not_a_cut():
    frames = []
    for i in range(60):
        level = i * 4
        frames.append(Frame(i, i / 30, solid((level, level, level))))
    shots, _ = detect_shots(frames, fps=30)
    assert len(shots) == 1


def test_a_one_frame_flash_is_not_two_shots_but_one_cut_pair_is_merged():
    # Dark, one bright frame, dark again: two spikes next to each other
    shots, _ = detect_shots(frames_of([((30, 30, 30), 20), ((250, 250, 250), 1),
                                       ((30, 30, 30), 20)]), fps=30)
    assert len(shots) <= 2


def test_detect_shots_with_one_frame_is_one_shot():
    shots, scores = detect_shots(frames_of([(RED, 1)]), fps=30)
    assert len(shots) == 1 and scores == [0.0]


def test_detect_shots_with_no_frames_is_rejected():
    with pytest.raises(ValueError):
        detect_shots([], fps=30)


def test_detect_shots_accepts_a_generator():
    shots, _ = detect_shots((f for f in frames_of([(RED, 8), (BLUE, 8)])), fps=30)
    assert len(shots) == 2


# ---------- real videos ----------

def make_cut_clip(path, colors, seconds=1, noise=0):
    """A video made of solid-color pieces, one per color, back to back."""
    inputs = []
    for color in colors:
        inputs += ["-f", "lavfi", "-i", f"color=c={color}:s=320x240:d={seconds}:r=30"]
    graph = f"concat=n={len(colors)}:v=1:a=0"
    if noise:
        graph += f",noise=alls={noise}:allf=t"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", graph,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_decoded_video_cuts_are_found_at_the_right_frames(tmp_path):
    clip = make_cut_clip(tmp_path / "cuts.mp4", ["red", "blue", "lime"])
    shots, _ = detect_shots(read_segment(clip, 0, 89), fps=30)
    assert [(s.start, s.end) for s in shots] == [(0, 29), (30, 59), (60, 89)]
    assert all(s.seconds == pytest.approx(1.0) for s in shots)


@needs_ffmpeg
def test_decoded_noisy_video_still_has_only_the_real_cuts(tmp_path):
    clip = make_cut_clip(tmp_path / "noisy.mp4", ["red", "blue"], noise=25)
    shots, _ = detect_shots(read_segment(clip, 0, 59), fps=30)
    assert [(s.start, s.end) for s in shots] == [(0, 29), (30, 59)]


@needs_ffmpeg
def test_a_segment_in_the_middle_of_a_video(tmp_path):
    clip = make_cut_clip(tmp_path / "cuts.mp4", ["red", "blue", "lime"])
    shots, _ = detect_shots(read_segment(clip, 20, 70), fps=30)
    assert [(s.start, s.end) for s in shots] == [(20, 29), (30, 59), (60, 70)]


@needs_ffmpeg
def test_a_moving_picture_without_cuts_has_one_shot(tmp_path):
    clip = tmp_path / "moving.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x240:rate=30:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    shots, _ = detect_shots(read_segment(clip, 0, 89), fps=30)
    assert len(shots) == 1