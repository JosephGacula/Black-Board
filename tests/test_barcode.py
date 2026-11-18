"""Tests for the color barcode: the average color, reading chosen frames,
and measuring the stripes of a segment of a video."""
import shutil
import subprocess

import numpy as np
import pytest

from blackboard.analyze import average_color
from blackboard.barcode import (
    BARCODE_COLUMNS, UPDATE_EVERY, barcode_columns, build_barcode, sample_indices,
    brightness_colors, segment_length,
)
from blackboard.decode import SEEK_GAP, DecodeError, read_frame, read_frames, read_segment

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


def solid(blue, green, red, height=8, width=8):
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = (blue, green, red)
    return image


def make_clip(path, inputs, filter_complex=None):
    command = ["ffmpeg", "-y", "-v", "error", *inputs]
    if filter_complex:
        command += ["-filter_complex", filter_complex]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", str(path)]
    subprocess.run(command, check=True)
    return path


def lavfi(source):
    return ["-f", "lavfi", "-i", source]


@pytest.fixture(scope="module")
def moving_clip(tmp_path_factory):
    """400 frames that all look different."""
    path = tmp_path_factory.mktemp("clips") / "moving.mp4"
    return make_clip(path, lavfi("testsrc2=size=160x120:rate=30:duration=13.4"))


@pytest.fixture(scope="module")
def red_then_blue(tmp_path_factory):
    """60 red frames followed by 60 blue frames."""
    path = tmp_path_factory.mktemp("clips") / "red_blue.mp4"
    return make_clip(
        path,
        [*lavfi("color=c=red:s=160x120:d=2:r=30"), *lavfi("color=c=blue:s=160x120:d=2:r=30")],
        "[0:v][1:v]concat=n=2:v=1:a=0",
    )


# ---------- average_color ----------

def test_a_solid_picture_has_that_color():
    assert average_color(solid(10, 200, 30)) == (10, 200, 30)


def test_the_result_is_three_whole_numbers():
    color = average_color(solid(10, 200, 30))
    assert len(color) == 3 and all(isinstance(value, int) for value in color)


def test_two_halves_average_out():
    picture = np.concatenate([solid(0, 0, 200, height=10), solid(0, 0, 100, height=10)])
    assert average_color(picture) == (0, 0, 150)


def test_black_bars_do_not_darken_the_color():
    bars = np.zeros((20, 40, 3), np.uint8)
    picture = np.concatenate([bars, solid(40, 120, 200, height=60, width=40), bars])
    assert average_color(picture) == (40, 120, 200)


def test_pillarbox_bars_are_left_out_too():
    bars = np.zeros((40, 10, 3), np.uint8)
    picture = np.concatenate([bars, solid(40, 120, 200, height=40, width=60), bars], axis=1)
    assert average_color(picture) == (40, 120, 200)


def test_a_black_picture_is_black():
    assert average_color(solid(0, 0, 0)) == (0, 0, 0)


def test_a_big_picture_gives_the_same_color_as_a_small_one():
    big = solid(30, 90, 220, height=1080, width=1920)
    assert average_color(big) == (30, 90, 220)


def test_average_color_rejects_bad_images():
    with pytest.raises(ValueError):
        average_color(np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError):
        average_color(np.zeros((0, 4, 3), np.uint8))


# ---------- how many stripes, and which frames ----------

def test_a_long_segment_gets_the_full_number_of_stripes():
    assert barcode_columns(0, 7499) == BARCODE_COLUMNS


def test_a_short_segment_gets_one_stripe_per_frame():
    assert barcode_columns(0, 29) == 30
    assert barcode_columns(5, 5) == 1


def test_the_number_of_stripes_can_be_changed():
    assert barcode_columns(0, 999, columns=50) == 50


def test_the_stripe_count_depends_on_the_segment_not_the_video():
    assert barcode_columns(1000, 1019) == 20
    assert barcode_columns(1000, 1019) == barcode_columns(0, 19)


def test_stripe_count_rejects_bad_numbers():
    with pytest.raises(ValueError):
        barcode_columns(10, 9)                  # ends before it starts
    with pytest.raises(ValueError):
        barcode_columns(-1, 10)
    with pytest.raises(ValueError):
        barcode_columns(0, 99, columns=0)


def test_segment_length_counts_both_ends():
    assert segment_length(0, 0) == 1
    assert segment_length(10, 19) == 10


def test_samples_are_the_middles_of_equal_slices():
    assert sample_indices(0, 99, columns=4) == [12, 37, 62, 87]


def test_samples_stay_inside_the_segment():
    assert sample_indices(100, 199, columns=4) == [112, 137, 162, 187]


def test_every_frame_is_sampled_when_there_are_few():
    assert sample_indices(0, 4) == [0, 1, 2, 3, 4]
    assert sample_indices(7, 9) == [7, 8, 9]
    assert sample_indices(3, 3) == [3]


@pytest.mark.parametrize("first,last", [
    (0, 0), (0, 1), (0, 2), (0, 6), (0, 49), (0, 398), (0, 399), (0, 400),
    (0, 798), (0, 7510), (0, 99999), (250, 250), (250, 900), (12345, 54321),
])
def test_samples_are_in_range_and_strictly_increasing(first, last):
    indices = sample_indices(first, last)
    assert len(indices) == barcode_columns(first, last)
    assert indices[0] >= first and indices[-1] <= last
    assert all(later > earlier for earlier, later in zip(indices, indices[1:]))


def test_samples_cover_the_whole_segment_evenly():
    indices = sample_indices(0, 3999, columns=400)
    gaps = {later - earlier for earlier, later in zip(indices, indices[1:])}
    assert gaps == {10}
    assert indices[0] == 5 and indices[-1] == 3995


def test_samples_of_a_segment_are_shifted_by_its_start():
    plain = sample_indices(0, 999, columns=40)
    shifted = sample_indices(500, 1499, columns=40)
    assert shifted == [index + 500 for index in plain]


# ---------- reading chosen frames ----------

@needs_ffmpeg
def test_read_frames_gives_the_frames_asked_for(moving_clip):
    frames = list(read_frames(moving_clip, [0, 5, 6, 399]))
    assert [frame.index for frame in frames] == [0, 5, 6, 399]
    assert frames[1].image.shape == (120, 160, 3)


@needs_ffmpeg
def test_read_frames_has_the_right_times(moving_clip):
    frames = list(read_frames(moving_clip, [30, 60]))
    assert [round(frame.time, 3) for frame in frames] == [1.0, 2.0]


@needs_ffmpeg
def test_frames_read_close_together_match_one_by_one_reads(moving_clip):
    indices = [10, 11, 15, 40, 41, 42]
    for frame in read_frames(moving_clip, indices):
        assert np.array_equal(frame.image, read_frame(moving_clip, frame.index).image)


@needs_ffmpeg
def test_frames_read_far_apart_match_one_by_one_reads(moving_clip):
    # Gaps bigger than SEEK_GAP make it jump instead of reading on
    indices = [3, 3 + SEEK_GAP + 40, 3 + 2 * (SEEK_GAP + 40), 399]
    for frame in read_frames(moving_clip, indices):
        assert np.array_equal(frame.image, read_frame(moving_clip, frame.index).image)


@needs_ffmpeg
def test_read_frames_matches_read_segment(moving_clip):
    sequential = {frame.index: frame.image for frame in read_segment(moving_clip, 20, 80, step=7)}
    for frame in read_frames(moving_clip, sorted(sequential)):
        assert np.array_equal(frame.image, sequential[frame.index])


@needs_ffmpeg
def test_read_frames_works_with_a_generator_of_numbers(moving_clip):
    frames = list(read_frames(moving_clip, (i for i in (1, 2, 3))))
    assert [frame.index for frame in frames] == [1, 2, 3]


@needs_ffmpeg
def test_read_frames_with_no_numbers_gives_nothing(moving_clip):
    assert list(read_frames(moving_clip, [])) == []


@needs_ffmpeg
@pytest.mark.parametrize("indices", [[5, 3], [4, 4], [-1, 3]])
def test_read_frames_rejects_bad_numbers(moving_clip, indices):
    with pytest.raises(DecodeError):
        list(read_frames(moving_clip, indices))


@needs_ffmpeg
def test_a_number_past_the_end_stops_only_when_reached(moving_clip):
    seen = []
    with pytest.raises(DecodeError):
        for frame in read_frames(moving_clip, [10, 20, 5000]):
            seen.append(frame.index)
    assert seen == [10, 20]


def test_read_frames_of_a_missing_file_raises(tmp_path):
    with pytest.raises(DecodeError):
        list(read_frames(tmp_path / "nothing.mp4", [0]))


# ---------- measuring the barcode of a segment ----------

@needs_ffmpeg
def test_the_stripes_follow_the_colors_of_the_video(red_then_blue):
    colors = build_barcode(red_then_blue, 0, 119, columns=10)
    assert colors.shape == (10, 3) and colors.dtype == np.uint8
    for stripe in colors[:5]:
        assert stripe[2] > 200 and stripe[0] < 60              # red
    for stripe in colors[5:]:
        assert stripe[0] > 200 and stripe[2] < 60              # blue


@needs_ffmpeg
def test_a_segment_with_fewer_frames_than_stripes_gets_a_stripe_per_frame(red_then_blue):
    assert len(build_barcode(red_then_blue, 0, 119)) == 120
    assert len(build_barcode(red_then_blue, 10, 19)) == 10


@needs_ffmpeg
def test_only_the_chosen_segment_is_measured(red_then_blue):
    # Frames 0-59 are red, 60-119 are blue
    reds = build_barcode(red_then_blue, 5, 50, columns=10)
    blues = build_barcode(red_then_blue, 70, 110, columns=10)
    assert all(stripe[2] > 200 and stripe[0] < 60 for stripe in reds)
    assert all(stripe[0] > 200 and stripe[2] < 60 for stripe in blues)


@needs_ffmpeg
def test_a_segment_across_the_change_shows_both_colors(red_then_blue):
    colors = build_barcode(red_then_blue, 40, 79, columns=8)
    assert colors[0][2] > 200 and colors[-1][0] > 200
    assert all(colors[i][2] > 200 for i in range(4))
    assert all(colors[i][0] > 200 for i in range(4, 8))


@needs_ffmpeg
def test_a_one_frame_segment_gives_one_stripe(red_then_blue):
    colors = build_barcode(red_then_blue, 100, 100)
    assert colors.shape == (1, 3) and colors[0][0] > 200


@needs_ffmpeg
def test_a_segment_ending_on_the_last_frame_works(red_then_blue):
    colors = build_barcode(red_then_blue, 100, 119)
    assert len(colors) == 20 and colors[-1][0] > 200


@needs_ffmpeg
def test_the_stripes_match_the_frames_measured_one_by_one(moving_clip):
    colors = build_barcode(moving_clip, 100, 199, columns=5)
    expected = [average_color(read_frame(moving_clip, index).image)
                for index in sample_indices(100, 199, columns=5)]
    assert [tuple(int(v) for v in stripe) for stripe in colors] == expected


@needs_ffmpeg
def test_progress_is_reported_every_few_stripes(red_then_blue):
    seen = []
    build_barcode(red_then_blue, 0, 119, columns=25,
                  on_update=lambda found: seen.append(len(found)))
    assert seen == [UPDATE_EVERY, 2 * UPDATE_EVERY]


@needs_ffmpeg
def test_progress_gives_stripes_in_order(red_then_blue):
    seen = []
    final = build_barcode(red_then_blue, 0, 119, columns=20,
                          on_update=lambda found: seen.append(found.copy()))
    for found in seen:
        assert np.array_equal(found, final[:len(found)])


@needs_ffmpeg
def test_it_stops_when_asked(red_then_blue):
    colors = build_barcode(red_then_blue, 0, 119, columns=20, should_stop=lambda: True)
    assert len(colors) == 1


@needs_ffmpeg
def test_it_stops_part_way(red_then_blue):
    calls = []

    def stop_after_three():
        calls.append(1)
        return len(calls) >= 3

    assert len(build_barcode(red_then_blue, 0, 119, columns=20,
                             should_stop=stop_after_three)) == 3


@needs_ffmpeg
def test_a_segment_longer_than_the_video_gives_the_stripes_it_can(red_then_blue):
    # 120 frames really, but the segment goes on to frame 149
    colors = build_barcode(red_then_blue, 0, 149, columns=10)
    assert 0 < len(colors) < 10
    assert colors[0][2] > 200                                  # the first is still red


@needs_ffmpeg
def test_a_segment_wholly_past_the_end_raises(red_then_blue):
    with pytest.raises(DecodeError):
        build_barcode(red_then_blue, 200, 260)


def test_a_backwards_segment_raises(tmp_path):
    with pytest.raises(ValueError):
        build_barcode(tmp_path / "nothing.mp4", 50, 10)


def test_a_file_that_cannot_be_read_raises(tmp_path):
    with pytest.raises(DecodeError):
        build_barcode(tmp_path / "nothing.mp4", 0, 99)


def test_a_text_file_renamed_to_mp4_raises(tmp_path):
    path = tmp_path / "fake.mp4"
    path.write_text("this is not a video")
    with pytest.raises(DecodeError):
        build_barcode(path, 0, 99)


# ---------- the brightness barcode ----------

def test_brightness_stripes_are_gray():
    gray = brightness_colors([(10, 200, 30), (0, 0, 255), (255, 0, 0)])
    assert gray.shape == (3, 3) and gray.dtype == np.uint8
    assert all(len(set(int(v) for v in row)) == 1 for row in gray)


def test_black_is_black_and_white_is_white():
    gray = brightness_colors([(0, 0, 0), (255, 255, 255)])
    assert gray[0][0] == 0 and gray[1][0] == 255


def test_gray_stays_the_same_gray():
    assert brightness_colors([(90, 90, 90)])[0][0] == 90


def test_green_is_brighter_than_red_and_red_brighter_than_blue():
    blue, green, red = (brightness_colors([c])[0][0]
                        for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    assert green > red > blue
    assert (blue, green, red) == (18, 182, 54)          # 0.0722, 0.7152, 0.2126


def test_brightness_keeps_the_order_and_count_of_stripes():
    gray = brightness_colors([(0, 0, 0), (128, 128, 128), (255, 255, 255)])
    assert [int(row[0]) for row in gray] == [0, 128, 255]


def test_no_stripes_gives_an_empty_result():
    assert brightness_colors(np.zeros((0, 3), np.uint8)).shape == (0, 3)


def test_brightness_matches_the_frame_brightness_measurement():
    from blackboard.analyze import analyze_frame
    picture = solid(40, 120, 200, height=60, width=60)
    expected = analyze_frame(picture).brightness * 255 / 100
    assert abs(int(brightness_colors([average_color(picture)])[0][0]) - expected) <= 1.5