"""Motion: how much things move in the picture.

Two neighboring frames are compared with optical flow, which finds how far
every part of the picture has moved from one frame to the next. The average
distance, as a share of the picture's width and scaled to one second, is the
motion of that moment. 10% per second means that things cross a tenth of the
picture's width in a second. It does not depend on the frame rate or on the
resolution of the video.

Like the color barcode, a segment is sampled at evenly spaced moments. Each
moment compares one frame with the next one, and gives one stripe of the
motion strip.

Flow can only follow things that keep their look while they move. It cannot
follow a cut, where the picture is replaced, and it cannot follow footage
that changes completely from frame to frame, such as flames, smoke, water
spray or a strobe. To tell them apart, a moment that cannot be followed is
checked against the pair of frames beside it. If those two follow each other
normally, the first pair held a cut, and it is left out. If they cannot be
followed either, the footage itself is chaotic. The moment is then kept with
a rough estimate, the average distance the flow found, and marked as rough.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from blackboard.analyze import black_bar_lines, crop_bars, luma, prepare
from blackboard.barcode import BARCODE_COLUMNS, UPDATE_EVERY, sample_indices
from blackboard.decode import DecodeError, read_frames

# Frames are shrunk to this width before the flow is measured. It makes the
# measurement fast and hides film grain and compression noise.
MOTION_WIDTH = 160

# On the strip, this much motion (% of the width per second) is the brightest
# color. Anything faster is drawn the same.
MOTION_FULL = 50.0

# Flow cannot follow a picture that has changed completely. After moving the
# second frame back along the flow it should look like the first one. If the
# frames differed by at least this many brightness levels (out of 255) and
# the flow removed less than 40% of the difference, the two frames are not
# the same scene seen a moment later, and no motion is reported.
UNRELATED_DIFFERENCE = 10.0
UNRELATED_RATIO = 0.6

# A stripe that is only a rough estimate is drawn this much washed out
ROUGH_WASH = 0.5
ROUGH_GRAY = 110

# The color of a stripe where no motion could be measured (a cut)
UNKNOWN_BGR = (68, 68, 68)

# Colors of the strip, from still to fast: dark purple, blue, teal, green, yellow.
# They are given as (blue, green, red).
_STOPS = np.array([
    (84, 1, 68), (139, 82, 59), (140, 145, 33), (98, 201, 94), (37, 231, 253),
], dtype=float)


def motion_between(first, second, fps):
    """The motion between two neighboring frames, in % of the picture width
    per second, or None if the flow cannot follow the change (a cut, or
    footage that changes completely from frame to frame).

    Black bars are cut off first (the bars of the first frame are used for
    both), so they don't count as picture that does not move.
    """
    if first.shape != second.shape:
        raise ValueError("The two frames must be the same size.")
    value, followed = measure_pair(prepare(first), prepare(second), fps)
    return value if followed else None


def measure_pair(small_first, small_second, fps):
    """Measure two neighboring frames that were already shrunk with prepare().

    Returns (motion, followed). `motion` is the average distance the flow
    found, in % of the picture width per second. `followed` says whether the
    flow explains how the first frame became the second. When it is False,
    the motion is only a rough estimate, or meaningless if it was a cut.
    """
    if small_first.shape != small_second.shape:
        raise ValueError("The two frames must be the same size.")
    if fps <= 0:
        raise ValueError("The frame rate must be above 0.")
    lines = black_bar_lines(luma(small_first))
    picture_first = crop_bars(small_first, lines)
    picture_second = crop_bars(small_second, lines)
    gray_first = _gray_small(picture_first)
    gray_second = _gray_small(picture_second)
    flow = cv2.calcOpticalFlowFarneback(
        gray_first, gray_second, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    followed = not _unrelated(gray_first, gray_second, flow)
    distance = float(np.hypot(flow[..., 0], flow[..., 1]).mean())
    return distance / gray_first.shape[1] * 100 * fps, followed


def _unrelated(gray_first, gray_second, flow):
    """True when the flow does not explain how the first frame became the
    second: they are different scenes, or things moved too far to follow."""
    height, width = gray_first.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    moved_back = cv2.remap(gray_second, grid_x + flow[..., 0], grid_y + flow[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    first = gray_first.astype(np.float32)
    before = float(np.abs(first - gray_second).mean())
    after = float(np.abs(first - moved_back).mean())
    return before >= UNRELATED_DIFFERENCE and after >= UNRELATED_RATIO * before


def _gray_small(picture):
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    if width > MOTION_WIDTH:
        gray = cv2.resize(gray, (MOTION_WIDTH, max(1, round(height * MOTION_WIDTH / width))),
                          interpolation=cv2.INTER_AREA)
    return gray


def motion_starts(first, last, columns=BARCODE_COLUMNS):
    """The first frame of each pair that is compared.

    These are the frames the color barcode samples. Every pair is the frame
    and the one after it, except the last frame of the segment, which is
    paired with the one before it so that both frames are in the segment.
    A segment needs at least two frames.
    """
    if last <= first:
        raise ValueError("Motion needs at least two frames.")
    return [min(start, last - 1) for start in sample_indices(first, last, columns)]


@dataclass(frozen=True)
class MotionData:
    """The motion of a segment, one value per stripe."""
    frames: list           # the first frame of the pair behind each stripe
    values: np.ndarray     # % of width per second; NaN where it was a cut
    rough: np.ndarray = None   # True where the value is only a rough estimate

    def rough_flags(self):
        if self.rough is None:
            return np.zeros(len(self.values), bool)
        return np.asarray(self.rough, bool)


def _pair_results(path, starts, fps, should_stop=None, on_each=None):
    """Measure the pair (start, start + 1) for each number in `starts`.

    Returns {start: (motion, followed)} for the pairs that were read. If
    should_stop() says so, the pairs measured so far are returned. If frames
    can't be read part of the way through, the ones before that are returned,
    and only if there are none the DecodeError is raised.
    """
    wanted = set(starts)
    needed = sorted(wanted | {start + 1 for start in starts})
    results = {}
    previous = None                     # (number, shrunk picture) of the last frame
    try:
        for frame in read_frames(path, needed):
            small = prepare(frame.image)
            if (previous is not None and previous[0] in wanted
                    and previous[0] + 1 == frame.index):
                results[previous[0]] = measure_pair(previous[1], small, fps)
                if on_each is not None:
                    on_each(previous[0], results)
                if should_stop is not None and should_stop():
                    break
            previous = (frame.index, small)
    except DecodeError:
        if not results:
            raise
    return results


def build_motion(path, first, last, fps, columns=BARCODE_COLUMNS,
                 on_update=None, should_stop=None):
    """Measure the motion of the stripes of a segment.

    Returns a MotionData. on_update(values_so_far) is called every few
    stripes, and should_stop() is checked after every stripe, like in
    barcode.build_barcode. If frames can't be read part of the way through,
    the stripes measured so far are given. Only if not even one stripe could
    be measured is the DecodeError raised.

    A stripe whose pair the flow cannot follow is checked against the pair
    beside it (the next one, or the one before it at the end of the segment).
    If that pair is followed, the first held a cut and the stripe is NaN.
    If not, the footage is chaotic, and the stripe keeps the flow's rough
    estimate and is marked in `rough`. When a segment is too short to have a
    neighboring pair, such a stripe counts as a cut.
    """
    starts = sorted(set(motion_starts(first, last, columns)))
    order = {start: position for position, start in enumerate(starts)}
    shown = []

    def progress(start, results):
        followed = results[start][1]
        shown.append(results[start][0] if followed else np.nan)
        if on_update is not None and len(shown) % UPDATE_EVERY == 0:
            on_update(np.array(shown, float))

    results = _pair_results(path, starts, fps, should_stop, on_each=progress)
    done = [start for start in starts if start in results]
    stopped = len(done) < len(starts)

    # Pairs that were not followed: a cut, or chaotic footage?
    doubtful = [start for start in done if not results[start][1]]
    neighbor = {start: (start + 1 if start + 2 <= last else start - 1)
                for start in doubtful if last - first >= 2}
    neighbor_followed = {}
    if neighbor and not stopped:
        try:
            checked = _pair_results(path, sorted(set(neighbor.values())), fps, should_stop)
            neighbor_followed = {start: checked[other][1]
                                 for start, other in neighbor.items() if other in checked}
        except DecodeError:
            pass                        # no answer: they are treated as cuts

    values, rough = [], []
    for start in done:
        motion, followed = results[start]
        if followed:
            values.append(motion)
            rough.append(False)
        elif neighbor_followed.get(start) is False:
            values.append(motion)       # chaotic footage: the rough estimate
            rough.append(True)
        else:
            values.append(np.nan)       # a cut
            rough.append(False)
    return MotionData(frames=done, values=np.array(values, float),
                      rough=np.array(rough, bool))


@dataclass(frozen=True)
class MotionSummary:
    """Numbers that describe the motion of a segment."""
    average: float         # % of width per second
    median: float
    peak: float
    peak_frame: int        # the busiest moment
    calmest: float
    calmest_frame: int
    measured: int          # stripes with a value
    skipped: int           # stripes left out because they fell on a cut
    rough: int = 0         # of the measured ones, how many are rough estimates


def summarize_motion(data):
    """Average, median, busiest and calmest moment of the motion."""
    values = np.asarray(data.values, float)
    known = ~np.isnan(values)
    if not known.any():
        raise ValueError("There is no motion to summarize.")
    good = values[known]
    frames = np.asarray(data.frames)[known]
    return MotionSummary(
        average=float(good.mean()),
        median=float(np.median(good)),
        peak=float(good.max()),
        peak_frame=int(frames[good.argmax()]),
        calmest=float(good.min()),
        calmest_frame=int(frames[good.argmin()]),
        measured=int(known.sum()),
        skipped=int((~known).sum()),
        rough=int(data.rough_flags()[known].sum()),
    )


def motion_colors(values, full=MOTION_FULL, rough=None):
    """The colors of the motion strip: an (n x 3) uint8 array, B G R order.

    0 is dark purple, `full` and above is yellow, with blue, teal and green
    between. Stripes with no value (NaN) are gray. Stripes marked in `rough`
    (an array of True/False) have the same colors, washed out.
    """
    values = np.asarray(values, float).reshape(-1)
    unknown = np.isnan(values)
    share = np.clip(np.nan_to_num(values, nan=0.0) / full, 0.0, 1.0)
    position = share * (len(_STOPS) - 1)
    low = np.minimum(position.astype(int), len(_STOPS) - 2)
    mix = (position - low)[:, None]
    colors = _STOPS[low] * (1 - mix) + _STOPS[low + 1] * mix
    if rough is not None:
        washed = np.asarray(rough, bool).reshape(-1)
        colors[washed] = colors[washed] * (1 - ROUGH_WASH) + ROUGH_GRAY * ROUGH_WASH
    result = np.rint(colors).astype(np.uint8)
    result[unknown] = UNKNOWN_BGR
    return result.reshape(-1, 3)