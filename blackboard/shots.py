"""Cut detection: finding where one shot ends and the next one begins.

A cut is a sudden change of picture between two neighboring frames. For
every frame we measure how much the picture changed since the frame before
it (the "change score"). A cut is a frame whose score is both high and much
higher than the frames around it. Frames in between are not touched, so a
fast camera move that changes every frame a little is not mistaken for a
cut. Slow dissolves and fades change the picture gradually, so they are not
found. Only hard cuts are.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from blackboard.decode import DecodeError, read_segment

# Frames are shrunk to this width before they are compared. It makes the
# comparison fast, and it hides film grain and compression noise.
SIGNATURE_WIDTH = 64

# A frame needs at least this change score (0 to 100) to count as a cut.
MIN_CUT_SCORE = 12.0

# And its score must be at least this many times the typical score of the
# frames around it. This is what ignores fast motion and flashing lights.
CUT_RATIO = 3.0

# How many frames on each side are looked at to judge what is "typical"
NEIGHBOR_FRAMES = 5

# Two cuts closer together than this many frames count as one, and the
# stronger one is kept. Flashes and dissolves can make a double spike.
MIN_SHOT_FRAMES = 3

# The progress callback is called after every this many frames
PROGRESS_EVERY = 20


def signature(image):
    """A small, fixed-size summary of one frame, to compare with another.

    The frame is shrunk and turned into hue, saturation and value (HSV).
    Hue is turned into two numbers, cos and sin of its angle, so that red at
    the start and end of the color wheel counts as the same color. Everything
    is kept as numbers from 0 to 1.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a height x width x 3 image.")
    if image.size == 0:
        raise ValueError("The image has no pixels.")
    height, width = image.shape[:2]
    if width > SIGNATURE_WIDTH:
        new_height = max(1, round(height * SIGNATURE_WIDTH / width))
        image = cv2.resize(image, (SIGNATURE_WIDTH, new_height),
                           interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    angle = hsv[..., 0] * (2 * np.pi / 180)          # OpenCV hue is 0..179
    saturation = hsv[..., 1] / 255
    value = hsv[..., 2] / 255
    # Gray pixels have no real hue, so their hue is scaled by saturation
    return np.stack([np.cos(angle) * saturation, np.sin(angle) * saturation,
                     value], axis=-1)


def change_score(first, second):
    """How much the picture changed between two signatures, from 0 to 100.

    0 means identical. Each pixel is compared with the pixel in the same
    place, and the average distance is scaled so that all black next to all
    white scores 100. Changes of color count too, and the score stops at 100.
    """
    if first.shape != second.shape:
        # Frames of different sizes (it should not happen inside one video)
        # are compared as if they were the same size.
        second = cv2.resize(second, (first.shape[1], first.shape[0]),
                            interpolation=cv2.INTER_AREA)
    distance = np.sqrt(((first - second) ** 2).sum(axis=-1))
    return float(min(distance.mean() * 100, 100.0))


class ChangeMeter:
    """Feeds frames in one by one and reports how much each one changed."""

    def __init__(self):
        self._last = None

    def add(self, image):
        """The change score of this frame against the previous one.

        The first frame has nothing to compare with, so its score is 0.
        """
        current = signature(image)
        score = 0.0 if self._last is None else change_score(self._last, current)
        self._last = current
        return score


def change_scores(images):
    """The change score of every image in a sequence (the first is 0)."""
    meter = ChangeMeter()
    return [meter.add(image) for image in images]


def find_cuts(scores, min_score=MIN_CUT_SCORE, ratio=CUT_RATIO,
              neighbors=NEIGHBOR_FRAMES, min_gap=MIN_SHOT_FRAMES):
    """The positions in `scores` where a new shot starts.

    A position is a cut when its score is at least `min_score` and at least
    `ratio` times the median score of the `neighbors` positions on each side.
    When cuts are closer than `min_gap` positions the strongest one is kept.
    Position 0 is never a cut, because it is where the first shot starts.
    """
    scores = np.asarray(scores, dtype=float)
    candidates = []
    for position in range(1, len(scores)):
        score = scores[position]
        if score < min_score:
            continue
        around = np.concatenate([scores[max(0, position - neighbors):position],
                                 scores[position + 1:position + 1 + neighbors]])
        typical = float(np.median(around)) if len(around) else 0.0
        if score >= ratio * typical:
            candidates.append(position)

    # Of cuts that are too close together, keep the strongest
    kept = []
    for position in candidates:
        if kept and position - kept[-1] < min_gap:
            if scores[position] > scores[kept[-1]]:
                kept[-1] = position
        else:
            kept.append(position)
    return kept


@dataclass(frozen=True)
class Shot:
    """One shot: a run of frames between two cuts."""
    number: int          # 1 for the first shot, and so on
    start: int           # first frame (frame number in the video)
    end: int             # last frame, included
    frames: int          # how many frames long
    seconds: float       # how long it lasts


def shots_from_cuts(cut_frames, first_frame, last_frame, fps):
    """Turn cut frame numbers into a list of shots.

    `cut_frames` are the frame numbers where new shots start. They must lie
    inside the segment, after its first frame. The shots then cover every
    frame from `first_frame` to `last_frame` with no gaps.
    """
    if fps <= 0:
        raise ValueError("The frame rate must be above 0.")
    if last_frame < first_frame:
        raise ValueError("The segment must not end before it starts.")
    cuts = sorted(set(cut_frames))
    for cut in cuts:
        if not first_frame < cut <= last_frame:
            raise ValueError(f"Cut at frame {cut} is outside the segment.")
    starts = [first_frame] + cuts
    ends = [cut - 1 for cut in cuts] + [last_frame]
    return [Shot(number=i + 1, start=s, end=e, frames=e - s + 1,
                 seconds=(e - s + 1) / fps)
            for i, (s, e) in enumerate(zip(starts, ends))]


@dataclass(frozen=True)
class ShotSummary:
    """Numbers that describe all the shots together."""
    count: int
    average: float       # seconds
    median: float        # seconds
    shortest: float      # seconds
    longest: float       # seconds
    cuts_per_minute: float


def summarize_shots(shots):
    """Average, median, shortest and longest shot, and cutting pace."""
    if not shots:
        raise ValueError("There are no shots to summarize.")
    lengths = np.array([shot.seconds for shot in shots])
    total = float(lengths.sum())
    return ShotSummary(
        count=len(shots),
        average=float(lengths.mean()),
        median=float(np.median(lengths)),
        shortest=float(lengths.min()),
        longest=float(lengths.max()),
        cuts_per_minute=(len(shots) - 1) / total * 60 if total > 0 else 0.0,
    )


def detect_shots(frames, fps):
    """Find the shots in a run of decoded frames.

    `frames` is any iterable of decode.Frame objects (every frame, in order).
    Returns (shots, scores), where scores holds one change score per frame.
    """
    scores = []
    numbers = []
    meter = ChangeMeter()
    for frame in frames:
        scores.append(meter.add(frame.image))
        numbers.append(frame.index)
    if not numbers:
        raise ValueError("There are no frames to look at.")
    cut_positions = find_cuts(scores)
    shots = shots_from_cuts([numbers[p] for p in cut_positions],
                            numbers[0], numbers[-1], fps)
    return shots, scores


@dataclass(frozen=True)
class SegmentShots:
    """The shots found in one segment of a video."""
    first: int                 # first frame looked at
    last: int                  # last frame looked at
    shots: list                # the Shot records, in order
    summary: ShotSummary
    cuts: list                 # frame numbers where a new shot starts


def detect_segment_shots(path, first, last, fps, on_progress=None, should_stop=None):
    """Read frames first..last of a video and find the shots in them.

    on_progress(percent) is called every few frames, with a whole number from
    0 to 100. should_stop() is checked after every frame, and when it returns
    True nothing is returned (None). Every frame of the segment is read, one
    after another.

    If frames can't be read part of the way through (some videos report more
    frames than they hold), the shots in the frames that were read are given.
    Only if not even one frame could be read is the DecodeError raised.
    """
    if first < 0 or last < first:
        raise ValueError("The segment must be a range of frames, start first.")
    length = last - first + 1
    meter = ChangeMeter()
    scores, numbers = [], []
    try:
        for frame in read_segment(path, first, last):
            scores.append(meter.add(frame.image))
            numbers.append(frame.index)
            if on_progress is not None and len(numbers) % PROGRESS_EVERY == 0:
                on_progress(int(len(numbers) * 100 / length))
            if should_stop is not None and should_stop():
                return None
    except DecodeError:
        if not numbers:
            raise
    cuts = [numbers[position] for position in find_cuts(scores)]
    shots = shots_from_cuts(cuts, numbers[0], numbers[-1], fps)
    return SegmentShots(first=numbers[0], last=numbers[-1], shots=shots,
                        summary=summarize_shots(shots), cuts=cuts)