from dataclasses import dataclass

import cv2
import numpy as np


class DecodeError(Exception):
    """Raised when frames can't be read from a video."""


@dataclass
class Frame:
    index: int          # frame number, starting at 0
    time: float         # seconds from the start of the video
    image: np.ndarray   # height x width x 3 array, colors in B, G, R order


def _open(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise DecodeError(f"Could not open video: {path}")
    return cap


def _frame_count_and_fps(cap):
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if count <= 0 or fps <= 0:
        raise DecodeError("This video does not report its length or frame rate.")
    return count, fps


def _check_range(start, end, step, count):
    if step < 1:
        raise DecodeError("Step must be 1 or more.")
    if start < 0 or end >= count:
        raise DecodeError(f"Frame numbers must be between 0 and {count - 1}.")
    if start > end:
        raise DecodeError("The segment start must not be after its end.")


def read_segment(path, start, end, step=1):
    """Yield a Frame for each frame from start to end (both included).

    step=1 gives every frame, step=5 gives every 5th frame, and so on.
    Frames are streamed one at a time, so memory use stays small.
    Note: errors are raised when you start looping over the result.
    """
    cap = _open(path)
    try:
        count, fps = _frame_count_and_fps(cap)
        _check_range(start, end, step, count)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for index in range(start, end + 1):
            if (index - start) % step == 0:
                ok, image = cap.read()      # decode and keep this frame
            else:
                ok = cap.grab()             # decode but skip it (faster)
                image = None
            if not ok:
                raise DecodeError(f"Could not read frame {index}.")
            if image is not None:
                yield Frame(index=index, time=index / fps, image=image)
    finally:
        cap.release()


def read_frame(path, index):
    """Return the single Frame at this index."""
    return next(read_segment(path, index, index))


# When the next wanted frame is farther away than this, jump straight to it
# instead of decoding all the frames in between. A jump has to decode from the
# nearest keyframe, so for close frames reading straight on is faster.
SEEK_GAP = 90


def read_frames(path, indices):
    """Yield a Frame for each frame number in `indices`.

    The numbers must be in increasing order. Frames are streamed one at a
    time. Close frames are reached by reading on, far ones by jumping.
    A number past the end of the video raises DecodeError only when the loop
    gets to it, so the frames before it are still given.
    Note: errors are raised when you start looping over the result.
    """
    indices = list(indices)
    cap = _open(path)
    try:
        count, fps = _frame_count_and_fps(cap)
        if any(index < 0 for index in indices):
            raise DecodeError("Frame numbers must not be negative.")
        for before, after in zip(indices, indices[1:]):
            if after <= before:
                raise DecodeError("Frame numbers must be in increasing order.")

        position = None       # the number of the frame cap.read() gives next
        for wanted in indices:
            # Checked as we go, so frames before a bad number are still given
            if wanted >= count:
                raise DecodeError(f"Frame numbers must be between 0 and {count - 1}.")
            if position is None or wanted - position > SEEK_GAP:
                cap.set(cv2.CAP_PROP_POS_FRAMES, wanted)
                position = wanted
            while position < wanted:
                if not cap.grab():
                    raise DecodeError(f"Could not read frame {position}.")
                position += 1
            ok, image = cap.read()
            position += 1
            if not ok:
                raise DecodeError(f"Could not read frame {wanted}.")
            yield Frame(index=wanted, time=wanted / fps, image=image)
    finally:
        cap.release()