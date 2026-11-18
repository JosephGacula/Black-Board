"""Shows why stripes of the motion strip are measured or left out.

Run it from the project folder, with the virtual environment on:

    python motion_check.py "C:\\path\\to\\video.mp4" FIRST LAST [HOW_MANY]

For HOW_MANY evenly spaced frames in FIRST..LAST (default 20) it compares the
frame with the next one and prints:
  diff    how different the two frames are (0-255; 10 or more counts as 'changed')
  after   how different they are once the flow has moved the second frame back
  ratio   after / diff (0.6 or more, with diff 10 or more, is left out)
  flow    the average distance moved, in pixels of the 160-wide picture
  motion  the result in % of the picture width per second, or 'left out'
  ndiff, nratio   the same two numbers when each frame's brightness is first
                  evened out (same average and spread), which removes flicker

To see frames, add LOOK and a frame number instead of HOW_MANY:

    python motion_check.py "C:\\path\\to\\video.mp4" FIRST LAST look 4061

That writes look.jpg: the frame and the next five, side by side.
"""
import sys

import cv2
import numpy as np

from blackboard.analyze import black_bar_lines, crop_bars, luma, prepare
from blackboard.decode import read_frames
from blackboard.motion import (
    UNRELATED_DIFFERENCE, UNRELATED_RATIO, _gray_small,
)
from blackboard.probe import probe


def look(path, start):
    """Save the frame `start` and the five after it side by side."""
    tiles = []
    for frame in read_frames(path, list(range(start, start + 6))):
        small = prepare(frame.image)
        tile = cv2.resize(small, (320, max(1, round(small.shape[0] * 320 / small.shape[1]))))
        cv2.putText(tile, str(frame.index), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        tiles.append(tile)
    height = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
             for t in tiles]
    cv2.imwrite("look.jpg", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    print("Wrote look.jpg")


def evened(gray):
    """The picture with its brightness set to a fixed average and spread."""
    g = gray.astype(np.float32)
    std = g.std() or 1.0
    return np.clip((g - g.mean()) / std * 40 + 128, 0, 255).astype(np.uint8)


def residual(ga, gb):
    flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = ga.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    back = cv2.remap(gb, gx + flow[..., 0], gy + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    diff = float(np.abs(ga.astype(np.float32) - gb).mean())
    after = float(np.abs(ga.astype(np.float32) - back).mean())
    return flow, diff, after


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return
    path, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if len(sys.argv) > 5 and sys.argv[4].lower() == "look":
        look(path, int(sys.argv[5]))
        return
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    fps = probe(path).fps
    starts = sorted({int(first + (last - 1 - first) * i / max(count - 1, 1))
                     for i in range(count)})
    wanted = set(starts)
    needed = sorted(wanted | {s + 1 for s in starts})
    print(f"{'frame':>7} {'diff':>6} {'after':>6} {'ratio':>6} {'flow px':>8}  {'motion':<10} {'ndiff':>6} {'nratio':>6}")
    previous = None
    for frame in read_frames(path, needed):
        if previous and previous[0] in wanted and previous[0] + 1 == frame.index:
            a, b = previous[1], frame.image
            small_a, small_b = prepare(a), prepare(b)
            lines = black_bar_lines(luma(small_a))
            ga = _gray_small(crop_bars(small_a, lines))
            gb = _gray_small(crop_bars(small_b, lines))
            flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            h, w = ga.shape
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            back = cv2.remap(gb, gx + flow[..., 0], gy + flow[..., 1],
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            diff = float(np.abs(ga.astype(np.float32) - gb).mean())
            after = float(np.abs(ga.astype(np.float32) - back).mean())
            ratio = after / diff if diff else 0.0
            px = float(np.hypot(flow[..., 0], flow[..., 1]).mean())
            out = diff >= UNRELATED_DIFFERENCE and after >= UNRELATED_RATIO * diff
            motion = "left out" if out else f"{px / w * 100 * fps:.1f} %/s"
            _, nd, na = residual(evened(ga), evened(gb))
            nratio = na / nd if nd else 0.0
            print(f"{previous[0]:>7} {diff:>6.1f} {after:>6.1f} {ratio:>6.2f} {px:>8.2f}  {motion:<10} {nd:>6.1f} {nratio:>6.2f}")
        previous = (frame.index, frame.image)


main()