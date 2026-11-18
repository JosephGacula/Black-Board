"""Does this video suit the places people post videos?

Every placement (YouTube video, YouTube Shorts, Instagram Reels, ...) is a
row of facts in PLACEMENTS: the shapes it is made for, the shapes it accepts,
how long and how big a video may be. check_placement() compares one video
against one row and says, check by check, whether it is fine (OK), works with
a catch (WARN) or is a poor fit (FAIL).

The numbers in the table come from the platforms' public guides and from
sites that summarize them, as they stood in SPECS_CHECKED. Platforms change
them from time to time, so they are all in this one table, in plain data.
Where the guides disagreed, the table uses the cautious figure and says so
in `note`.
"""
import json
import os
import subprocess
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from blackboard.analyze import black_bar_lines, luma, prepare
from blackboard.decode import DecodeError, read_frame

SPECS_CHECKED = "November 2026"

OK, WARN, FAIL = "ok", "warn", "fail"

# The checks that decide the verdict, worst last
_RANK = {OK: 0, WARN: 1, FAIL: 2}

VERDICT_TEXT = {OK: "Great fit", WARN: "Works, with caveats", FAIL: "Poor fit"}

# The picture shape is judged on how much of the screen the picture fills
SAME_SHAPE = 0.98          # at least this much counts as exactly the right shape
CLOSE_ENOUGH = 0.85        # bars this thin are not worth a warning
TOO_SMALL = 0.50           # a picture filling less than this is a poor fit

# Black bars baked into the file: how many frames are looked at, how many of
# them must be readable, and how big the bars must be before they matter
BAR_SAMPLES = 9
BAR_MIN_SAMPLES = 3
BARS_MATTER = 0.03         # share of the width or height, both bars together
SHAPE_MATCH = 0.03         # frame shapes this close count as the same shape

MIN_FPS = 23.0
MAX_FPS = 60.0

KB, MB, GB = 1000, 1000 ** 2, 1000 ** 3

# Common shapes, so "1.78" can be called "16:9"
NAMED_SHAPES = [
    ((16, 9), "16:9"), ((9, 16), "9:16"), ((1, 1), "1:1"), ((4, 5), "4:5"),
    ((5, 4), "5:4"), ((4, 3), "4:3"), ((3, 4), "3:4"), ((3, 2), "3:2"),
    ((2, 3), "2:3"), ((21, 9), "21:9"), ((1.91, 1), "1.91:1"),
    ((2.39, 1), "2.39:1"),
]
NAME_TOLERANCE = 0.02


@dataclass(frozen=True)
class Placement:
    key: str
    platform: str
    name: str
    ideal: tuple                 # shapes it is made for, as (width, height) pairs
    accept_min: float            # narrowest / widest shape it shows as this kind
    accept_max: float            # of post, as width divided by height
    outside: str                 # what happens to a shape outside that range
    good_short: int = 1080       # recommended length of the picture's short side
    min_short: Optional[int] = None          # the size it asks for; smaller still posts, but looks soft
    length_min: Optional[float] = None       # seconds
    length_soft: Optional[float] = None      # longer still works, but is not the sweet spot
    length_hard: Optional[float] = None      # longer is not allowed
    size_soft: Optional[int] = None          # bytes
    size_hard: Optional[int] = None
    formats: Optional[tuple] = None          # file extensions it takes; None = nearly any
    note: str = ""


PLACEMENTS = [
    Placement(
        "youtube", "YouTube", "YouTube video",
        ideal=((16, 9),), accept_min=0.4, accept_max=2.4,
        outside="YouTube would still play it, with black bars around it.",
        length_hard=12 * 3600, size_hard=256 * GB,
        note="Up to 12 hours for verified accounts, 15 minutes otherwise."),
    Placement(
        "shorts", "YouTube", "YouTube Shorts",
        ideal=((9, 16),), accept_min=0.4, accept_max=1.0,
        outside="A video wider than tall is posted as a normal video, not a Short.",
        length_soft=60, length_hard=180, size_hard=2 * GB,
        note="Shorts may be up to 3 minutes, but most are written for 60 seconds or less."),
    Placement(
        "ig_feed", "Instagram", "Instagram feed post",
        ideal=((4, 5), (1, 1), (1.91, 1)), accept_min=0.8, accept_max=1.91,
        outside="Instagram crops the picture to its nearest allowed frame.",
        min_short=None, length_min=3, length_hard=3600, size_hard=4 * GB,
        formats=("mp4", "mov"),
        note="Guides disagree on the length limit; 60 minutes is the highest one given."),
    Placement(
        "ig_reels", "Instagram", "Instagram Reels",
        ideal=((9, 16),), accept_min=0.5, accept_max=1.91,
        outside="Instagram crops the picture to fit a Reel.",
        min_short=720, length_min=3, length_soft=180, length_hard=900,
        size_soft=650 * MB, size_hard=4 * GB, formats=("mp4", "mov"),
        note="Reels recorded in the app are up to 3 minutes; uploads may be longer."),
    Placement(
        "ig_stories", "Instagram", "Instagram Stories",
        ideal=((9, 16),), accept_min=0.5, accept_max=1.91,
        outside="Instagram crops the picture to fit a Story.",
        min_short=500, length_min=1, length_soft=15, length_hard=60,
        size_hard=4 * GB, formats=("mp4", "mov"),
        note="A Story slide lasts 15 seconds; longer videos are cut into slides."),
    Placement(
        "tiktok", "TikTok", "TikTok",
        ideal=((9, 16),), accept_min=0.5, accept_max=1.8,
        outside="TikTok crops the picture to fit.",
        min_short=720, length_soft=600, length_hard=3600,
        size_soft=287 * MB, size_hard=10 * GB, formats=("mp4", "mov"),
        note="Uploads from a phone are limited to about 72 MB (Android) or 287 MB (iPhone)."),
    Placement(
        "x", "X", "X (Twitter)",
        ideal=((16, 9), (1, 1), (9, 16)), accept_min=0.33, accept_max=3.0,
        outside="X crops the picture to fit.",
        good_short=720, length_min=0.5, length_hard=140, size_hard=512 * MB,
        formats=("mp4", "mov"),
        note="Free accounts are limited to 140 seconds."),
    Placement(
        "linkedin", "LinkedIn", "LinkedIn",
        ideal=((16, 9), (1, 1), (4, 5), (9, 16)), accept_min=0.417, accept_max=2.4,
        outside="LinkedIn does not accept shapes this extreme.",
        good_short=720, length_min=3, length_hard=600, size_hard=5 * GB,
        formats=("mp4", "mov"),
        note="Accepts shapes from 1:2.4 to 2.4:1."),
]

PLACEMENTS_BY_KEY = {p.key: p for p in PLACEMENTS}
DEFAULT_KEYS = ("youtube", "shorts", "ig_reels", "tiktok")


# ---------- the facts about one video ----------

@dataclass(frozen=True)
class Bars:
    """Black bars baked into the picture, as shares of the frame (0 to 1)."""
    top: float
    bottom: float
    left: float
    right: float
    frame_ratio: float = 0.0     # width / height of the frame they were measured in

    @property
    def horizontal(self):
        """Both side bars together, as a share of the width."""
        return self.left + self.right

    @property
    def vertical(self):
        """The top and bottom bars together, as a share of the height."""
        return self.top + self.bottom

    @property
    def matter(self):
        return self.horizontal >= BARS_MATTER or self.vertical >= BARS_MATTER


@dataclass(frozen=True)
class Clip:
    """What the checks need to know about a video, as the viewer sees it."""
    width: int                   # as displayed: rotation and pixel shape applied
    height: int
    fps: float
    duration: float              # seconds
    size: Optional[int] = None   # bytes of the file, if known
    extension: str = ""          # lower case, without the dot
    bars: Optional[Bars] = None  # black bars inside the picture, if looked for

    @property
    def has_bars(self):
        return self.bars is not None and self.bars.matter

    @property
    def picture_width(self):
        """The width of the picture itself, without black bars baked into it."""
        if not self.has_bars:
            return self.width
        return max(1, round(self.width * (1 - self.bars.horizontal)))

    @property
    def picture_height(self):
        if not self.has_bars:
            return self.height
        return max(1, round(self.height * (1 - self.bars.vertical)))


def display_size(width, height, sar=1.0, rotation=0):
    """The size a viewer sees. Phones often store a vertical video as a
    landscape one plus a note "turn it 90 degrees", and some files have pixels
    that are not square; the platforms use the turned, square-pixel shape."""
    shown_width = width * sar if sar and sar > 0 else width
    if int(round(rotation)) % 180 == 90:
        return height, int(round(shown_width))
    return int(round(shown_width)), height


def _fraction(text):
    try:
        top, bottom = str(text).split(":")
        top, bottom = float(top), float(bottom)
        return top / bottom if top > 0 and bottom > 0 else 1.0
    except (ValueError, ZeroDivisionError):
        return 1.0


def read_display_geometry(path):
    """(width, height) as displayed, or None if ffprobe cannot tell."""
    command = ["ffprobe", "-v", "quiet", "-print_format", "json", "-select_streams", "v:0",
               "-show_streams", str(path)]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=30)
        stream = json.loads(result.stdout)["streams"][0]
        rotation = 0.0
        for side in stream.get("side_data_list", []):
            if "rotation" in side:
                rotation = float(side["rotation"])
        if not rotation:
            rotation = float(stream.get("tags", {}).get("rotate", 0))
        return display_size(int(stream["width"]), int(stream["height"]),
                            _fraction(stream.get("sample_aspect_ratio", "1:1")), rotation)
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
        return None


def clip_from(path, info):
    """Gather the facts about an opened video (info is the VideoInfo from probe())."""
    size_wh = read_display_geometry(path) or (info.width, info.height)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    extension = os.path.splitext(str(path))[1].lstrip(".").lower()
    return Clip(size_wh[0], size_wh[1], info.fps, info.duration, size, extension)


# ---------- black bars baked into the picture ----------

def sample_frame_numbers(frame_count, samples=BAR_SAMPLES):
    """Frame numbers spread evenly over the video, away from the very ends."""
    if frame_count <= 0:
        return []
    count = min(samples, frame_count)
    return sorted({min(frame_count - 1, int((n + 0.5) * frame_count / count))
                   for n in range(count)})


def find_bars(path, frame_count, samples=BAR_SAMPLES, should_stop=None):
    """Look for black bars that are part of the picture, in frames spread over
    the video. Returns Bars, or None if too few frames could be read.

    One dark scene can look like it has bars, so a bar only counts if most of
    the frames have it: each side takes the middle (median) value."""
    found, ratio = [], 0.0
    for number in sample_frame_numbers(frame_count, samples):
        if should_stop is not None and should_stop():
            return None
        try:
            image = read_frame(path, number).image
        except (DecodeError, OSError):
            continue
        ratio = image.shape[1] / image.shape[0]
        gray = luma(prepare(image))
        height, width = gray.shape
        top, bottom, left, right = black_bar_lines(gray)
        found.append((top / height, bottom / height, left / width, right / width))
    if len(found) < BAR_MIN_SAMPLES:
        return None
    top, bottom, left, right = (float(v) for v in np.median(np.array(found), axis=0))
    return Bars(top, bottom, left, right, ratio)


def with_bars(clip, bars):
    """The clip with its bars known. The bars were measured in a decoded frame;
    if that frame is turned compared with the clip, top/bottom and left/right
    swap places. If the shapes do not match either way, the bars are ignored."""
    if bars is None or bars.frame_ratio <= 0:
        return clip
    wanted = clip.width / clip.height
    if abs(bars.frame_ratio / wanted - 1) <= SHAPE_MATCH:
        return replace(clip, bars=bars)
    if abs(bars.frame_ratio * wanted - 1) <= SHAPE_MATCH:
        turned = Bars(bars.left, bars.right, bars.top, bars.bottom, 1 / bars.frame_ratio)
        return replace(clip, bars=turned)
    return clip


# ---------- words for numbers ----------

def shape_name(ratio):
    """'9:16' for 0.5625, or '1.37:1' if it is not a common shape."""
    for (top, bottom), name in NAMED_SHAPES:
        if abs(ratio / (top / bottom) - 1) <= NAME_TOLERANCE:
            return name
    return f"{ratio:.2f}:1" if ratio >= 1 else f"1:{1 / ratio:.2f}"


def shape_text(width, height):
    return shape_name(width / height)


def duration_text(seconds):
    if seconds < 60:
        return f"{seconds:g} s" if seconds == int(seconds) else f"{seconds:.1f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    return f"{minutes}:{rest:02d}"


def size_text(size):
    if size >= GB:
        return f"{size / GB:.1f} GB"
    if size >= MB:
        return f"{size / MB:.0f} MB"
    return f"{size / KB:.0f} KB"


def percent(share):
    return f"{share * 100:.0f} %"


# ---------- the checks ----------

@dataclass(frozen=True)
class Check:
    name: str
    status: str
    message: str


@dataclass(frozen=True)
class Report:
    placement: Placement
    checks: list = field(default_factory=list)

    @property
    def verdict(self):
        return max((c.status for c in self.checks), key=_RANK.get, default=OK)

    @property
    def verdict_text(self):
        return VERDICT_TEXT[self.verdict]

    @property
    def problems(self):
        """The checks that are not plainly fine, worst first."""
        bad = [c for c in self.checks if c.status != OK]
        return sorted(bad, key=lambda c: -_RANK[c.status])

    @property
    def score(self):
        """For sorting reports, best first: fewer and milder problems rank higher."""
        return (_RANK[self.verdict], sum(_RANK[c.status] for c in self.checks))


def fill_share(ratio, ideal_ratio):
    """How much of an ideal frame the picture fills when it is fitted inside
    it (the rest is black bars). 1.0 is a perfect match."""
    return min(ratio, ideal_ratio) / max(ratio, ideal_ratio)


def closest_ideal(ratio, ideal):
    """The (width, height) in `ideal` that the picture fills best."""
    return max(ideal, key=lambda pair: fill_share(ratio, pair[0] / pair[1]))


def check_shape(clip, place):
    ratio = clip.picture_width / clip.picture_height
    pair = closest_ideal(ratio, place.ideal)
    target = pair[0] / pair[1]
    fill = fill_share(ratio, target)
    mine, wanted = shape_name(ratio), shape_name(target)
    crop = 1 - fill                      # share of the picture lost by filling the frame

    inside = " inside the black bars" if clip.has_bars else ""
    if fill >= SAME_SHAPE:
        return Check("Picture shape", OK,
                     f"{mine}{inside}, the shape {place.name} is made for.")
    detail = (f"Your picture is {mine}{inside}; {place.name} is made for {wanted}. "
              f"Fitted inside it, the picture fills {percent(fill)} of the screen; "
              f"filling the screen instead crops away {percent(crop)} of the picture.")
    if not place.accept_min <= ratio <= place.accept_max:
        return Check("Picture shape", FAIL, f"{detail} {place.outside}")
    if fill >= CLOSE_ENOUGH:
        return Check("Picture shape", OK, f"{mine}{inside}, close to {wanted}. Only thin bars.")
    return Check("Picture shape", FAIL if fill < TOO_SMALL else WARN, detail)


def check_resolution(clip, place):
    """A small picture is never refused outright, so it only ever warns; a size
    under the minimum the platform asks for gets the stronger wording."""
    short = min(clip.picture_width, clip.picture_height)
    if place.min_short and short < place.min_short:
        return Check("Resolution", WARN,
                     f"The short side is {short} px; {place.name} asks for at least "
                     f"{place.min_short} px, so it may look soft.")
    if short < place.good_short:
        return Check("Resolution", WARN,
                     f"The short side is {short} px; {place.good_short} px looks sharper.")
    return Check("Resolution", OK, f"The short side is {short} px, which is plenty.")


def check_bars(clip, place):
    """Black bars in the file are posted along with the picture."""
    if not clip.has_bars:
        return None
    parts = []
    if clip.bars.horizontal >= BARS_MATTER:
        parts.append(f"{percent(clip.bars.horizontal)} of the width (left and right)")
    if clip.bars.vertical >= BARS_MATTER:
        parts.append(f"{percent(clip.bars.vertical)} of the height (top and bottom)")
    return Check("Black bars", WARN,
                 f"The file has black bars taking {' and '.join(parts)}. {place.name} shows "
                 f"them as part of your video, so crop them off before posting. The other "
                 f"checks judge the {clip.picture_width}x{clip.picture_height} picture "
                 f"inside the bars.")


def check_length(clip, place):
    length = clip.duration
    if place.length_min and length < place.length_min:
        return Check("Length", FAIL, f"{duration_text(length)} is shorter than the "
                                     f"{duration_text(place.length_min)} minimum.")
    if place.length_hard and length > place.length_hard:
        return Check("Length", FAIL, f"{duration_text(length)} is longer than the "
                                     f"{duration_text(place.length_hard)} limit.")
    if place.length_soft and length > place.length_soft:
        return Check("Length", WARN, f"{duration_text(length)} is allowed, but it is "
                                     f"longer than the usual {duration_text(place.length_soft)}.")
    return Check("Length", OK, f"{duration_text(length)} is within what {place.name} allows.")


def check_frame_rate(clip, place):
    fps = clip.fps
    if fps < MIN_FPS:
        return Check("Frame rate", WARN, f"{fps:.2f} fps is low; {MIN_FPS:g} fps or more "
                                         "plays smoothly everywhere.")
    if fps > MAX_FPS:
        return Check("Frame rate", WARN, f"{fps:.2f} fps is above {MAX_FPS:g} fps, "
                                         "so it may be reduced when uploaded.")
    return Check("Frame rate", OK, f"{fps:.2f} fps is fine.")


def check_file_size(clip, place):
    if clip.size is None or not (place.size_soft or place.size_hard):
        return None
    if place.size_hard and clip.size > place.size_hard:
        return Check("File size", FAIL, f"{size_text(clip.size)} is above the "
                                        f"{size_text(place.size_hard)} limit.")
    if place.size_soft and clip.size > place.size_soft:
        return Check("File size", WARN, f"{size_text(clip.size)} is allowed, but uploads "
                                        f"from a phone may fail above {size_text(place.size_soft)}.")
    return Check("File size", OK, f"{size_text(clip.size)} is within the limit.")


def check_file_type(clip, place):
    if not place.formats or not clip.extension:
        return None
    if clip.extension in place.formats:
        return Check("File type", OK, f".{clip.extension} is accepted.")
    wanted = " or ".join(f".{f}" for f in place.formats)
    return Check("File type", WARN, f".{clip.extension} may be refused; {wanted} is safe.")


def check_placement(clip, place):
    """Compare one video with one placement."""
    checks = [check_shape(clip, place), check_bars(clip, place), check_resolution(clip, place),
              check_length(clip, place), check_frame_rate(clip, place),
              check_file_size(clip, place), check_file_type(clip, place)]
    return Report(place, [c for c in checks if c is not None])


def check_places(clip, keys=None):
    """Reports for the chosen placements (all of them if keys is None), best fit first."""
    keys = [p.key for p in PLACEMENTS] if keys is None else list(keys)
    order = {p.key: n for n, p in enumerate(PLACEMENTS)}
    reports = [check_placement(clip, PLACEMENTS_BY_KEY[k]) for k in keys
               if k in PLACEMENTS_BY_KEY]
    return sorted(reports, key=lambda r: (r.score, order[r.placement.key]))


def clip_summary(clip):
    """One line that says what is being checked."""
    parts = [f"{clip.width}x{clip.height} ({shape_text(clip.width, clip.height)})"]
    if clip.has_bars:
        parts.append(f"picture inside black bars {clip.picture_width}x{clip.picture_height} "
                     f"({shape_text(clip.picture_width, clip.picture_height)})")
    parts += [f"{clip.fps:.2f} fps", duration_text(clip.duration)]
    if clip.size is not None:
        parts.append(size_text(clip.size))
    return ", ".join(parts)