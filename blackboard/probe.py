import json
import subprocess
from dataclasses import dataclass
from typing import Optional
from fractions import Fraction


@dataclass
class VideoInfo:
    # Always present in a valid video
    codec: str                 # e.g. "h264"
    width: int                 # pixels
    height: int                # pixels
    fps: float                 # frames per second, e.g. 29.97
    duration: float            # seconds
    has_audio: bool

    # ffprobe sometimes leaves these out, so they can be None
    profile: Optional[str] = None       # e.g. "High"
    bitrate: Optional[int] = None       # video stream, bits per second
    frame_count: Optional[int] = None
    pix_fmt: Optional[str] = None       # e.g. "yuv420p"
    color_space: Optional[str] = None   # e.g. "bt709"
    keyframe_count: Optional[int] = None   # number of keyframes (I-frames)
    avg_gop: Optional[float] = None        # average frames between keyframes
    max_gop: Optional[int] = None          # longest gap between keyframes

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

class ProbeError(Exception):
    """Raised when a video can't be read."""


def run_ffprobe(path):
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        raise ProbeError("ffprobe was not found on this computer.")

    if result.returncode != 0:
        raise ProbeError(f"Could not read this file as a video: {path}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ProbeError("ffprobe returned output that could not be understood.")

def _to_int(value):
    """Convert ffprobe's string values like "2356052" to int, or None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value):
    """Convert ffprobe's string values like "11.578233" to float, or None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def get_gop_info(path, stream_index):
    """Count keyframes and work out GOP length for one video stream.

    Returns (keyframe_count, avg_gop, max_gop), or (None, None, None)
    if ffprobe can't report it.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", str(stream_index),
        "-show_entries", "packet=flags",
        "-of", "csv=p=0",
        str(path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        return None, None, None

    # One line per frame (packet), e.g. "K__" for a keyframe, "___" otherwise
    flags = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    keyframe_positions = [i for i, f in enumerate(flags) if "K" in f]
    if not keyframe_positions:
        return None, None, None

    # GOP lengths: frames from one keyframe up to the next (and to the end)
    boundaries = keyframe_positions + [len(flags)]
    gop_lengths = [boundaries[i + 1] - boundaries[i]
                   for i in range(len(keyframe_positions))]

    avg_gop = sum(gop_lengths) / len(gop_lengths)
    return len(keyframe_positions), avg_gop, max(gop_lengths)

def _parse_fps(stream):
    """Turn "30000/1001" into 29.97. Some files give "0/0", so try both fields."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            fps = float(Fraction(stream[key]))
        except (KeyError, ZeroDivisionError, ValueError):
            continue
        if fps > 0:
            return fps
    return 0.0


def probe(path):
    data = run_ffprobe(path)
    streams = data.get("streams", [])

    # 1. Find the real video stream (skip embedded cover art)
    video = None
    for s in streams:
        is_video = s.get("codec_type") == "video"
        is_cover_art = s.get("disposition", {}).get("attached_pic", 0) == 1
        if is_video and not is_cover_art:
            video = s
            break
    if video is None:
        raise ProbeError("No video stream found in this file.")

    # 2. Is there an audio stream?
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    # 3. Duration: prefer the video stream, fall back to the container
    duration = _to_float(video.get("duration"))
    if duration is None:
        duration = _to_float(data.get("format", {}).get("duration"))
    if duration is None:
        duration = 0.0

    # 4. Keyframes and GOP length
    keyframes, avg_gop, max_gop = get_gop_info(path, video.get("index", 0))

    # 5. Build the result
    try:
        return VideoInfo(
            codec=video["codec_name"],
            width=video["width"],
            height=video["height"],
            fps=_parse_fps(video),
            duration=duration,
            has_audio=has_audio,
            profile=video.get("profile"),
            bitrate=_to_int(video.get("bit_rate")),
            frame_count=_to_int(video.get("nb_frames")),
            pix_fmt=video.get("pix_fmt"),
            color_space=video.get("color_space"),
            keyframe_count=keyframes,
            avg_gop=avg_gop,
            max_gop=max_gop,
        )
    except KeyError as e:
        raise ProbeError(f"The video stream is missing required data: {e}")
    
if __name__ == "__main__":
    print(probe("samples/test1.mp4"))