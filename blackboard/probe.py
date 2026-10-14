import json
import subprocess
from dataclasses import dataclass
from typing import Optional


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

    
if __name__ == "__main__":
    data = run_ffprobe("samples/test1.mp4")
    print(data["format"]["duration"])
    try:
        run_ffprobe("samples/nope.mp4")
    except ProbeError as e:
        print("Caught:", e)