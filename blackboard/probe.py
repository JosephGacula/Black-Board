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

if __name__ == "__main__":
    info = VideoInfo(codec="h264", width=1920, height=1008,
                     fps=29.97, duration=11.58, has_audio=True)
    print(info)
    print(info.resolution)