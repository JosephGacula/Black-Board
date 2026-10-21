"""Plain-language explanations of each measurement.

Used for the hover tooltips in the window, and later in the PDF report.
Keys must match the field names shown in the window.
"""

FIELD_HELP = {
    "Codec": (
        "The compression format used to store the video, such as H.264 or "
        "HEVC. The codec decides how frames are squeezed into a smaller "
        "file, and what a device must support to play it back."
    ),
    "Profile": (
        "A set of features the codec is allowed to use. Higher profiles "
        "(Baseline, Main, High) use more advanced compression tools, which "
        "gives smaller files at the same quality but needs more work to decode."
    ),
    "Resolution": (
        "The width and height of each frame in pixels. 1920x1080 is "
        "'Full HD' and 3840x2160 is '4K'. More pixels means more detail "
        "and more data to store per frame."
    ),
    "Frame rate": (
        "How many frames are shown per second. Film uses 24, and web video "
        "is often 30 or 60. A value like 29.97 is really 30000/1001, a "
        "leftover from the way early color TV worked."
    ),
    "Duration": "The length of the video in seconds.",
    "Frames": (
        "The total number of individual images in the video. It is roughly "
        "the duration times the frame rate."
    ),
    "Bitrate": (
        "How much data is used for each second of video, in kilobits per "
        "second (video only, not audio). Higher bitrates keep more detail "
        "but make bigger files: file size is about bitrate x duration."
    ),
    "Pixel format": (
        "How color is stored for each pixel. 'yuv420p' keeps brightness for "
        "every pixel but color for only one pixel in each 2x2 block, because "
        "eyes notice brightness more than color. It saves space with little "
        "visible loss."
    ),
    "Color space": (
        "The standard that defines what the color values mean. bt709 is "
        "the standard for HD video, and bt2020 is used for HDR and UHD. "
        "'unknown' means the file doesn't say, so players have to guess."
    ),
    "Keyframes": (
        "The number of keyframes (I-frames): complete images that can be "
        "decoded on their own. The frames between them store only what "
        "changed, so seeking and editing have to start from a keyframe."
    ),
    "Average GOP": (
        "GOP means 'group of pictures': the frames from one keyframe up to "
        "the next. A short GOP (about 30 frames, or 1 second) seeks and "
        "edits quickly but needs more data. A long GOP (150 or more) "
        "compresses better."
    ),
    "Audio": "Whether the file contains an audio track.",
}