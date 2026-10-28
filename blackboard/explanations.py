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
    "Audio tracks": (
        "How many separate audio tracks the file holds. Most videos have "
        "one. Extra tracks might carry another language, commentary, or a "
        "separate mix. The details below describe the first track."
    ),
    "Audio codec": (
        "The compression format used to store the sound, such as AAC (the "
        "usual choice for MP4) or Opus. Like a video codec, it decides how "
        "small the audio is squeezed and which players can play it."
    ),
    "Sample rate": (
        "How many times per second the sound is measured, in kilohertz. "
        "44.1 kHz is CD quality and 48 kHz is the standard for video. "
        "Higher rates can capture higher pitches, but 48 kHz already "
        "covers everything people can hear."
    ),
    "Channels": (
        "How many separate sound signals are stored. Mono is one (the same "
        "sound from every speaker), stereo is two (left and right), and "
        "surround formats have more, such as 6 for 5.1."
    ),
    "Audio bitrate": (
        "How much data is used for each second of sound, in kilobits per "
        "second. Higher bitrates keep more audio detail and make bigger "
        "files. Stereo AAC is often somewhere around 128 to 192 kbps."
    ),
    "Audio length": (
        "Whether the sound runs for the same time as the picture. A gap of "
        "a fraction of a second is normal. A bigger gap usually means the "
        "sound is cut off early, or the picture keeps going after the "
        "sound has ended."
    ),
}

# Explanations for the "Frame analysis" rows. Keys must match ANALYSIS_NAMES
# in the window.
ANALYSIS_HELP = {
    "Brightness": (
        "How light the picture is on average, from 0% (pure black) to "
        "100% (pure white). Low numbers mean a dark image and high numbers "
        "a bright one. It is measured the way the eye sees it, so green "
        "counts the most and blue the least."
    ),
}