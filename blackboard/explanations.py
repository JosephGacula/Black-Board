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
    "Contrast": (
        "How far apart the dark and bright areas of the picture are. 0% "
        "means flat: everything is about the same brightness. 100% means "
        "deep blacks and bright whites. Low contrast looks soft or foggy, "
        "and high contrast looks punchy. The darkest and brightest 5% of "
        "pixels are ignored, so a few stray pixels don't change it."
    ),
    "Crushed blacks": (
        "The share of the picture that is pure black or very close to it "
        "(about 2% brightness or less). A large share means shadow detail "
        "is lost, because dark areas have all turned into one flat black. "
        "Some black is normal, such as bars at the top and bottom of a "
        "widescreen picture or a dark background."
    ),
    "Blown highlights": (
        "The share of the picture that is pure white or very close to it "
        "(about 98% brightness or more). A large share means highlight "
        "detail is lost, because bright areas have all turned into one "
        "flat white. Some white is normal, such as a white background or "
        "a light source in the shot."
    ),
    "Dark areas": (
        "The share of the picture in the darkest third of the brightness "
        "range (about 0% to 33%). A high share means a dark, moody or "
        "underexposed image."
    ),
    "Mid-tones": (
        "The share of the picture in the middle third of the brightness "
        "range (about 33% to 67%). Most of the detail in a normally "
        "exposed picture sits here."
    ),
    "Bright areas": (
        "The share of the picture in the brightest third of the brightness "
        "range (about 67% to 100%). A high share means a bright, airy or "
        "overexposed image."
    ),
    "Saturation": (
        "How strong the colors are, from 0% (black and white, no color at "
        "all) to 100% (fully vivid). Gray and white areas count as 0%, so a "
        "mostly gray picture scores low even if it has a bright spot of "
        "color. Very dark areas are left out, because the color there is "
        "mostly camera noise."
    ),
    "Dominant hue": (
        "The most common color in the picture, such as red, green or blue, "
        "with its position on the color wheel in degrees (red is 0, green "
        "120, blue 240). Only areas with a clear color vote, so gray, black "
        "and white are ignored. It says None if the picture has almost no "
        "color."
    ),
    "Black bars": (
        "Solid black bars along the edges of the picture: at the top and "
        "bottom (letterbox, common with widescreen film) or at the sides "
        "(pillarbox, common with tall phone video on a wide screen). Each "
        "number is how thick the bar is as a share of the picture's height "
        "or width. It says None if there are no bars. The other "
        "measurements here leave the bars out, so they describe the "
        "picture itself. A very dark scene with pure black edges can look "
        "like bars."
    ),
}

# Shown when hovering over the histogram chart.
HISTOGRAM_HELP = (
    "A histogram of the picture's brightness. Each bar shows how many "
    "pixels have that brightness, from black on the left to white on the "
    "right. Tall bars on the left mean a dark image and on the right a "
    "bright one. A bar running all the way up at either end means "
    "clipping. The faint lines mark the dark, mid and bright thirds. "
    "Black bars around the picture are left out."
)

# Shown when hovering over the exposure warnings checkbox and its legend.
EXPOSURE_HELP = (
    "Paints the parts of the picture where detail is lost. Blue marks "
    "crushed blacks: areas that are pure black or very close to it. Red "
    "marks blown highlights: areas that are pure white or very close to "
    "it. These are the same pixels counted in the Crushed blacks and Blown "
    "highlights numbers. Black bars around the picture are not painted."
)

# Shown when hovering over the waveform.
WAVEFORM_HELP = (
    "Shows how bright the picture is at each position across its width. "
    "The left edge of the graph is the left edge of the picture and the "
    "right edge is the right edge. Up is brighter: the bottom is pure "
    "black and the top is pure white, with lines at 25%, 50% and 75%. "
    "Each column of the graph is a brightness histogram for that part of "
    "the picture, and the brighter the trace, the more of the picture "
    "sits at that level. A bright window on the right shows up as a "
    "bright patch at the top right. Black bars around the picture are "
    "left out."
)

# Shown when hovering over the vectorscope.
VECTORSCOPE_HELP = (
    "Shows which colors are in the picture and how strong they are. "
    "Every pixel is placed on a color map: gray and white land in the "
    "middle, and the more vivid a color is, the farther from the middle "
    "it sits. The direction tells you the color. The six small colored "
    "boxes mark where pure red, yellow, green, cyan, blue and magenta "
    "would land. A cloud leaning toward one box means the picture has a "
    "color cast. A tight blob in the middle means muted colors or black "
    "and white. The brighter the trace, the more pixels sit there. Black "
    "bars around the picture are left out."
)

# Shown when hovering over the color palette.
PALETTE_HELP = (
    "The main colors of the picture, found by grouping similar-looking "
    "pixels together and averaging each group. Each tile shows one color "
    "with its share of the picture, biggest first. The thin strip along "
    "the bottom is split by those shares, so you can see how much of the "
    "picture each color really covers. Black bars around the picture are "
    "left out. A picture with only a few different colors shows fewer "
    "than five."
)