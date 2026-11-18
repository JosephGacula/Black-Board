import colorsys
import textwrap

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QSizePolicy, QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QSlider, QSpinBox, QCheckBox, QComboBox, QStackedWidget,
)

from blackboard.analyze import (
    EXPOSURE_BLOWN_BGR, EXPOSURE_CRUSHED_BGR, analyze_frame, exposure_overlay,
    hue_name,
)
from blackboard.barcode import barcode_columns, brightness_colors, build_barcode
from blackboard.decode import read_frame, DecodeError
from blackboard.explanations import (
    ANALYSIS_HELP, BARCODE_HELP, BRIGHTNESS_BARCODE_HELP, EXPOSURE_HELP, MOTION_HELP, FIELD_HELP, HISTOGRAM_HELP, PALETTE_HELP,
    PLATFORM_HELP, SEGMENT_HELP, SHOTS_HELP, VECTORSCOPE_HELP, WAVEFORM_HELP,
)
from blackboard.motion import build_motion, motion_colors, summarize_motion
from blackboard.platforms import DEFAULT_KEYS, clip_from, find_bars, with_bars
from blackboard.probe import probe, ProbeError
from blackboard.shots import detect_segment_shots
from blackboard.ui.barcode_view import BarcodeView
from blackboard.ui.histogram_view import HistogramView
from blackboard.ui.palette_view import PaletteView
from blackboard.ui.platform_dialog import PlatformDialog
from blackboard.ui.segment_picker import SegmentBand, SegmentPicker
from blackboard.ui.shot_list import ShotListView
from blackboard.ui.vectorscope_view import VectorscopeView
from blackboard.ui.waveform_view import WaveformView

FIELD_NAMES = ["Codec", "Profile", "Resolution", "Frame rate", "Duration",
               "Frames", "Bitrate", "Pixel format", "Color space",
               "Keyframes", "Average GOP", "Audio", "Audio tracks",
               "Audio codec", "Sample rate", "Channels", "Audio bitrate",
               "Audio length"]

# Rows that describe the audio track itself; they say "n/a" if there is none
AUDIO_DETAIL_FIELDS = ["Audio codec", "Sample rate", "Channels",
                       "Audio bitrate", "Audio length"]

# Rows of the "Frame analysis" panel under the preview
ANALYSIS_NAMES = ["Brightness", "Contrast", "Crushed blacks",
                  "Blown highlights", "Saturation", "Dark areas",
                  "Mid-tones", "Bright areas", "Dominant hue", "Black bars"]

# The charts the drop-down above the chart area can show, in order
SCOPE_NAMES = ["Luminance histogram", "Waveform", "Color palette", "Vectorscope"]

# The panel fills this many rows, then starts a new column, so it never
# grows taller however many measurements we add.
ANALYSIS_ROWS = 5

# Side of the little square that shows the dominant hue, in pixels
SWATCH_SIZE = 14

PREVIEW_W, PREVIEW_H = 640, 360

# Text of the button that starts the color barcode analysis, and of the same
# button while the analysis runs
ANALYZE_TEXT = "Analyze color barcode"
CANCEL_TEXT = "Cancel"
SHOTS_TEXT = "Detect shots"
MOTION_TEXT = "Analyze motion"
PLATFORMS_TEXT = "Check platforms"


def fmt(value, template="{}"):
    """Show 'unknown' for missing values instead of 'None'."""
    return "unknown" if value is None else template.format(value)


def channel_label(channels):
    """1 -> 'Mono', 2 -> 'Stereo', 6 -> '6 channels'."""
    if channels is None:
        return "unknown"
    names = {1: "Mono", 2: "Stereo"}
    return names.get(channels, f"{channels} channels")


def audio_length_text(info):
    """Say whether the audio and video lengths match, in plain words."""
    diff = info.audio_length_diff
    if diff is None:
        return "unknown"
    if info.audio_matches_video:
        return f"Matches video (off by {abs(diff):.2f} s)"
    if diff < 0:
        return f"Audio is {abs(diff):.2f} s shorter"
    return f"Audio is {diff:.2f} s longer"


def hue_text(degrees):
    """Dominant hue for the panel: 'Purple (275\u00b0)', or 'None' if neutral."""
    if degrees is None:
        return "None (neutral)"
    return f"{hue_name(degrees)} ({degrees:.0f}\u00b0)"


def bars_text(bars):
    """Black bars for the panel: 'None', 'Top/bottom 12.5 %', 'Left 5.0 %'..."""
    if not bars.any:
        return "None"

    def pair(first, second, first_name, second_name):
        if not first and not second:
            return None
        if abs(first - second) <= 1.0:                 # same on both sides
            return f"{first_name}/{second_name} {(first + second) / 2:.1f} %"
        parts = [f"{name} {value:.1f} %"
                 for name, value in ((first_name, first), (second_name, second))
                 if value]
        return ", ".join(parts)

    parts = [pair(bars.top, bars.bottom, "top", "bottom"),
             pair(bars.left, bars.right, "left", "right")]
    text = ", ".join(part for part in parts if part)
    return text[0].upper() + text[1:]              # only the first word is capitalized


def motion_text(summary):
    """The line under the motion strip."""
    text = (f"Motion: average {summary.average:.1f} % of the picture width per second, "
            f"busiest {summary.peak:.1f} at frame {summary.peak_frame}, "
            f"calmest {summary.calmest:.1f} at frame {summary.calmest_frame}")
    if summary.skipped:
        cuts = "a cut" if summary.skipped == 1 else f"{summary.skipped} cuts"
        text += f" ({cuts} left out)"
    if summary.rough:
        share = round(summary.rough / summary.measured * 100)
        text += (f". {share} % of the strip, shown washed out, is a rough estimate: "
                 f"the footage changes too much between frames to follow")
    return text


def bgr_hex(color):
    """A (B, G, R) color as '#rrggbb', for styling the legend."""
    blue, green, red = color
    return f"#{red:02x}{green:02x}{blue:02x}"


def hue_color(degrees):
    """The hue at full strength as '#rrggbb', for the little color square.

    It is always fully vivid and bright, so it stays easy to see even when
    the frame itself is dark. (0 is red, 120 green, 240 blue.)
    """
    red, green, blue = colorsys.hsv_to_rgb((degrees % 360) / 360, 1.0, 1.0)
    return "#{:02x}{:02x}{:02x}".format(
        round(red * 255), round(green * 255), round(blue * 255))


def last_frame_index(info):
    """Number of the video's last frame, or None if we can't tell."""
    if info.frame_count:
        return info.frame_count - 1
    if info.duration > 0 and info.fps > 0:
        return max(int(round(info.duration * info.fps)) - 1, 0)
    return None


def shrink_for_preview(image):
    """Shrink an OpenCV frame (a B, G, R array) to fit the preview box.
    Never enlarges it."""
    h, w = image.shape[:2]
    scale = min(PREVIEW_W / w, PREVIEW_H / h, 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def frame_to_qimage(image):
    """Shrink an OpenCV frame to fit the preview box and turn it into a QImage."""
    image = shrink_for_preview(image)
    h, w = image.shape[:2]
    qimage = QImage(image.data, w, h, image.strides[0], QImage.Format.Format_BGR888)
    return qimage.copy()   # copy, so the QImage owns its pixels


class ProbeThread(QThread):
    """Runs probe() in the background so the window stays responsive."""
    progress = Signal(int)    # 0-100, or -1 when the amount of work is unknown
    done = Signal(object)     # the finished VideoInfo
    failed = Signal(str)      # an error message to show the user

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            info = probe(self.path, progress=self.progress.emit)
        except ProbeError as e:
            self.failed.emit(str(e))
        except Exception as e:  # anything unexpected: report it, don't hang
            self.failed.emit(f"Unexpected error: {e}")
        else:
            self.done.emit(info)


class FrameThread(QThread):
    """Reads one frame in the background, turns it into a QImage and
    measures it."""
    # video path, frame number, QImage, FrameStats (None if measuring failed)
    # and the QImage with exposure warnings painted on (None if that failed)
    loaded = Signal(str, int, object, object, object)
    failed = Signal(str, int, str)      # video path, frame number, message

    def __init__(self, path, index, parent=None):
        super().__init__(parent)
        self.path = path
        self.index = index

    def run(self):
        try:
            frame = read_frame(self.path, self.index)
            shrunk = shrink_for_preview(frame.image)
            image = frame_to_qimage(shrunk)
        except DecodeError as e:
            self.failed.emit(self.path, self.index, str(e))
            return
        except Exception as e:
            self.failed.emit(self.path, self.index, f"Unexpected error: {e}")
            return

        # A problem in the measurements shouldn't stop the picture showing
        try:
            stats = analyze_frame(frame.image)
        except Exception:
            stats = None
        try:
            overlay = frame_to_qimage(exposure_overlay(shrunk))
        except Exception:
            overlay = None
        self.loaded.emit(self.path, self.index, image, stats, overlay)


class BarsThread(QThread):
    """Looks for black bars baked into the picture, in frames across the video."""
    found = Signal(str, object)         # video path, Bars (None if it could not tell)

    def __init__(self, path, frame_count, parent=None):
        super().__init__(parent)
        self.path = path
        self.frame_count = frame_count

    def run(self):
        self.setPriority(QThread.Priority.LowPriority)
        try:
            bars = find_bars(self.path, self.frame_count,
                             should_stop=self.isInterruptionRequested)
        except Exception:
            bars = None
        if not self.isInterruptionRequested():
            self.found.emit(self.path, bars)


class BarcodeThread(QThread):
    """Measures the stripes of the color barcode of one segment in the
    background.

    Every signal carries a generation number, so the window can tell results
    for the video that is open now from late ones for a video it has left.
    """
    updated = Signal(int, object)     # generation, the stripes measured so far
    done = Signal(int, object)        # generation, all the stripes
    failed = Signal(int, str)         # generation, an error message

    def __init__(self, generation, path, first, last, parent=None):
        super().__init__(parent)
        self.generation = generation
        self.path = path
        self.first = first
        self.last = last

    def run(self):
        try:
            colors = build_barcode(
                self.path, self.first, self.last,
                on_update=lambda found: self.updated.emit(self.generation, found),
                should_stop=self.isInterruptionRequested,
            )
        except DecodeError as e:
            self.failed.emit(self.generation, str(e))
        except Exception as e:
            self.failed.emit(self.generation, f"Unexpected error: {e}")
        else:
            if not self.isInterruptionRequested():
                self.done.emit(self.generation, colors)


class ShotThread(QThread):
    """Finds the shots of one segment in the background.

    Like the barcode thread, every signal carries a generation number so
    late results for an older run are recognized.
    """
    progress = Signal(int, int)       # generation, percent done
    done = Signal(int, object)        # generation, a SegmentShots
    failed = Signal(int, str)         # generation, an error message

    def __init__(self, generation, path, first, last, fps, parent=None):
        super().__init__(parent)
        self.generation = generation
        self.path = path
        self.first = first
        self.last = last
        self.fps = fps

    def run(self):
        try:
            result = detect_segment_shots(
                self.path, self.first, self.last, self.fps,
                on_progress=lambda percent: self.progress.emit(self.generation, percent),
                should_stop=self.isInterruptionRequested,
            )
        except DecodeError as e:
            self.failed.emit(self.generation, str(e))
        except Exception as e:
            self.failed.emit(self.generation, f"Unexpected error: {e}")
        else:
            if result is not None and not self.isInterruptionRequested():
                self.done.emit(self.generation, result)


class MotionThread(QThread):
    """Measures the motion strip of one segment in the background."""
    updated = Signal(int, object)     # generation, the values so far
    done = Signal(int, object)        # generation, a MotionData
    failed = Signal(int, str)         # generation, an error message

    def __init__(self, generation, path, first, last, fps, parent=None):
        super().__init__(parent)
        self.generation = generation
        self.path = path
        self.first = first
        self.last = last
        self.fps = fps

    def run(self):
        try:
            data = build_motion(
                self.path, self.first, self.last, self.fps,
                on_update=lambda found: self.updated.emit(self.generation, found),
                should_stop=self.isInterruptionRequested,
            )
        except DecodeError as e:
            self.failed.emit(self.generation, str(e))
        except Exception as e:
            self.failed.emit(self.generation, f"Unexpected error: {e}")
        else:
            if not self.isInterruptionRequested():
                self.done.emit(self.generation, data)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Black Board")
        self.resize(1000, 900)

        self.probe_thread = None
        self.frame_thread = None
        self.barcode_thread = None
        self.barcode_generation = 0   # which barcode run we are waiting for
        self.barcode_running = False
        self.shots_thread = None
        self.shots_generation = 0     # which shot detection run we are waiting for
        self.shots_running = False
        self.motion_thread = None
        self.motion_generation = 0    # which motion run we are waiting for
        self.motion_running = False
        self.motion_frames = []       # the frame behind each stripe of the motion strip
        self.current_path = ""
        self.info = None          # the VideoInfo of the open video
        self.platform_choices = list(DEFAULT_KEYS)   # places picked in the platform window
        self.bars_found = {}      # video path -> the black bars found in it (or None)
        self.wanted_index = 0     # the frame the user is asking for
        self.last_requested = -1  # the frame we most recently started loading

        # Wait a moment after the slider stops moving before loading a frame
        self.frame_timer = QTimer(self)
        self.frame_timer.setSingleShot(True)
        self.frame_timer.setInterval(120)
        self.frame_timer.timeout.connect(self.request_frame)

        # --- Top: open button, file name, progress bar ---
        self.open_button = QPushButton("Open video...")
        self.open_button.clicked.connect(self.choose_file)

        self.file_label = QLabel("No video selected")
        self.file_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        # Keep its space reserved so the layout doesn't jump when it appears
        policy = self.progress_bar.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.progress_bar.setSizePolicy(policy)

        # --- Left: one label per measurement ---
        self.fields = {}
        form = QFormLayout()
        for name in FIELD_NAMES:
            # Hovering over either the name or the value shows an explanation
            tip = textwrap.fill(FIELD_HELP[name], width=55)
            name_label = QLabel(name + ":")
            name_label.setToolTip(tip)
            value_label = QLabel("-")
            value_label.setToolTip(tip)
            self.fields[name] = value_label
            form.addRow(name_label, value_label)

        # --- Right: frame preview with slider and number box ---
        self.preview = QLabel()
        self.preview.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background-color: #111; color: #aaa;")

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_spin = QSpinBox()
        # Keep the slider and the number box in step with each other
        self.frame_slider.valueChanged.connect(self.frame_spin.setValue)
        self.frame_spin.valueChanged.connect(self.frame_slider.setValue)
        self.frame_spin.valueChanged.connect(self.on_frame_changed)

        # Choosing a segment: boxes for where it starts and ends, and a band
        # under the slider that lights up the chosen stretch
        segment_tip = textwrap.fill(SEGMENT_HELP, width=55)
        self.segment_band = SegmentBand()
        self.segment_band.setToolTip(segment_tip)
        self.segment_picker = SegmentPicker()
        self.segment_picker.setToolTip(segment_tip)
        self.segment_picker.changed.connect(self.segment_band.set_segment)
        self.segment_picker.start_from_current.connect(self.use_current_as_start)
        self.segment_picker.end_from_current.connect(self.use_current_as_end)

        # Analyzing the segment's colors is a separate step the user starts
        self.analyze_button = QPushButton(ANALYZE_TEXT)
        self.analyze_button.setToolTip(textwrap.fill(BARCODE_HELP, width=55))
        self.analyze_button.clicked.connect(self.on_analyze_clicked)
        self.barcode_status = QLabel("")

        # Measuring the motion of the segment is a third step
        motion_tip = textwrap.fill(MOTION_HELP, width=55)
        self.motion_button = QPushButton(MOTION_TEXT)
        self.motion_button.setToolTip(motion_tip)
        self.motion_button.clicked.connect(self.on_motion_clicked)
        self.motion_view = BarcodeView()
        self.motion_view.setToolTip(motion_tip)
        self.motion_view.seek_requested.connect(self.frame_spin.setValue)
        self.motion_status = QLabel("")
        self.motion_status.setWordWrap(True)      # the line can be long
        self.motion_status.setMinimumWidth(100)
        self.motion_status.setToolTip(motion_tip)

        # Checking how well the video suits the places it will be posted. This
        # only reads the file's facts, so it is instant and never blocks anything.
        platform_tip = textwrap.fill(PLATFORM_HELP, width=55)
        self.platforms_button = QPushButton(PLATFORMS_TEXT)
        self.platforms_button.setToolTip(platform_tip)
        self.platforms_button.clicked.connect(self.on_platforms_clicked)

        # Finding the shots (cuts) of the segment is another step the user starts
        shots_tip = textwrap.fill(SHOTS_HELP, width=55)
        self.shots_button = QPushButton(SHOTS_TEXT)
        self.shots_button.setToolTip(shots_tip)
        self.shots_button.clicked.connect(self.on_shots_clicked)
        self.shots_title = QLabel("Shots")
        self.shots_title.setStyleSheet("font-weight: bold;")
        self.shots_title.setToolTip(shots_tip)
        self.shot_list = ShotListView()
        self.shot_list.setToolTip(shots_tip)
        self.shot_list.seek_requested.connect(self.frame_spin.setValue)

        # The segment as a strip of color
        self.barcode_view = BarcodeView()
        self.barcode_view.setToolTip(textwrap.fill(BARCODE_HELP, width=55))
        self.barcode_view.seek_requested.connect(self.frame_spin.setValue)

        # The same stripes as gray: how bright each moment is
        self.brightness_view = BarcodeView()
        self.brightness_view.setToolTip(textwrap.fill(BRIGHTNESS_BARCODE_HELP, width=55))
        self.brightness_view.seek_requested.connect(self.frame_spin.setValue)

        self.frame_info = QLabel("")

        # Optional exposure warnings painted on the preview
        self.exposure_check = QCheckBox("Show exposure warnings")
        self.exposure_check.setToolTip(textwrap.fill(EXPOSURE_HELP, width=55))
        self.exposure_check.toggled.connect(self.show_current_picture)
        self.exposure_legend = QLabel(
            f'<span style="color:{bgr_hex(EXPOSURE_CRUSHED_BGR)}">\u25a0</span> crushed blacks'
            f'&nbsp;&nbsp;<span style="color:{bgr_hex(EXPOSURE_BLOWN_BGR)}">\u25a0</span> blown highlights')
        self.exposure_legend.setToolTip(textwrap.fill(EXPOSURE_HELP, width=55))
        self.exposure_legend.hide()
        self.normal_picture = None       # the preview as it looks
        self.warning_picture = None      # the same with warnings painted on

        # --- Right, under the preview: measurements for the current frame ---
        self.analysis_title = QLabel("Frame analysis")
        self.analysis_title.setStyleSheet("font-weight: bold;")
        self.analysis_fields = {}
        self.analysis_grid = QGridLayout()
        self.analysis_grid.setHorizontalSpacing(10)

        # A small square showing the dominant hue; it sits right after that
        # row's text, and stays hidden until there is a hue to show
        self.hue_swatch = QLabel()
        self.hue_swatch.setFixedSize(SWATCH_SIZE, SWATCH_SIZE)
        self.hue_swatch.setToolTip(textwrap.fill(ANALYSIS_HELP["Dominant hue"], width=55))
        self.hue_swatch.hide()

        for i, name in enumerate(ANALYSIS_NAMES):
            # Hovering over either the name or the value shows an explanation
            tip = textwrap.fill(ANALYSIS_HELP[name], width=55)
            name_label = QLabel(name + ":")
            name_label.setToolTip(tip)
            value_label = QLabel("-")
            value_label.setToolTip(tip)
            self.analysis_fields[name] = value_label

            # Each column of the panel uses three grid columns: name, value
            # and a gap before the next column
            row, column = i % ANALYSIS_ROWS, i // ANALYSIS_ROWS * 3
            self.analysis_grid.addWidget(name_label, row, column)
            if name == "Dominant hue":
                # The text and the color square share one cell
                self.hue_cell = QWidget()
                cell_layout = QHBoxLayout(self.hue_cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(6)
                cell_layout.addWidget(value_label)
                cell_layout.addWidget(self.hue_swatch)
                cell_layout.addStretch()
                self.analysis_grid.addWidget(self.hue_cell, row, column + 1)
            else:
                self.analysis_grid.addWidget(value_label, row, column + 1)
            self.analysis_grid.setColumnMinimumWidth(column + 2, 28)

        # Spare width goes to the right, so the columns stay together
        last_column = (len(ANALYSIS_NAMES) - 1) // ANALYSIS_ROWS * 3 + 2
        self.analysis_grid.setColumnStretch(last_column, 1)

        # The brightness histogram sits beside the numbers, with its own
        # heading so it lines up with "Frame analysis" on the left
        histogram_tip = textwrap.fill(HISTOGRAM_HELP, width=55)
        waveform_tip = textwrap.fill(WAVEFORM_HELP, width=55)
        self.histogram_view = HistogramView()
        self.histogram_view.setToolTip(histogram_tip)
        self.waveform_view = WaveformView()
        self.waveform_view.setToolTip(waveform_tip)
        palette_tip_text = textwrap.fill(PALETTE_HELP, width=55)
        self.palette_view = PaletteView()
        self.palette_view.set_help(palette_tip_text)
        vectorscope_tip = textwrap.fill(VECTORSCOPE_HELP, width=55)
        self.vectorscope_view = VectorscopeView()
        self.vectorscope_view.setToolTip(vectorscope_tip)

        # One chart area, with a drop-down above it to pick which chart
        self.scope_stack = QStackedWidget()
        self.scope_stack.addWidget(self.histogram_view)
        self.scope_stack.addWidget(self.waveform_view)
        self.scope_stack.addWidget(self.palette_view)
        self.scope_stack.addWidget(self.vectorscope_view)
        self.scope_stack.setFixedHeight(self.histogram_view.height())
        self.scope_selector = QComboBox()
        self.scope_selector.addItems(SCOPE_NAMES)
        self.scope_selector.setStyleSheet("font-weight: bold;")
        self.scope_tips = [histogram_tip, waveform_tip, palette_tip_text,
                           vectorscope_tip]
        for position, tip in enumerate(self.scope_tips):
            self.scope_selector.setItemData(position, tip, Qt.ItemDataRole.ToolTipRole)
        self.scope_selector.setToolTip(histogram_tip)
        self.scope_selector.currentIndexChanged.connect(self.on_scope_changed)

        numbers_column = QVBoxLayout()
        numbers_column.addWidget(self.analysis_title)
        numbers_column.addLayout(self.analysis_grid)
        numbers_column.addStretch()

        histogram_column = QVBoxLayout()
        histogram_column.addWidget(self.scope_selector)
        histogram_column.addWidget(self.scope_stack)
        histogram_column.addStretch()

        analysis_row = QHBoxLayout()
        analysis_row.addLayout(numbers_column)
        analysis_row.addSpacing(16)
        analysis_row.addLayout(histogram_column, 1)

        self.clear_preview()

        # The band sits right under the slider and is as wide as it is
        slider_grid = QGridLayout()
        slider_grid.setVerticalSpacing(2)
        slider_grid.addWidget(self.frame_slider, 0, 0)
        slider_grid.addWidget(self.frame_spin, 0, 1)
        slider_grid.addWidget(self.segment_band, 1, 0)

        analyze_row = QHBoxLayout()
        analyze_row.addWidget(self.analyze_button)
        analyze_row.addWidget(self.shots_button)
        analyze_row.addWidget(self.motion_button)
        analyze_row.addWidget(self.platforms_button)
        analyze_row.addWidget(self.barcode_status, 1)

        preview_column = QVBoxLayout()
        preview_column.addWidget(self.preview)
        preview_column.addLayout(slider_grid)
        preview_column.addWidget(self.segment_picker)
        preview_column.addLayout(analyze_row)
        preview_column.addWidget(self.barcode_view)
        preview_column.addWidget(self.brightness_view)
        preview_column.addWidget(self.motion_view)
        preview_column.addWidget(self.motion_status)
        info_row = QHBoxLayout()
        info_row.addWidget(self.frame_info)
        info_row.addStretch()
        info_row.addWidget(self.exposure_legend)
        info_row.addWidget(self.exposure_check)
        preview_column.addLayout(info_row)
        preview_column.addSpacing(8)
        preview_column.addLayout(analysis_row)
        preview_column.addStretch()

        # The shot list sits on the left, under the video's technical details
        left_column = QVBoxLayout()
        left_column.addLayout(form)
        left_column.addSpacing(12)
        left_column.addWidget(self.shots_title)
        left_column.addWidget(self.shot_list)
        left_column.addStretch()

        body = QHBoxLayout()
        body.addLayout(left_column)
        body.addLayout(preview_column)

        layout = QVBoxLayout()
        layout.addWidget(self.open_button)
        layout.addWidget(self.file_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(body)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # ---------- opening a video and probing it ----------

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video", "",
            "Video files (*.mp4 *.mov *.m4v *.mkv *.avi);;All files (*)",
        )
        if not path:
            return  # the user cancelled

        self.current_path = path
        self.clear_info()
        self.clear_preview()
        self.file_label.setText(f"Reading {path} ...")
        self.open_button.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.probe_thread = ProbeThread(path, self)
        self.probe_thread.progress.connect(self.on_progress)
        self.probe_thread.done.connect(self.on_done)
        self.probe_thread.failed.connect(self.on_failed)
        self.probe_thread.start()

    def on_progress(self, percent):
        if percent < 0:
            self.progress_bar.setRange(0, 0)     # busy animation
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percent)

    def on_done(self, info):
        self.end_probe()
        self.file_label.setText(self.current_path)
        self.show_info(info)
        self.setup_preview(info)

    def on_failed(self, message):
        self.end_probe()
        self.file_label.setText("No video selected")
        QMessageBox.critical(self, "Could not open video", message)

    def end_probe(self):
        self.progress_bar.setVisible(False)
        self.open_button.setEnabled(True)

    def clear_info(self):
        for label in self.fields.values():
            label.setText("-")

    def show_info(self, info):
        f = self.fields
        f["Codec"].setText(info.codec)
        f["Profile"].setText(fmt(info.profile))
        f["Resolution"].setText(info.resolution)
        f["Frame rate"].setText(f"{info.fps:.2f} fps")
        f["Duration"].setText(f"{info.duration:.2f} s")
        f["Frames"].setText(fmt(info.frame_count))
        f["Bitrate"].setText(fmt(None if info.bitrate is None else info.bitrate / 1000,
                                 "{:.0f} kbps"))
        f["Pixel format"].setText(fmt(info.pix_fmt))
        f["Color space"].setText(fmt(info.color_space))
        f["Keyframes"].setText(fmt(info.keyframe_count))
        f["Average GOP"].setText(fmt(info.avg_gop, "{:.1f} frames"))
        f["Audio"].setText("Yes" if info.has_audio else "No")
        f["Audio tracks"].setText(str(info.audio_track_count))

        if not info.has_audio:
            for name in AUDIO_DETAIL_FIELDS:
                f[name].setText("n/a")
            return
        f["Audio codec"].setText(fmt(info.audio_codec))
        f["Sample rate"].setText(
            fmt(None if info.audio_sample_rate is None
                else info.audio_sample_rate / 1000, "{:g} kHz"))
        f["Channels"].setText(channel_label(info.audio_channels))
        f["Audio bitrate"].setText(
            fmt(None if info.audio_bitrate is None
                else info.audio_bitrate / 1000, "{:.0f} kbps"))
        f["Audio length"].setText(audio_length_text(info))

    # ---------- the frame preview ----------

    def clear_preview(self):
        self.info = None
        self.frame_timer.stop()
        self.frame_slider.setEnabled(False)
        self.frame_spin.setEnabled(False)
        self.preview.clear()
        self.preview.setText("Open a video to see its frames")
        self.frame_info.setText("")
        self.normal_picture = None
        self.warning_picture = None
        self.exposure_check.setEnabled(False)
        self.exposure_legend.hide()
        self.stop_barcode()
        self.segment_picker.clear()
        self.segment_band.clear()
        self.barcode_view.clear("Open a video to see its color barcode")
        self.brightness_view.clear("Open a video to see its brightness barcode")
        self.barcode_status.setText("")
        self.stop_shots()
        self.shot_list.clear("Open a video to find its shots")
        self.stop_motion()
        self.motion_view.clear("Open a video to see its motion")
        self.motion_status.setText("")
        self.motion_frames = []
        self.shots_running = False
        self.barcode_running = False
        self.motion_running = False
        self.refresh_controls()
        self.clear_analysis()

    def set_hue_swatch(self, degrees):
        """Show the little color square for this hue, or hide it for None."""
        if degrees is None:
            self.hue_swatch.hide()
            return
        self.hue_swatch.setStyleSheet(
            f"background-color: {hue_color(degrees)}; border: 1px solid #888888;")
        self.hue_swatch.show()

    def on_scope_changed(self, position):
        """Show the chart picked in the drop-down."""
        self.scope_stack.setCurrentIndex(position)
        self.scope_selector.setToolTip(self.scope_tips[position])

    def clear_analysis(self):
        for label in self.analysis_fields.values():
            label.setText("-")
        self.histogram_view.set_histogram(None)
        self.waveform_view.set_waveform(None)
        self.palette_view.set_palette(None)
        self.vectorscope_view.set_vectorscope(None)
        self.set_hue_swatch(None)

    def show_analysis(self, stats):
        if stats is None:
            for label in self.analysis_fields.values():
                label.setText("unavailable")
            self.histogram_view.set_histogram(None)
            self.waveform_view.set_waveform(None)
            self.palette_view.set_palette(None)
            self.vectorscope_view.set_vectorscope(None)
            self.set_hue_swatch(None)
            return
        self.analysis_fields["Brightness"].setText(f"{stats.brightness:.1f} %")
        self.analysis_fields["Contrast"].setText(f"{stats.contrast:.1f} %")
        self.analysis_fields["Crushed blacks"].setText(f"{stats.crushed_blacks:.1f} %")
        self.analysis_fields["Blown highlights"].setText(f"{stats.blown_highlights:.1f} %")
        self.analysis_fields["Dark areas"].setText(f"{stats.dark_share:.1f} %")
        self.analysis_fields["Mid-tones"].setText(f"{stats.mid_share:.1f} %")
        self.analysis_fields["Bright areas"].setText(f"{stats.bright_share:.1f} %")
        self.analysis_fields["Saturation"].setText(f"{stats.saturation:.1f} %")
        self.analysis_fields["Dominant hue"].setText(hue_text(stats.dominant_hue))
        self.set_hue_swatch(stats.dominant_hue)
        self.analysis_fields["Black bars"].setText(bars_text(stats.black_bars))
        self.histogram_view.set_histogram(stats.histogram)
        self.waveform_view.set_waveform(stats.waveform)
        self.palette_view.set_palette(stats.palette)
        self.vectorscope_view.set_vectorscope(stats.vectorscope)

    def setup_preview(self, info):
        last = last_frame_index(info)
        if last is None:
            self.preview.setText("Frame preview is not available for this video")
            return

        self.info = info
        self.wanted_index = 0
        self.last_requested = -1
        # Set the range and go back to frame 0 without triggering a load
        for widget in (self.frame_slider, self.frame_spin):
            widget.blockSignals(True)
            widget.setRange(0, last)
            widget.setValue(0)
            widget.blockSignals(False)
        self.frame_slider.setEnabled(True)
        self.frame_spin.setEnabled(True)
        self.exposure_check.setEnabled(True)
        self.update_frame_label(0)
        self.preview.setText("Loading frame...")
        self.request_frame()

        # The whole video is selected to begin with; nothing is analyzed
        # until the user presses the button
        self.segment_picker.set_video(last + 1, self.info.fps)
        self.segment_band.set_total(last + 1)
        self.barcode_view.clear("Choose a segment, then press Analyze")
        self.brightness_view.clear("The brightness barcode appears with the color barcode")
        self.barcode_status.setText("")
        self.shot_list.clear("Press Detect shots to find the cuts")
        self.motion_view.clear("Press Analyze motion to measure it")
        self.motion_status.setText("")
        self.motion_frames = []
        self.refresh_controls()

    # ---------- choosing a segment and analyzing its color barcode ----------

    def use_current_as_start(self):
        if self.info is not None:
            self.segment_picker.set_first(self.wanted_index)

    def use_current_as_end(self):
        if self.info is not None:
            self.segment_picker.set_last(self.wanted_index)

    def on_analyze_clicked(self):
        """The one button: start the analysis, or cancel it while it runs."""
        if self.barcode_running:
            self.cancel_barcode()
        elif self.info is not None:
            first, last = self.segment_picker.segment()
            self.start_barcode(first, last)

    def start_barcode(self, first, last):
        """Start measuring the barcode of frames first..last in the background."""
        self.stop_barcode()
        total = barcode_columns(first, last)
        self.barcode_view.begin(first, last, total, "Building the color barcode...")
        self.barcode_view.set_position(self.wanted_index)
        self.brightness_view.begin(first, last, total, "Building the brightness barcode...")
        self.brightness_view.set_position(self.wanted_index)
        self.barcode_running = True
        self.refresh_controls()                     # the segment stays put meanwhile
        self.barcode_status.setText(f"Analyzing frames {first} to {last}...")
        thread = BarcodeThread(self.barcode_generation, self.current_path,
                               first, last, self)
        thread.updated.connect(self.on_barcode_updated)
        thread.done.connect(self.on_barcode_done)
        thread.failed.connect(self.on_barcode_failed)
        self.barcode_thread = thread
        # Low priority, so the slider and the preview stay quick meanwhile
        thread.start(QThread.Priority.LowPriority)

    def stop_barcode(self):
        """Stop any barcode being measured. Its late results are ignored."""
        self.barcode_generation += 1
        if self.barcode_thread is not None and self.barcode_thread.isRunning():
            self.barcode_thread.requestInterruption()

    def cancel_barcode(self):
        """The user pressed Cancel: drop the half-finished barcode."""
        self.stop_barcode()
        self.end_barcode_run()
        self.barcode_view.clear("Cancelled. Press Analyze to build it.")
        self.brightness_view.clear("Cancelled. Press Analyze to build it.")
        self.barcode_status.setText("Cancelled")

    def end_barcode_run(self):
        """Go back to the state where a new analysis can be started."""
        self.barcode_running = False
        self.refresh_controls()

    def refresh_controls(self):
        """Switch the segment boxes and the three buttons on or off. Only one
        analysis runs at a time, because all of them read the video."""
        is_open = self.info is not None
        running = {"barcode": self.barcode_running, "shots": self.shots_running,
                   "motion": self.motion_running}

        def free_for(name):
            return is_open and not any(on for key, on in running.items() if key != name)

        self.segment_picker.setEnabled(is_open and not any(running.values()))
        self.analyze_button.setEnabled(free_for("barcode"))
        self.shots_button.setEnabled(free_for("shots"))
        self.motion_button.setEnabled(free_for("motion"))
        self.platforms_button.setEnabled(is_open)
        self.analyze_button.setText(CANCEL_TEXT if self.barcode_running else ANALYZE_TEXT)
        self.shots_button.setText(CANCEL_TEXT if self.shots_running else SHOTS_TEXT)
        self.motion_button.setText(CANCEL_TEXT if self.motion_running else MOTION_TEXT)

    # ---------- how well the video suits the places it will be posted ----------

    def on_platforms_clicked(self):
        """Open the window where the user picks places and sees the verdicts.
        Looking for black bars reads some frames, so it runs in the background
        and the window updates itself when it is done."""
        if self.info is None:
            return
        path = self.current_path
        clip = clip_from(path, self.info)
        known = path in self.bars_found
        if known:
            clip = with_bars(clip, self.bars_found[path])
        dialog = PlatformDialog(clip, self.platform_choices, self,
                                help_text=textwrap.fill(PLATFORM_HELP, width=55),
                                searching=not known)
        thread = None
        frames = last_frame_index(self.info)
        if not known and frames is not None:
            thread = BarsThread(path, frames + 1, self)
            thread.found.connect(lambda found_path, bars:
                                 self.on_bars_found(dialog, clip, found_path, bars))
            thread.start()
        elif not known:
            dialog.search_failed()
        dialog.exec()
        if thread is not None:
            thread.requestInterruption()
            thread.wait()
        self.platform_choices = dialog.selected_keys()     # remembered while the app runs

    def on_bars_found(self, dialog, clip, path, bars):
        self.bars_found[path] = bars
        if bars is None:
            dialog.search_failed()
        else:
            dialog.set_clip(with_bars(clip, bars))

    # ---------- measuring the motion of the segment ----------

    def on_motion_clicked(self):
        """The one button: start measuring motion, or cancel while it runs."""
        if self.motion_running:
            self.cancel_motion()
        elif self.info is not None:
            first, last = self.segment_picker.segment()
            self.start_motion(first, last)

    def start_motion(self, first, last):
        """Start measuring the motion of frames first..last in the background."""
        self.stop_motion()
        if last <= first:
            self.motion_view.clear("Motion needs a segment of at least two frames")
            self.motion_status.setText("")
            return
        total = barcode_columns(first, last)
        self.motion_view.begin(first, last, total, "Measuring the motion...")
        self.motion_view.set_position(self.wanted_index)
        self.motion_status.setText("")
        self.motion_running = True
        self.refresh_controls()
        thread = MotionThread(self.motion_generation, self.current_path,
                              first, last, self.info.fps, self)
        thread.updated.connect(self.on_motion_updated)
        thread.done.connect(self.on_motion_done)
        thread.failed.connect(self.on_motion_failed)
        self.motion_thread = thread
        thread.start(QThread.Priority.LowPriority)

    def stop_motion(self):
        """Stop any motion measurement. Its late results are ignored."""
        self.motion_generation += 1
        if self.motion_thread is not None and self.motion_thread.isRunning():
            self.motion_thread.requestInterruption()

    def cancel_motion(self):
        self.stop_motion()
        self.motion_running = False
        self.refresh_controls()
        self.motion_view.clear("Cancelled. Press Analyze motion to measure it.")
        self.motion_status.setText("")

    def on_motion_updated(self, generation, values):
        if generation == self.motion_generation:
            self.motion_view.set_colors(motion_colors(values))
            self.motion_status.setText(
                f"Measuring the motion: {len(values)} of {self.motion_view.total} stripes")

    def on_motion_done(self, generation, data):
        if generation == self.motion_generation:
            self.motion_running = False
            self.refresh_controls()
            self.motion_frames = list(data.frames)
            self.motion_view.set_colors(motion_colors(data.values, rough=data.rough))
            try:
                summary = summarize_motion(data)
            except ValueError:
                self.motion_status.setText("No motion could be measured in this segment")
                return
            self.motion_status.setText(motion_text(summary))

    def on_motion_failed(self, generation, message):
        if generation == self.motion_generation:
            self.motion_running = False
            self.refresh_controls()
            self.motion_view.set_message("The motion is not available for this segment")
            self.motion_status.setText("")

    # ---------- finding the shots of the segment ----------

    def on_shots_clicked(self):
        """The one button: start finding shots, or cancel while it runs."""
        if self.shots_running:
            self.cancel_shots()
        elif self.info is not None:
            first, last = self.segment_picker.segment()
            self.start_shots(first, last)

    def start_shots(self, first, last):
        """Start finding the shots of frames first..last in the background."""
        self.stop_shots()
        self.segment_band.set_cuts([])
        self.shot_list.clear("Looking for cuts... 0 %")
        self.shots_running = True
        self.refresh_controls()
        thread = ShotThread(self.shots_generation, self.current_path,
                            first, last, self.info.fps, self)
        thread.progress.connect(self.on_shots_progress)
        thread.done.connect(self.on_shots_done)
        thread.failed.connect(self.on_shots_failed)
        self.shots_thread = thread
        thread.start(QThread.Priority.LowPriority)

    def stop_shots(self):
        """Stop any shot detection. Its late results are ignored."""
        self.shots_generation += 1
        if self.shots_thread is not None and self.shots_thread.isRunning():
            self.shots_thread.requestInterruption()

    def cancel_shots(self):
        self.stop_shots()
        self.shots_running = False
        self.refresh_controls()
        self.shot_list.clear("Cancelled. Press Detect shots to try again.")

    def on_shots_progress(self, generation, percent):
        if generation == self.shots_generation:
            self.shot_list.set_message(f"Looking for cuts... {percent} %")

    def on_shots_done(self, generation, result):
        if generation == self.shots_generation:
            self.shots_running = False
            self.refresh_controls()
            self.shot_list.set_shots(result.shots, result.summary,
                                     result.first, result.last)
            self.shot_list.set_position(self.wanted_index)
            self.segment_band.set_cuts(result.cuts)

    def on_shots_failed(self, generation, message):
        if generation == self.shots_generation:
            self.shots_running = False
            self.refresh_controls()
            self.shot_list.clear("The shots could not be found for this segment")

    def on_barcode_updated(self, generation, colors):
        if generation == self.barcode_generation:
            self.barcode_view.set_colors(colors)
            self.brightness_view.set_colors(brightness_colors(colors))
            self.barcode_status.setText(
                f"Analyzing frames {self.barcode_view.first_frame} to "
                f"{self.barcode_view.last_frame}: {len(colors)} of "
                f"{self.barcode_view.total} stripes")

    def on_barcode_done(self, generation, colors):
        if generation == self.barcode_generation:
            self.barcode_view.set_colors(colors)
            self.brightness_view.set_colors(brightness_colors(colors))
            self.end_barcode_run()
            self.barcode_status.setText(
                f"Color barcode of frames {self.barcode_view.first_frame} to "
                f"{self.barcode_view.last_frame}")

    def on_barcode_failed(self, generation, message):
        if generation == self.barcode_generation:
            self.end_barcode_run()
            self.barcode_view.set_message("The color barcode is not available for this segment")
            self.brightness_view.set_message("The brightness barcode is not available for this segment")
            self.barcode_status.setText("The analysis failed")

    def update_frame_label(self, index):
        seconds = index / self.info.fps if self.info.fps > 0 else 0.0
        self.frame_info.setText(
            f"Frame {index} of {self.frame_slider.maximum()}   |   {seconds:.2f} s"
        )

    def on_frame_changed(self, index):
        if self.info is None:
            return
        self.wanted_index = index
        self.barcode_view.set_position(index)
        self.brightness_view.set_position(index)
        self.motion_view.set_position(index)
        self.segment_band.set_position(index)
        self.shot_list.set_position(index)
        self.update_frame_label(index)       # update the text right away
        self.frame_timer.start()             # load the image once it settles

    def request_frame(self):
        if self.info is None:
            return
        if self.frame_thread is not None and self.frame_thread.isRunning():
            return   # on_frame_thread_finished will load the latest request
        self.start_frame_thread()

    def start_frame_thread(self):
        self.last_requested = self.wanted_index
        thread = FrameThread(self.current_path, self.wanted_index, self)
        thread.loaded.connect(self.on_frame_loaded)
        thread.failed.connect(self.on_frame_failed)
        thread.finished.connect(self.on_frame_thread_finished)
        self.frame_thread = thread
        thread.start()

    def on_frame_thread_finished(self):
        # If the user moved on while that frame was loading, load the new one
        if self.info is not None and self.wanted_index != self.last_requested:
            self.start_frame_thread()

    def on_frame_loaded(self, path, index, image, stats, overlay=None):
        if path != self.current_path:
            return   # a late result for a video that is no longer open
        self.normal_picture = QPixmap.fromImage(image)
        self.warning_picture = None if overlay is None else QPixmap.fromImage(overlay)
        self.show_current_picture()
        self.show_analysis(stats)

    def show_current_picture(self):
        """Show the preview, with exposure warnings if the box is ticked."""
        warnings_on = self.exposure_check.isChecked()
        self.exposure_legend.setVisible(warnings_on and self.normal_picture is not None)
        if self.normal_picture is None:
            return
        use_warnings = warnings_on and self.warning_picture is not None
        self.preview.setPixmap(self.warning_picture if use_warnings
                               else self.normal_picture)

    def on_frame_failed(self, path, index, message):
        if path != self.current_path:
            return
        self.clear_analysis()
        self.normal_picture = None
        self.warning_picture = None
        self.exposure_legend.hide()
        self.preview.clear()
        self.preview.setText(f"Could not read frame {index}.\n{message}")

    def closeEvent(self, event):
        # Don't close the window while a background thread is still working
        self.stop_barcode()
        self.stop_shots()
        self.stop_motion()
        for thread in self.findChildren(QThread):
            if thread.isRunning():
                thread.wait()
        event.accept()