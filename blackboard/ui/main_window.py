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
from blackboard.decode import read_frame, DecodeError
from blackboard.explanations import (
    ANALYSIS_HELP, EXPOSURE_HELP, FIELD_HELP, HISTOGRAM_HELP, PALETTE_HELP,
    VECTORSCOPE_HELP, WAVEFORM_HELP,
)
from blackboard.probe import probe, ProbeError
from blackboard.ui.histogram_view import HistogramView
from blackboard.ui.palette_view import PaletteView
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Black Board")
        self.resize(1000, 700)

        self.probe_thread = None
        self.frame_thread = None
        self.current_path = ""
        self.info = None          # the VideoInfo of the open video
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

        slider_row = QHBoxLayout()
        slider_row.addWidget(self.frame_slider)
        slider_row.addWidget(self.frame_spin)

        preview_column = QVBoxLayout()
        preview_column.addWidget(self.preview)
        preview_column.addLayout(slider_row)
        info_row = QHBoxLayout()
        info_row.addWidget(self.frame_info)
        info_row.addStretch()
        info_row.addWidget(self.exposure_legend)
        info_row.addWidget(self.exposure_check)
        preview_column.addLayout(info_row)
        preview_column.addSpacing(8)
        preview_column.addLayout(analysis_row)
        preview_column.addStretch()

        body = QHBoxLayout()
        body.addLayout(form)
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

    def update_frame_label(self, index):
        seconds = index / self.info.fps if self.info.fps > 0 else 0.0
        self.frame_info.setText(
            f"Frame {index} of {self.frame_slider.maximum()}   |   {seconds:.2f} s"
        )

    def on_frame_changed(self, index):
        if self.info is None:
            return
        self.wanted_index = index
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
        for thread in self.findChildren(QThread):
            if thread.isRunning():
                thread.wait()
        event.accept()