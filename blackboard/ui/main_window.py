import textwrap

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QSlider, QSpinBox,
)

from blackboard.analyze import analyze_frame
from blackboard.decode import read_frame, DecodeError
from blackboard.explanations import ANALYSIS_HELP, FIELD_HELP
from blackboard.probe import probe, ProbeError

FIELD_NAMES = ["Codec", "Profile", "Resolution", "Frame rate", "Duration",
               "Frames", "Bitrate", "Pixel format", "Color space",
               "Keyframes", "Average GOP", "Audio", "Audio tracks",
               "Audio codec", "Sample rate", "Channels", "Audio bitrate",
               "Audio length"]

# Rows that describe the audio track itself; they say "n/a" if there is none
AUDIO_DETAIL_FIELDS = ["Audio codec", "Sample rate", "Channels",
                       "Audio bitrate", "Audio length"]

# Rows of the "Frame analysis" panel under the preview
ANALYSIS_NAMES = ["Brightness", "Contrast"]

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


def last_frame_index(info):
    """Number of the video's last frame, or None if we can't tell."""
    if info.frame_count:
        return info.frame_count - 1
    if info.duration > 0 and info.fps > 0:
        return max(int(round(info.duration * info.fps)) - 1, 0)
    return None


def frame_to_qimage(image):
    """Shrink an OpenCV frame (a B, G, R array) to fit the preview box
    and turn it into a QImage."""
    h, w = image.shape[:2]
    scale = min(PREVIEW_W / w, PREVIEW_H / h, 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
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
    loaded = Signal(str, int, object, object)
    failed = Signal(str, int, str)      # video path, frame number, message

    def __init__(self, path, index, parent=None):
        super().__init__(parent)
        self.path = path
        self.index = index

    def run(self):
        try:
            frame = read_frame(self.path, self.index)
            image = frame_to_qimage(frame.image)
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
        self.loaded.emit(self.path, self.index, image, stats)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Black Board")
        self.resize(1000, 660)

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

        # --- Right, under the preview: measurements for the current frame ---
        self.analysis_title = QLabel("Frame analysis")
        self.analysis_title.setStyleSheet("font-weight: bold;")
        self.analysis_fields = {}
        analysis_form = QFormLayout()
        for name in ANALYSIS_NAMES:
            # Hovering over either the name or the value shows an explanation
            tip = textwrap.fill(ANALYSIS_HELP[name], width=55)
            name_label = QLabel(name + ":")
            name_label.setToolTip(tip)
            value_label = QLabel("-")
            value_label.setToolTip(tip)
            self.analysis_fields[name] = value_label
            analysis_form.addRow(name_label, value_label)

        self.clear_preview()

        slider_row = QHBoxLayout()
        slider_row.addWidget(self.frame_slider)
        slider_row.addWidget(self.frame_spin)

        preview_column = QVBoxLayout()
        preview_column.addWidget(self.preview)
        preview_column.addLayout(slider_row)
        preview_column.addWidget(self.frame_info)
        preview_column.addSpacing(8)
        preview_column.addWidget(self.analysis_title)
        preview_column.addLayout(analysis_form)
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
        self.clear_analysis()

    def clear_analysis(self):
        for label in self.analysis_fields.values():
            label.setText("-")

    def show_analysis(self, stats):
        if stats is None:
            for label in self.analysis_fields.values():
                label.setText("unavailable")
            return
        self.analysis_fields["Brightness"].setText(f"{stats.brightness:.1f} %")
        self.analysis_fields["Contrast"].setText(f"{stats.contrast:.1f} %")

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

    def on_frame_loaded(self, path, index, image, stats):
        if path != self.current_path:
            return   # a late result for a video that is no longer open
        self.preview.setPixmap(QPixmap.fromImage(image))
        self.show_analysis(stats)

    def on_frame_failed(self, path, index, message):
        if path != self.current_path:
            return
        self.clear_analysis()
        self.preview.clear()
        self.preview.setText(f"Could not read frame {index}.\n{message}")

    def closeEvent(self, event):
        # Don't close the window while a background thread is still working
        for thread in self.findChildren(QThread):
            if thread.isRunning():
                thread.wait()
        event.accept()