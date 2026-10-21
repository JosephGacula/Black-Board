from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QFormLayout, QSizePolicy,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
)

from blackboard.probe import probe, ProbeError

FIELD_NAMES = ["Codec", "Profile", "Resolution", "Frame rate", "Duration",
               "Frames", "Bitrate", "Pixel format", "Color space",
               "Keyframes", "Average GOP", "Audio"]


def fmt(value, template="{}"):
    """Show 'unknown' for missing values instead of 'None'."""
    return "unknown" if value is None else template.format(value)


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Black Board")
        self.resize(480, 500)

        self.probe_thread = None
        self.current_path = ""

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

        # One label per measurement, filled in after a video is probed
        self.fields = {}
        form = QFormLayout()
        for name in FIELD_NAMES:
            label = QLabel("-")
            self.fields[name] = label
            form.addRow(name + ":", label)

        layout = QVBoxLayout()
        layout.addWidget(self.open_button)
        layout.addWidget(self.file_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(form)
        layout.addStretch()

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video", "",
            "Video files (*.mp4 *.mov *.m4v *.mkv *.avi);;All files (*)",
        )
        if not path:
            return  # the user cancelled

        self.current_path = path
        self.clear_info()
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

    def closeEvent(self, event):
        # Don't close the window while the background thread is still working
        if self.probe_thread is not None and self.probe_thread.isRunning():
            self.probe_thread.wait()
        event.accept()