from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QFormLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox,
)

from blackboard.probe import probe, ProbeError


def fmt(value, template="{}"):
    """Show 'unknown' for missing values instead of 'None'."""
    return "unknown" if value is None else template.format(value)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Black Board")
        self.resize(480, 460)

        self.open_button = QPushButton("Open video...")
        self.open_button.clicked.connect(self.choose_file)

        self.file_label = QLabel("No video selected")
        self.file_label.setWordWrap(True)

        # One label per measurement, filled in after a video is probed
        self.fields = {}
        form = QFormLayout()
        for name in ["Codec", "Profile", "Resolution", "Frame rate", "Duration",
                     "Frames", "Bitrate", "Pixel format", "Color space",
                     "Keyframes", "Average GOP", "Audio"]:
            label = QLabel("-")
            self.fields[name] = label
            form.addRow(name + ":", label)

        layout = QVBoxLayout()
        layout.addWidget(self.open_button)
        layout.addWidget(self.file_label)
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

        try:
            info = probe(path)
        except ProbeError as e:
            QMessageBox.critical(self, "Could not open video", str(e))
            return

        self.file_label.setText(path)
        self.show_info(info)

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