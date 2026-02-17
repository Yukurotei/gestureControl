import subprocess
import re
import glob as globmod

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QComboBox, QProgressBar, QMessageBox, QFrame,
    QSizePolicy,
)

from config import ConfigManager
from tracking.devices import VirtualDeviceManager
from tracking.calibration import CalibrationThread
from tracking.thread import TrackingThread
from ui.camera_widget import CameraPreview
from ui.settings_panel import SettingsPanel
from ui.styles import DARK_STYLESHEET

# App states
IDLE = "idle"
CALIBRATING = "calibrating"
READY = "ready"
TRACKING = "tracking"


def enumerate_cameras():
    cameras = []
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            current_name = ""
            for line in lines:
                if not line.startswith('\t'):
                    current_name = line.strip().rstrip(':')
                    # Remove USB path info in parentheses for cleaner display
                    clean_name = re.sub(r'\s*\(.*\)\s*$', '', current_name)
                    if clean_name:
                        current_name = clean_name
                else:
                    device = line.strip()
                    match = re.match(r'/dev/video(\d+)', device)
                    if match:
                        index = int(match.group(1))
                        cameras.append({"index": index, "name": f"{current_name} ({device})"})
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not cameras:
        for device_path in sorted(globmod.glob('/dev/video*')):
            match = re.match(r'/dev/video(\d+)', device_path)
            if match:
                index = int(match.group(1))
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    cameras.append({"index": index, "name": f"Camera {index} ({device_path})"})
                    cap.release()

    return cameras


class MainWindow(QMainWindow):
    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config_manager = config_manager
        self.state = IDLE
        self.center_x = 0.0
        self.center_y = 0.0
        self.calibration_thread = None
        self.tracking_thread = None

        # Create virtual devices
        self.device_manager = VirtualDeviceManager()
        try:
            self.device_manager.create_devices()
        except Exception as ex:
            QMessageBox.critical(
                self, "Permission Error",
                f"Cannot create virtual input devices:\n{ex}\n\n"
                "Make sure you're in the 'input' group:\n"
                "sudo usermod -a -G input $USER"
            )

        self.setWindowTitle("Gesture Control")
        self.setMinimumSize(1000, 650)
        self.resize(1200, 750)
        self.setStyleSheet(DARK_STYLESHEET)

        self._build_ui()
        self._update_button_states()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(12)

        # Header row
        header = QHBoxLayout()
        title = QLabel("Gesture Control")
        title.setObjectName("header-title")
        header.addWidget(title)
        header.addStretch()

        cam_label = QLabel("Camera:")
        cam_label.setObjectName("status-label")
        header.addWidget(cam_label)

        self.camera_combo = QComboBox()
        self._populate_cameras()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        header.addWidget(self.camera_combo)

        main_layout.addLayout(header)

        # Separator
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        main_layout.addWidget(sep)

        # Body: left (camera + controls) | right (settings)
        body = QHBoxLayout()
        body.setSpacing(16)

        # Left panel
        left = QVBoxLayout()
        left.setSpacing(8)

        self.camera_preview = CameraPreview()
        self.camera_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        left.addWidget(self.camera_preview, stretch=1)

        # Progress bar (shown during calibration)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 30)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        left.addWidget(self.progress_bar)

        # Status row
        status_layout = QHBoxLayout()
        status_layout.setSpacing(24)

        self.face_status = QLabel("Face: --")
        self.face_status.setObjectName("status-value")
        status_layout.addWidget(self.face_status)

        self.hand_status = QLabel("Hands: --")
        self.hand_status.setObjectName("status-value")
        status_layout.addWidget(self.hand_status)

        self.gesture_status = QLabel("Gesture: --")
        self.gesture_status.setObjectName("status-value")
        status_layout.addWidget(self.gesture_status)

        status_layout.addStretch()
        left.addLayout(status_layout)

        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.btn_calibrate = QPushButton("Calibrate")
        self.btn_calibrate.setObjectName("btn-calibrate")
        self.btn_calibrate.clicked.connect(self._on_calibrate)
        btn_layout.addWidget(self.btn_calibrate)

        self.btn_start = QPushButton("Start Tracking")
        self.btn_start.setObjectName("btn-start")
        self.btn_start.clicked.connect(self._on_start)
        btn_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btn-stop")
        self.btn_stop.clicked.connect(self._on_stop)
        btn_layout.addWidget(self.btn_stop)

        btn_layout.addStretch()
        left.addLayout(btn_layout)

        body.addLayout(left, stretch=65)

        # Right panel: settings
        self.settings_panel = SettingsPanel(self.config_manager)
        self.settings_panel.settings_changed.connect(self._on_settings_changed)
        self.settings_panel.setMinimumWidth(280)
        self.settings_panel.setMaximumWidth(360)
        body.addWidget(self.settings_panel, stretch=35)

        main_layout.addLayout(body, stretch=1)

    def _populate_cameras(self):
        self.camera_combo.clear()
        self._cameras = enumerate_cameras()
        if not self._cameras:
            self.camera_combo.addItem("No cameras found", -1)
        else:
            for cam in self._cameras:
                self.camera_combo.addItem(cam["name"], cam["index"])

    def _get_camera_index(self):
        data = self.camera_combo.currentData()
        return data if data is not None and data >= 0 else 0

    def _on_camera_changed(self):
        if self.state == TRACKING:
            self._on_stop()
        if self.state in (READY, CALIBRATING):
            if self.calibration_thread and self.calibration_thread.isRunning():
                self.calibration_thread.stop()
                self.calibration_thread.wait(3000)
            self.state = IDLE
            self._update_button_states()
            self.camera_preview.clear_frame()
            self.progress_bar.setVisible(False)

    def _on_calibrate(self):
        if self.state == TRACKING:
            self._on_stop()

        camera_index = self._get_camera_index()
        self.state = CALIBRATING
        self._update_button_states()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.calibration_thread = CalibrationThread(camera_index=camera_index)
        self.calibration_thread.progress.connect(self._on_calibration_progress)
        self.calibration_thread.frame_ready.connect(self.camera_preview.update_frame)
        self.calibration_thread.calibration_done.connect(self._on_calibration_done)
        self.calibration_thread.calibration_failed.connect(self._on_calibration_failed)
        self.calibration_thread.start()

    def _on_calibration_progress(self, current, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)

    def _on_calibration_done(self, center_x, center_y):
        self.center_x = center_x
        self.center_y = center_y
        self.state = READY
        self.progress_bar.setVisible(False)
        self._update_button_states()
        self.face_status.setText("Face: Calibrated")

    def _on_calibration_failed(self, msg):
        self.state = IDLE
        self.progress_bar.setVisible(False)
        self._update_button_states()
        QMessageBox.warning(self, "Calibration Failed", msg)

    def _on_start(self):
        if self.state != READY:
            return

        camera_index = self._get_camera_index()
        config = self.config_manager.get_all()

        self.tracking_thread = TrackingThread(
            camera_index=camera_index,
            center_x=self.center_x,
            center_y=self.center_y,
            config=config,
            device_manager=self.device_manager,
        )
        self.tracking_thread.frame_ready.connect(self.camera_preview.update_frame)
        self.tracking_thread.status_update.connect(self._on_status_update)
        self.tracking_thread.error.connect(self._on_tracking_error)
        self.tracking_thread.start()

        self.state = TRACKING
        self._update_button_states()

    def _on_stop(self):
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
            self.tracking_thread.wait(5000)
            self.tracking_thread = None

        self.state = READY
        self._update_button_states()
        self.camera_preview.clear_frame()
        self.face_status.setText("Face: --")
        self.hand_status.setText("Hands: --")
        self.gesture_status.setText("Gesture: --")

    def _on_status_update(self, status):
        face = "Detected" if status.get("face_detected") else "No face"
        face_color = "#a6e3a1" if status.get("face_detected") else "#f38ba8"
        self.face_status.setText(f"Face: {face}")
        self.face_status.setStyleSheet(f"color: {face_color}; font-weight: bold;")

        hcount = status.get("hand_count", 0)
        fist = " (Fist)" if status.get("is_fist") else ""
        hand_color = "#a6e3a1" if hcount > 0 else "#a6adc8"
        self.hand_status.setText(f"Hands: {hcount}{fist}")
        self.hand_status.setStyleSheet(f"color: {hand_color}; font-weight: bold;")

        gesture = status.get("last_gesture") or "--"
        self.gesture_status.setText(f"Gesture: {gesture}")

    def _on_tracking_error(self, msg):
        self.state = READY
        self._update_button_states()
        QMessageBox.warning(self, "Tracking Error", msg)

    def _on_settings_changed(self, config):
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.update_config(config)

    def _update_button_states(self):
        self.btn_calibrate.setEnabled(self.state in (IDLE, READY))
        self.btn_start.setEnabled(self.state == READY)
        self.btn_stop.setEnabled(self.state == TRACKING)
        self.camera_combo.setEnabled(self.state not in (CALIBRATING, TRACKING))

    def closeEvent(self, event):
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
            self.tracking_thread.wait(5000)
        if self.calibration_thread and self.calibration_thread.isRunning():
            self.calibration_thread.stop()
            self.calibration_thread.wait(3000)
        self.device_manager.close_devices()
        event.accept()
