"""
Main application window — Neural Interface layout.
"""

import sys
import subprocess
import re
import glob as globmod

import cv2
from PyQt6.QtCore    import Qt
from PyQt6.QtGui     import QColor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QComboBox, QProgressBar, QMessageBox, QFrame,
    QSizePolicy,
)

from config            import ConfigManager
from tracking.devices  import VirtualDeviceManager
from tracking.calibration import CalibrationThread
from tracking.thread   import TrackingThread
from ui.camera_widget  import CameraPreview
from ui.settings_panel import SettingsPanel
from ui.styles         import STYLESHEET
from ui.effects        import PulsingOrb, make_glow_effect
from ui.intro_widget   import IntroWidget

# ── App states ────────────────────────────────────────────────────
IDLE        = "idle"
CALIBRATING = "calibrating"
READY       = "ready"
TRACKING    = "tracking"

_STATE_UI = {
    IDLE:        ("IDLE",        "#404868"),
    CALIBRATING: ("CALIBRATING", "#ffaa00"),
    READY:       ("STANDBY",     "#7b2fff"),
    TRACKING:    ("TRACKING",    "#00d4ff"),
}

# ── Camera enumeration ────────────────────────────────────────────

def enumerate_cameras():
    if sys.platform == 'win32':
        return _enumerate_cameras_windows()
    return _enumerate_cameras_linux()


def _enumerate_cameras_linux():
    cameras = []
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines        = result.stdout.strip().split('\n')
            current_name = ""
            for line in lines:
                if not line.startswith('\t'):
                    current_name = line.strip().rstrip(':')
                    clean = re.sub(r'\s*\(.*\)\s*$', '', current_name)
                    if clean:
                        current_name = clean
                else:
                    device = line.strip()
                    m = re.match(r'/dev/video(\d+)', device)
                    if m:
                        cameras.append({
                            "index": int(m.group(1)),
                            "name":  f"{current_name} ({device})",
                        })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not cameras:
        for device_path in sorted(globmod.glob('/dev/video*')):
            m = re.match(r'/dev/video(\d+)', device_path)
            if m:
                idx = int(m.group(1))
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cameras.append({"index": idx, "name": f"Camera {idx} ({device_path})"})
                    cap.release()
    return cameras


def _enumerate_cameras_windows():
    cameras = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            name = cap.getBackendName()
            cameras.append({
                "index": idx,
                "name":  f"Camera {idx}" if not name else f"{name} ({idx})",
            })
            cap.release()
    return cameras


# ── Main window ───────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config_manager    = config_manager
        self.state             = IDLE
        self.center_x          = 0.0
        self.center_y          = 0.0
        self.calibration_thread = None
        self.tracking_thread   = None

        self.device_manager = VirtualDeviceManager()
        try:
            self.device_manager.create_devices()
        except Exception as ex:
            QMessageBox.critical(
                self, "Permission Error",
                f"Cannot create virtual input devices:\n{ex}\n\n"
                "Make sure you're in the 'input' group:\n"
                "sudo usermod -a -G input $USER",
            )

        self.setWindowTitle("Gesture.AI — Neural Interface")
        self.setMinimumSize(1000, 650)
        self.resize(1220, 760)
        self.setStyleSheet(STYLESHEET)

        self._build_ui()
        self._update_button_states()

        # Intro animation (overlay on top of central widget)
        self._intro = IntroWidget(self.centralWidget())
        self._intro.resize(self.centralWidget().size())
        self._intro.finished.connect(self._on_intro_done)
        self._intro.start()

    # ── resize ────────────────────────────────────────────────────

    def resizeEvent(self, event):  # type: ignore
        super().resizeEvent(event)
        if hasattr(self, '_intro') and self._intro.isVisible():
            self._intro.resize(self.centralWidget().size())

    def _on_intro_done(self):
        pass  # UI already visible underneath

    # ── build UI ──────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header bar ────────────────────────────────────────────
        header = QWidget()
        header.setObjectName("header_widget")
        header.setFixedHeight(56)

        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 0, 20, 0)
        hl.setSpacing(12)

        logo_col = QVBoxLayout()
        logo_col.setSpacing(1)

        app_name = QLabel("◉  GESTURE.AI")
        app_name.setObjectName("app_name")
        logo_col.addWidget(app_name)

        app_sub = QLabel("NEURAL INTERFACE CONTROL SYSTEM")
        app_sub.setObjectName("app_sub")
        logo_col.addWidget(app_sub)

        hl.addLayout(logo_col)
        hl.addStretch()

        cam_label = QLabel("CAM")
        cam_label.setObjectName("cam_label")
        hl.addWidget(cam_label)

        self.camera_combo = QComboBox()
        self._populate_cameras()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        hl.addWidget(self.camera_combo)

        root.addWidget(header)

        # ── Body ──────────────────────────────────────────────────
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Camera preview (left, expanding)
        self.camera_preview = CameraPreview()
        self.camera_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        body.addWidget(self.camera_preview, stretch=1)

        # Right panel
        right = QWidget()
        right.setObjectName("right_panel")
        right.setFixedWidth(315)

        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 12, 10, 10)
        rl.setSpacing(7)

        # ── State card ────────────────────────────────────────────
        state_card = QFrame()
        state_card.setObjectName("state_card")
        scl = QVBoxLayout(state_card)
        scl.setContentsMargins(14, 10, 14, 10)
        scl.setSpacing(4)

        s_lbl = QLabel("SYSTEM")
        s_lbl.setObjectName("card_label")
        scl.addWidget(s_lbl)

        self.state_value = QLabel("IDLE")
        self.state_value.setObjectName("state_value")
        scl.addWidget(self.state_value)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 30)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        scl.addWidget(self.progress_bar)

        rl.addWidget(state_card)

        # ── Detection card ────────────────────────────────────────
        detect_card = QFrame()
        detect_card.setObjectName("detect_card")
        dcl = QVBoxLayout(detect_card)
        dcl.setContentsMargins(14, 10, 14, 10)
        dcl.setSpacing(7)

        d_lbl = QLabel("DETECTION")
        d_lbl.setObjectName("card_label")
        dcl.addWidget(d_lbl)

        self.face_orb  = PulsingOrb(size=8)
        self.face_val  = QLabel("--")
        dcl.addLayout(self._status_row(self.face_orb,  "FACE",    self.face_val))

        self.hands_orb = PulsingOrb(size=8)
        self.hands_val = QLabel("--")
        dcl.addLayout(self._status_row(self.hands_orb, "HANDS",   self.hands_val))

        self.gest_orb  = PulsingOrb(size=8)
        self.gest_val  = QLabel("--")
        dcl.addLayout(self._status_row(self.gest_orb,  "GESTURE", self.gest_val))

        for lbl in (self.face_val, self.hands_val, self.gest_val):
            lbl.setObjectName("status_value")

        rl.addWidget(detect_card)

        # ── Controls card ─────────────────────────────────────────
        ctrl_card = QFrame()
        ctrl_card.setObjectName("controls_card")
        ccl = QVBoxLayout(ctrl_card)
        ccl.setContentsMargins(14, 10, 14, 10)
        ccl.setSpacing(6)

        ctrl_lbl = QLabel("CONTROLS")
        ctrl_lbl.setObjectName("card_label")
        ccl.addWidget(ctrl_lbl)

        self.btn_calibrate = QPushButton("CALIBRATE")
        self.btn_calibrate.setObjectName("btn_calibrate")
        self.btn_calibrate.clicked.connect(self._on_calibrate)
        self.btn_calibrate.setGraphicsEffect(make_glow_effect("#7b2fff", 18))
        ccl.addWidget(self.btn_calibrate)

        self.btn_start = QPushButton("START TRACKING")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_start.setGraphicsEffect(make_glow_effect("#00ff88", 18))
        ccl.addWidget(self.btn_start)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setGraphicsEffect(make_glow_effect("#ff2255", 18))
        ccl.addWidget(self.btn_stop)

        rl.addWidget(ctrl_card)

        # ── Settings panel ────────────────────────────────────────
        self.settings_panel = SettingsPanel(self.config_manager)
        self.settings_panel.settings_changed.connect(self._on_settings_changed)
        rl.addWidget(self.settings_panel, stretch=1)

        body.addWidget(right)
        root.addLayout(body, stretch=1)

    # ── helper ────────────────────────────────────────────────────

    @staticmethod
    def _status_row(orb: PulsingOrb, name: str, value_lbl: QLabel) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(7)
        row.addWidget(orb)
        lbl = QLabel(name)
        lbl.setObjectName("status_name")
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(value_lbl)
        return row

    # ── camera enumeration ────────────────────────────────────────

    def _populate_cameras(self):
        self.camera_combo.clear()
        self._cameras = enumerate_cameras()
        if not self._cameras:
            self.camera_combo.addItem("No cameras found", -1)
        else:
            for cam in self._cameras:
                self.camera_combo.addItem(cam["name"], cam["index"])

    def _get_camera_index(self) -> int:
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

    # ── calibration ───────────────────────────────────────────────

    def _on_calibrate(self):
        if self.state == TRACKING:
            self._on_stop()

        camera_index = self._get_camera_index()
        self.state   = CALIBRATING
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
        self.camera_preview.set_calibration_progress(current, total)

    def _on_calibration_done(self, center_x, center_y):
        self.center_x = center_x
        self.center_y = center_y
        self.state    = READY
        self.progress_bar.setVisible(False)
        self._update_button_states()
        self.face_val.setText("Calibrated")
        self.face_orb.set_mode('active')

    def _on_calibration_failed(self, msg):
        self.state = IDLE
        self.progress_bar.setVisible(False)
        self._update_button_states()
        QMessageBox.warning(self, "Calibration Failed", msg)

    # ── tracking ──────────────────────────────────────────────────

    def _on_start(self):
        if self.state != READY:
            return

        camera_index = self._get_camera_index()
        config       = self.config_manager.get_all()

        self.tracking_thread = TrackingThread(
            camera_index   = camera_index,
            center_x       = self.center_x,
            center_y       = self.center_y,
            config         = config,
            device_manager = self.device_manager,
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

        self.face_val.setText("--");  self.face_orb.set_mode('off')
        self.hands_val.setText("--"); self.hands_orb.set_mode('off')
        self.gest_val.setText("--");  self.gest_orb.set_mode('off')

    def _on_status_update(self, status: dict):
        face = status.get("face_detected", False)
        self.face_val.setText("Detected" if face else "No Face")
        self.face_val.setStyleSheet(
            "color: #00d4ff;" if face else "color: #ff2255;"
        )
        self.face_orb.set_mode('cyan' if face else 'error')

        hcount  = status.get("hand_count", 0)
        is_fist = status.get("is_fist", False)
        self.hands_val.setText(
            f"{hcount}" + (" Fist" if is_fist else "") if hcount else "0"
        )
        self.hands_val.setStyleSheet(
            "color: #7b2fff;" if hcount else "color: #404868;"
        )
        self.hands_orb.set_mode('purple' if hcount else 'off')

        gesture = status.get("last_gesture") or "--"
        self.gest_val.setText(gesture)
        self.gest_val.setStyleSheet(
            "color: #ff2faa;" if gesture != "--" else "color: #404868;"
        )
        self.gest_orb.set_mode('magenta' if gesture != "--" else 'off')

    def _on_tracking_error(self, msg: str):
        self.state = READY
        self._update_button_states()
        QMessageBox.warning(self, "Tracking Error", msg)

    def _on_settings_changed(self, config: dict):
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.update_config(config)

    # ── button / state sync ───────────────────────────────────────

    def _update_button_states(self):
        self.btn_calibrate.setEnabled(self.state in (IDLE, READY))
        self.btn_start.setEnabled(self.state == READY)
        self.btn_stop.setEnabled(self.state == TRACKING)
        self.camera_combo.setEnabled(self.state not in (CALIBRATING, TRACKING))
        self._refresh_state_display()

    def _refresh_state_display(self):
        text, color = _STATE_UI.get(self.state, ("IDLE", "#404868"))
        self.state_value.setText(text)
        self.state_value.setStyleSheet(
            f"color: {color}; font-size: 20px; font-weight: 700;"
            f" letter-spacing: 3px; font-family: 'Courier New', monospace;"
        )
        self.camera_preview.set_tracking_state(self.state)

    # ── cleanup ───────────────────────────────────────────────────

    def closeEvent(self, event):  # type: ignore
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
            self.tracking_thread.wait(5000)
        if self.calibration_thread and self.calibration_thread.isRunning():
            self.calibration_thread.stop()
            self.calibration_thread.wait(3000)
        self.device_manager.close_devices()
        event.accept()
