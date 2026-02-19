"""
Settings panel — grouped, scrollable parameter controls.
"""

from PyQt6.QtCore    import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
    QScrollArea, QFrame,
)

from config import ConfigManager

# ── Setting groups (ordered display) ────────────────────────────
_GROUPS = [
    ("GENERAL",     ["SENSITIVITY_MULTIPLIER", "FPS"]),
    ("LEFT CLICK",  ["THUMB_INDEX_THRESHOLD", "THUMB_INDEX_MIN_THRESHOLD"]),
    ("RIGHT CLICK", [
        "THUMB_PINKIE_THRESHOLD",
        "THUMB_PINKIE_MIN_THRESHOLD",
        "FIST_THUMB_INDEX_RIGHT_CLICK",
    ]),
    ("SNAP / CLOSE", [
        "THUMB_MIDDLE_THRESHOLD",
        "THUMB_MIDDLE_MIN_THRESHOLD",
        "SNAP_TIME_WINDOW_SECONDS",
        "SNAP_DISTANCE_THRESHOLD",
        "SNAP_MODE",
    ]),
    ("SCROLL",      ["SCROLL_SENSITIVITY", "FIST_CURLED_FINGERS_AMOUNT"]),
    ("SWIPE",       ["SWIPE_VELOCITY_THRESHOLD"]),
]


class SettingsPanel(QWidget):
    settings_changed = pyqtSignal(dict)

    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self._widgets: dict = {}

        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._do_save)

        self._build_ui()

    # ── layout ───────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout  = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 8)
        layout.setSpacing(5)

        # Section heading
        sec = QLabel("PARAMETERS")
        sec.setObjectName("card_label")
        layout.addWidget(sec)

        for group_name, keys in _GROUPS:
            group_frame = QFrame()
            group_frame.setObjectName("settings_group")
            gl = QVBoxLayout(group_frame)
            gl.setContentsMargins(10, 8, 10, 8)
            gl.setSpacing(5)

            g_lbl = QLabel(group_name)
            g_lbl.setObjectName("group_title")
            gl.addWidget(g_lbl)

            for key in keys:
                if key not in ConfigManager.DEFAULTS:
                    continue

                default      = ConfigManager.DEFAULTS[key]
                desc         = ConfigManager.DESCRIPTIONS.get(key, "")
                display_name = ConfigManager.DISPLAY_NAMES.get(key, key)
                current      = self.config_manager.get(key)

                row = QHBoxLayout()
                row.setSpacing(6)

                name_lbl = QLabel(display_name)
                name_lbl.setObjectName("setting_name")
                name_lbl.setWordWrap(True)
                row.addWidget(name_lbl, stretch=1)

                if isinstance(default, bool):
                    widget = QCheckBox()
                    widget.setChecked(bool(current))
                    widget.toggled.connect(
                        lambda checked, k=key: self._on_value_changed(k, checked)
                    )

                elif isinstance(default, float):
                    widget = QDoubleSpinBox()
                    if key in ConfigManager.VALUE_RANGES:
                        mn, mx, step = ConfigManager.VALUE_RANGES[key]
                        widget.setRange(mn, mx)
                        widget.setSingleStep(step)
                        widget.setDecimals(3)
                    widget.setValue(float(current))
                    widget.valueChanged.connect(
                        lambda val, k=key: self._on_value_changed(k, val)
                    )

                elif isinstance(default, int):
                    widget = QSpinBox()
                    if key in ConfigManager.VALUE_RANGES:
                        mn, mx, step = ConfigManager.VALUE_RANGES[key]
                        widget.setRange(int(mn), int(mx))
                        widget.setSingleStep(int(step))
                    widget.setValue(int(current))
                    widget.valueChanged.connect(
                        lambda val, k=key: self._on_value_changed(k, val)
                    )

                elif isinstance(default, str) and key in ConfigManager.VALUE_OPTIONS:
                    widget = QComboBox()
                    for opt in ConfigManager.VALUE_OPTIONS[key]:
                        widget.addItem(opt)
                    widget.setCurrentText(str(current))
                    widget.currentTextChanged.connect(
                        lambda val, k=key: self._on_value_changed(k, val)
                    )

                else:
                    continue

                row.addWidget(widget)
                self._widgets[key] = widget
                gl.addLayout(row)

                if desc:
                    desc_lbl = QLabel(desc)
                    desc_lbl.setObjectName("setting_desc")
                    desc_lbl.setWordWrap(True)
                    gl.addWidget(desc_lbl)

            layout.addWidget(group_frame)

        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    # ── change handling ──────────────────────────────────────────

    def _on_value_changed(self, key, value):
        self.config_manager.set(key, value)
        self._save_timer.start()

    def _do_save(self):
        self.config_manager.save()
        self.settings_changed.emit(self.config_manager.get_all())
