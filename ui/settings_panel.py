from PyQt6.QtCore import pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QCheckBox, QScrollArea, QFrame,
)

from config import ConfigManager


class SettingsPanel(QWidget):
    settings_changed = pyqtSignal(dict)

    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self._widgets = {}
        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._do_save)

        self._build_ui()

    def _build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Settings")
        title.setObjectName("section-title")
        outer_layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            scroll.horizontalScrollBarPolicy()  # keep default
        )

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 4, 8, 4)
        layout.setSpacing(2)

        order = list(ConfigManager.DEFAULTS.keys())

        for key in order:
            default = ConfigManager.DEFAULTS[key]
            desc = ConfigManager.DESCRIPTIONS.get(key, "")
            display_name = ConfigManager.DISPLAY_NAMES.get(key, key)
            current = self.config_manager.get(key)

            name_label = QLabel(display_name)
            name_label.setObjectName("setting-name")
            layout.addWidget(name_label)

            if isinstance(default, bool):
                widget = QCheckBox()
                widget.setChecked(bool(current))
                widget.toggled.connect(lambda checked, k=key: self._on_value_changed(k, checked))
            elif isinstance(default, float):
                widget = QDoubleSpinBox()
                if key in ConfigManager.VALUE_RANGES:
                    mn, mx, step = ConfigManager.VALUE_RANGES[key]
                    widget.setRange(mn, mx)
                    widget.setSingleStep(step)
                    widget.setDecimals(2)
                widget.setValue(float(current))
                widget.valueChanged.connect(lambda val, k=key: self._on_value_changed(k, val))
            elif isinstance(default, int):
                widget = QSpinBox()
                if key in ConfigManager.VALUE_RANGES:
                    mn, mx, step = ConfigManager.VALUE_RANGES[key]
                    widget.setRange(int(mn), int(mx))
                    widget.setSingleStep(int(step))
                widget.setValue(int(current))
                widget.valueChanged.connect(lambda val, k=key: self._on_value_changed(k, val))
            else:
                continue

            layout.addWidget(widget)
            self._widgets[key] = widget

            desc_label = QLabel(desc)
            desc_label.setObjectName("setting-desc")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)

            sep = QFrame()
            sep.setObjectName("separator")
            sep.setFrameShape(QFrame.Shape.HLine)
            layout.addWidget(sep)

        layout.addStretch()
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

    def _on_value_changed(self, key, value):
        self.config_manager.set(key, value)
        self._save_timer.start()

    def _do_save(self):
        self.config_manager.save()
        self.settings_changed.emit(self.config_manager.get_all())
