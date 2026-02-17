DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Noto Sans", sans-serif;
    font-size: 13px;
}

QLabel {
    color: #cdd6f4;
    background: transparent;
}

QLabel#header-title {
    font-size: 18px;
    font-weight: bold;
    color: #cdd6f4;
}

QLabel#status-label {
    font-size: 12px;
    color: #a6adc8;
    padding: 2px 0px;
}

QLabel#status-value {
    font-size: 12px;
    font-weight: bold;
    padding: 2px 0px;
}

QLabel#setting-name {
    font-size: 13px;
    font-weight: bold;
    color: #cdd6f4;
}

QLabel#setting-desc {
    font-size: 11px;
    color: #6c7086;
    padding-bottom: 6px;
}

QLabel#section-title {
    font-size: 15px;
    font-weight: bold;
    color: #89b4fa;
    padding: 8px 0px 4px 0px;
}

QLabel#camera-preview-placeholder {
    color: #585b70;
    font-size: 14px;
    border: 2px dashed #45475a;
    border-radius: 8px;
}

QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 20px;
    font-size: 13px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #45475a;
    border-color: #585b70;
}

QPushButton:pressed {
    background-color: #585b70;
}

QPushButton:disabled {
    background-color: #1e1e2e;
    color: #45475a;
    border-color: #313244;
}

QPushButton#btn-calibrate {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
}

QPushButton#btn-calibrate:hover {
    background-color: #b4d0fb;
}

QPushButton#btn-calibrate:pressed {
    background-color: #74a8f7;
}

QPushButton#btn-calibrate:disabled {
    background-color: #45475a;
    color: #585b70;
}

QPushButton#btn-start {
    background-color: #a6e3a1;
    color: #1e1e2e;
    border: none;
}

QPushButton#btn-start:hover {
    background-color: #c0eebb;
}

QPushButton#btn-start:pressed {
    background-color: #8cd987;
}

QPushButton#btn-start:disabled {
    background-color: #45475a;
    color: #585b70;
}

QPushButton#btn-stop {
    background-color: #f38ba8;
    color: #1e1e2e;
    border: none;
}

QPushButton#btn-stop:hover {
    background-color: #f7adc0;
}

QPushButton#btn-stop:pressed {
    background-color: #e97a99;
}

QPushButton#btn-stop:disabled {
    background-color: #45475a;
    color: #585b70;
}

QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    min-width: 200px;
}

QComboBox:hover {
    border-color: #89b4fa;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #cdd6f4;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    selection-background-color: #45475a;
    selection-color: #cdd6f4;
}

QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 13px;
}

QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #89b4fa;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #89b4fa;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #45475a;
    border: none;
    width: 20px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #585b70;
}

QCheckBox {
    color: #cdd6f4;
    spacing: 8px;
    font-size: 13px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #45475a;
    background-color: #313244;
}

QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

QCheckBox::indicator:hover {
    border-color: #89b4fa;
}

QScrollArea {
    border: none;
    background: transparent;
}

QScrollArea > QWidget > QWidget {
    background: transparent;
}

QScrollBar:vertical {
    background-color: #1e1e2e;
    width: 8px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QProgressBar {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    text-align: center;
    color: #cdd6f4;
    font-size: 12px;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 5px;
}

QFrame#separator {
    background-color: #45475a;
    max-height: 1px;
}
"""
