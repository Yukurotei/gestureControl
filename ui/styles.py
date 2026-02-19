STYLESHEET = """
/* ═══════════════════════════════════════════════════════════
   GESTURE.AI — Neural Interface Control System
   Visual Design System v2.0
   Palette: Deep void (#05050f) + Cyan (#00d4ff) + Purple (#7b2fff)
═══════════════════════════════════════════════════════════ */

QMainWindow, QWidget {
    background-color: #05050f;
    color: #b8c4e0;
    font-family: 'Inter', 'SF Pro Display', 'Segoe UI', sans-serif;
    font-size: 13px;
}

QLabel {
    background: transparent;
    color: #b8c4e0;
}

/* ═══ HEADER BAR ═══ */

#header_widget {
    background-color: #07071a;
    border-bottom: 1px solid #13133a;
}

#app_name {
    color: #00d4ff;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 4px;
    font-family: 'Courier New', 'Consolas', monospace;
}

#app_sub {
    color: #252848;
    font-size: 9px;
    letter-spacing: 3px;
    font-family: 'Courier New', 'Consolas', monospace;
}

#cam_label {
    color: #252848;
    font-size: 9px;
    letter-spacing: 3px;
    font-family: 'Courier New', 'Consolas', monospace;
}

/* ═══ RIGHT SIDE PANEL ═══ */

#right_panel {
    background-color: #07071a;
    border-left: 1px solid #13133a;
}

/* ═══ CARDS ═══ */

#state_card, #detect_card, #controls_card {
    background-color: #0a0a1f;
    border: 1px solid #151535;
    border-radius: 10px;
}

#card_label {
    color: #252848;
    font-size: 9px;
    letter-spacing: 3px;
    font-family: 'Courier New', 'Consolas', monospace;
}

#state_value {
    color: #00d4ff;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 3px;
    font-family: 'Courier New', 'Consolas', monospace;
}

#status_name {
    color: #404868;
    font-size: 10px;
    letter-spacing: 2px;
    font-family: 'Courier New', 'Consolas', monospace;
}

#status_value {
    color: #7080a0;
    font-size: 12px;
    font-family: 'Courier New', 'Consolas', monospace;
}

/* ═══ BUTTONS ═══ */

QPushButton {
    background: #0a0a1f;
    color: #404868;
    border: 1px solid #151535;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    font-family: 'Courier New', 'Consolas', monospace;
    min-height: 36px;
}

QPushButton:hover {
    color: #c8d0f0;
    border-color: #303060;
    background: #0d0d25;
}

QPushButton:pressed {
    background: #080818;
}

QPushButton:disabled {
    color: #151528;
    border-color: #0a0a1a;
    background: #06060f;
}

/* Calibrate — Purple */
#btn_calibrate {
    color: #7b2fff;
    border-color: #3a1880;
    background: #0a061e;
}

#btn_calibrate:hover {
    color: #a060ff;
    border-color: #5a28c0;
    background: #0d0825;
}

#btn_calibrate:disabled {
    color: #18083a;
    border-color: #0d051e;
    background: #06030f;
}

/* Start — Green */
#btn_start {
    color: #00ff88;
    border-color: #007840;
    background: #061e10;
}

#btn_start:hover {
    color: #40ffa8;
    border-color: #00a858;
    background: #082514;
}

#btn_start:disabled {
    color: #082518;
    border-color: #051008;
    background: #040d08;
}

/* Stop — Red */
#btn_stop {
    color: #ff2255;
    border-color: #800020;
    background: #1e0608;
}

#btn_stop:hover {
    color: #ff5577;
    border-color: #c00030;
    background: #250810;
}

#btn_stop:disabled {
    color: #1e050c;
    border-color: #100208;
    background: #0d0308;
}

/* ═══ PROGRESS BAR ═══ */

QProgressBar {
    background: #07071a;
    border: 1px solid #151535;
    border-radius: 3px;
    max-height: 4px;
    color: transparent;
    text-align: center;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #7b2fff, stop:1 #00d4ff);
    border-radius: 3px;
}

/* ═══ CAMERA SELECTOR ═══ */

QComboBox {
    background: #0a0a1f;
    border: 1px solid #151535;
    border-radius: 6px;
    padding: 5px 10px;
    color: #505878;
    font-size: 11px;
    font-family: 'Courier New', 'Consolas', monospace;
    min-width: 140px;
    max-width: 240px;
}

QComboBox:hover {
    border-color: #252558;
    color: #8090b0;
}

QComboBox:focus {
    border-color: #00d4ff;
    color: #c8d0f0;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #404868;
    margin-right: 6px;
}

QComboBox QAbstractItemView {
    background: #0d0d25;
    border: 1px solid #252558;
    selection-background-color: #151540;
    selection-color: #00d4ff;
    outline: none;
    padding: 4px;
}

QComboBox QAbstractItemView::item {
    padding: 6px 10px;
    color: #505878;
    font-family: 'Courier New', 'Consolas', monospace;
    font-size: 11px;
}

QComboBox QAbstractItemView::item:hover {
    color: #c8d0f0;
    background: #151540;
}

/* ═══ SCROLL BARS ═══ */

QScrollArea {
    background: transparent;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background: transparent;
}

QScrollBar:vertical {
    background: #07071a;
    width: 5px;
    border-radius: 2px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #202050;
    border-radius: 2px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background: #303070;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}

/* ═══ SETTINGS ═══ */

#settings_group {
    background: #0a0a1f;
    border: 1px solid #151535;
    border-radius: 8px;
}

#group_title {
    color: #404868;
    font-size: 9px;
    letter-spacing: 3px;
    font-family: 'Courier New', 'Consolas', monospace;
}

#setting_name {
    color: #7888b0;
    font-size: 11px;
    font-family: 'Courier New', 'Consolas', monospace;
}

#setting_desc {
    color: #484e6a;
    font-size: 10px;
    font-family: 'Courier New', 'Consolas', monospace;
}

/* ═══ SPIN BOXES ═══ */

QDoubleSpinBox, QSpinBox {
    background: #0d0d25;
    border: 1px solid #151535;
    border-radius: 5px;
    padding: 4px 6px;
    color: #7080a0;
    font-family: 'Courier New', 'Consolas', monospace;
    font-size: 11px;
    min-width: 65px;
    max-width: 95px;
}

QDoubleSpinBox:focus, QSpinBox:focus {
    border-color: #00d4ff;
    color: #c8d0f0;
}

QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {
    background: #12123a;
    border: none;
    width: 16px;
}

QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {
    background: #1a1a4a;
}

/* ═══ CHECKBOXES ═══ */

QCheckBox {
    color: #505878;
    font-family: 'Courier New', 'Consolas', monospace;
    font-size: 11px;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #252558;
    border-radius: 3px;
    background: #0d0d25;
}

QCheckBox::indicator:checked {
    background: rgba(0, 212, 255, 0.12);
    border-color: #00d4ff;
}

QCheckBox::indicator:hover {
    border-color: #505878;
}

/* ═══ TOOLTIP ═══ */

QToolTip {
    background: #0d0d25;
    border: 1px solid #252558;
    color: #b8c4e0;
    padding: 5px 8px;
    border-radius: 4px;
    font-family: 'Courier New', 'Consolas', monospace;
    font-size: 11px;
}
"""
