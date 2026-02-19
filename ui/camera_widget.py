"""
Camera preview widget with animated overlay (corner brackets, scan line, state glow).
"""

import math

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui  import (
    QImage, QPixmap, QPainter, QColor, QPen, QLinearGradient, QFont,
)
from PyQt6.QtWidgets import QWidget

# ── Per-state colour palette ─────────────────────────────────────
_STATE_CFG = {
    #           border-RGBA               bracket-RGBA             scanline
    'idle':        (QColor(0,   90, 120, 80),  QColor(0,  90, 120, 100), False),
    'calibrating': (QColor(255, 170,  0, 130), QColor(255,170,   0, 160), True),
    'ready':       (QColor(123,  47, 255, 130), QColor(123, 47, 255, 160), False),
    'tracking':    (QColor(0,  212, 255, 170), QColor(0,  212, 255, 210), False),
}

_STATE_LABEL = {
    'idle':        ('NO SIGNAL',   QColor(0,  90, 120, 120)),
    'calibrating': ('CALIBRATING', QColor(255,170,  0, 160)),
    'ready':       ('STANDBY',     QColor(123, 47, 255, 160)),
    'tracking':    ('TRACKING',    QColor(0, 212, 255, 200)),
}


class CameraPreview(QWidget):
    """Live camera display with animated state overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self._pixmap   = None
        self._state    = 'idle'
        self._phase    = 0.0
        self._scan_pos = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    # ── public interface ─────────────────────────────────────────

    def update_frame(self, bgr_frame: np.ndarray):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qi)
        self.update()

    def clear_frame(self):
        self._pixmap = None
        self.update()

    def set_tracking_state(self, state: str):
        self._state = state
        if state != 'calibrating':
            self._scan_pos = 0.0

    def set_calibration_progress(self, current: int, total: int):
        """Drive scan-line position from calibration frame progress."""
        if total > 0:
            self._scan_pos = current / total

    # ── internals ────────────────────────────────────────────────

    def _tick(self):
        self._phase = (self._phase + 0.042) % (2 * math.pi)
        self.update()

    def resizeEvent(self, event):  # type: ignore
        super().resizeEvent(event)
        self.update()

    # ── painting ─────────────────────────────────────────────────

    def paintEvent(self, event):  # type: ignore
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        border_c, bracket_c, has_scan = _STATE_CFG.get(
            self._state, _STATE_CFG['idle']
        )
        pulse = 0.62 + 0.38 * math.sin(self._phase)

        # ── Background ───────────────────────────────────────────
        painter.fillRect(self.rect(), QColor(5, 5, 15))

        # ── Camera frame ─────────────────────────────────────────
        if self._pixmap:
            pw, ph = self._pixmap.width(), self._pixmap.height()
            scale  = min(w / pw, h / ph)
            dw, dh = int(pw * scale), int(ph * scale)
            dx, dy = (w - dw) // 2, (h - dh) // 2
            scaled = self._pixmap.scaled(
                dw, dh,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(dx, dy, scaled)
        else:
            # "NO SIGNAL" placeholder
            font = QFont('Courier New', 11)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 4)
            painter.setFont(font)
            painter.setPen(QColor(30, 40, 70))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "NO SIGNAL"
            )

        # ── Calibration scan line ────────────────────────────────
        if has_scan:
            sy = int(self._scan_pos * h)
            grad = QLinearGradient(0, sy - 13, 0, sy + 13)
            c0 = QColor(255, 170, 0, 0)
            c1 = QColor(255, 170, 0, 120)
            grad.setColorAt(0.0, c0)
            grad.setColorAt(0.5, c1)
            grad.setColorAt(1.0, c0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawRect(0, sy - 13, w, 26)
            painter.setPen(QPen(QColor(255, 170, 0, 170), 1))
            painter.drawLine(0, sy, w, sy)

        # ── Corner brackets ──────────────────────────────────────
        bc = QColor(bracket_c)
        bc.setAlpha(int(bracket_c.alpha() * pulse))
        pen = QPen(bc, 2)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        m  = 10
        bl = max(16, min(32, w // 14, h // 10))

        # TL
        painter.drawLine(m, m, m + bl, m);      painter.drawLine(m, m, m, m + bl)
        # TR
        painter.drawLine(w-m, m, w-m-bl, m);    painter.drawLine(w-m, m, w-m, m+bl)
        # BL
        painter.drawLine(m, h-m, m+bl, h-m);    painter.drawLine(m, h-m, m, h-m-bl)
        # BR
        painter.drawLine(w-m, h-m, w-m-bl, h-m); painter.drawLine(w-m, h-m, w-m, h-m-bl)

        # Accent dots at bracket vertices
        dot_c = QColor(bc)
        dot_c.setAlpha(min(255, int(bc.alpha() * 1.3)))
        painter.setBrush(dot_c)
        painter.setPen(Qt.PenStyle.NoPen)
        for (cx, cy) in [(m, m), (w-m, m), (m, h-m), (w-m, h-m)]:
            painter.drawEllipse(QRectF(cx - 2.0, cy - 2.0, 4.0, 4.0))

        # ── Border glow ──────────────────────────────────────────
        gc = QColor(border_c)
        gc.setAlpha(int(border_c.alpha() * pulse * 0.45))
        painter.setPen(QPen(gc, 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(1, 1, w - 2, h - 2)

        # ── State text (bottom-left, above bracket) ──────────────
        label_text, label_color = _STATE_LABEL.get(self._state, ('', QColor(0,0,0,0)))
        if label_text:
            lc = QColor(label_color)
            lc.setAlpha(int(label_color.alpha() * pulse))
            painter.setPen(lc)
            font = QFont('Courier New', 8)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
            painter.setFont(font)
            painter.drawText(m + bl + 10, h - m - 2, label_text)
