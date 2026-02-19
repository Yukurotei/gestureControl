"""
Animated effect widgets for Gesture.AI Neural Interface.
"""
import math

from PyQt6.QtWidgets import QWidget, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QRadialGradient

# ── Palette ───────────────────────────────────────────────────────
C_CYAN    = QColor(0, 212, 255)
C_PURPLE  = QColor(123, 47, 255)
C_MAGENTA = QColor(255, 47, 170)
C_GREEN   = QColor(0, 255, 136)
C_RED     = QColor(255, 34, 85)
C_ORANGE  = QColor(255, 170, 0)


class PulsingOrb(QWidget):
    """
    Animated glowing status orb.

    Modes: 'off', 'active', 'warning', 'error', 'cyan', 'purple', 'magenta'
    """

    _MODES: dict[str, tuple[QColor, float, float]] = {
        'off':     (QColor(20, 20, 50),  0.00, 0.02),
        'active':  (C_GREEN,             1.00, 0.06),
        'warning': (C_ORANGE,            0.85, 0.10),
        'error':   (C_RED,               1.00, 0.13),
        'cyan':    (C_CYAN,              1.00, 0.07),
        'purple':  (C_PURPLE,            0.90, 0.08),
        'magenta': (C_MAGENTA,           0.90, 0.09),
    }

    def __init__(self, size: int = 10, parent=None):
        super().__init__(parent)
        self._orb_r = size / 2
        self._phase = 0.0
        self._mode  = 'off'
        pad = int(size * 1.6)
        self.setFixedSize(size + pad, size + pad)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    def _tick(self):
        _, intensity, speed = self._MODES[self._mode]
        if intensity > 0:
            self._phase = (self._phase + speed) % (2 * math.pi)
        self.update()

    def set_mode(self, mode: str):
        if mode in self._MODES:
            self._mode = mode

    def paintEvent(self, event):  # type: ignore
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        color, intensity, _ = self._MODES[self._mode]
        cx = self.width()  / 2
        cy = self.height() / 2
        r  = self._orb_r

        if intensity > 0:
            pulse = 0.60 + 0.40 * math.sin(self._phase)
            a = int(intensity * pulse * 255)

            # Outer glow rings
            for i in range(5, 0, -1):
                g = QColor(color)
                g.setAlpha(max(0, int(a * 0.11 / i)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(g)
                rr = r + i * 3.4
                painter.drawEllipse(QRectF(cx - rr, cy - rr, rr * 2, rr * 2))

            # Core radial gradient
            grad = QRadialGradient(cx, cy - r * 0.25, r * 0.45)
            white = QColor(255, 255, 255, int(a * 0.85))
            mid   = QColor(color); mid.setAlpha(a)
            edge  = QColor(color); edge.setAlpha(int(a * 0.30))
            grad.setColorAt(0.0,  white)
            grad.setColorAt(0.45, mid)
            grad.setColorAt(1.0,  edge)
            painter.setBrush(grad)
        else:
            painter.setBrush(QColor(20, 20, 50))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))


def make_glow_effect(color: str, radius: int = 20) -> QGraphicsDropShadowEffect:
    """Return a QGraphicsDropShadowEffect that creates a coloured glow."""
    fx = QGraphicsDropShadowEffect()
    fx.setColor(QColor(color))
    fx.setBlurRadius(radius)
    fx.setOffset(0, 0)
    return fx
