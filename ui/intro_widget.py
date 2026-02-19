"""
Full-screen intro animation overlay for Gesture.AI Neural Interface.

Animation timeline (ms):
  0  – 600  : Particles materialise, grid fades in
  300–1500  : Title text types character-by-character
  600–1300  : Scan line sweeps top-to-bottom
  1500      : Subtitle fades in
  2000–2600 : Everything fades to transparent → widget hides
"""

import math
import random

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore    import Qt, QTimer, QRectF, pyqtSignal
from PyQt6.QtGui     import (
    QPainter, QColor, QPen, QLinearGradient, QFont,
)

_TITLE         = "GESTURE.AI"
_SUBTITLE      = "NEURAL INTERFACE CONTROL SYSTEM"
_TOTAL_MS      = 2700
_FADE_START_MS = 2050
_FADE_MS       = 650


class IntroWidget(QWidget):
    """Fullscreen animated intro overlay.  Call :meth:`start` to begin."""

    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)

        rng = random.Random(7)
        self._particles = [
            {
                'x':      rng.random(),
                'y':      rng.random(),
                'r':      rng.uniform(0.7, 2.4),
                'alpha':  0,
                'target': rng.randint(50, 170),
            }
            for _ in range(130)
        ]

        self._elapsed      = 0
        self._title_chars  = 0
        self._overall_alpha = 255

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # ── public ──────────────────────────────────────────────────

    def start(self):
        self._elapsed       = 0
        self._title_chars   = 0
        self._overall_alpha = 255
        for p in self._particles:
            p['alpha'] = 0
        self.show()
        self.raise_()
        self._timer.start(16)

    # ── animation loop ──────────────────────────────────────────

    def _tick(self):
        self._elapsed += 16
        t = self._elapsed

        # Particles fade in
        if t < 700:
            for p in self._particles:
                if p['alpha'] < p['target']:
                    p['alpha'] = min(p['alpha'] + 5, p['target'])

        # Title types out  (300–1500 ms)
        if 300 <= t <= 1500:
            self._title_chars = min(
                int((t - 300) / 1200 * len(_TITLE)),
                len(_TITLE)
            )
        elif t > 1500:
            self._title_chars = len(_TITLE)

        # Fade out
        if t >= _FADE_START_MS:
            fade = (t - _FADE_START_MS) / _FADE_MS
            self._overall_alpha = max(0, int(255 * (1.0 - fade)))

        if t >= _TOTAL_MS:
            self._timer.stop()
            self.finished.emit()
            self.hide()
            return

        self.update()

    # ── paint ───────────────────────────────────────────────────

    def paintEvent(self, event):  # type: ignore
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        t    = self._elapsed
        oa   = self._overall_alpha / 255.0   # overall opacity factor

        # ── Background ──────────────────────────────────────────
        painter.fillRect(self.rect(), QColor(5, 5, 15, self._overall_alpha))

        # ── Grid ────────────────────────────────────────────────
        gc = QColor(0, 212, 255, max(0, int(10 * oa)))
        painter.setPen(QPen(gc, 1))
        for x in range(0, w, 56):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, 56):
            painter.drawLine(0, y, w, y)

        # ── Particles ───────────────────────────────────────────
        painter.setPen(Qt.PenStyle.NoPen)
        for p in self._particles:
            pa = max(0, int(p['alpha'] * oa))
            painter.setBrush(QColor(0, 212, 255, pa))
            px = int(p['x'] * w)
            py = int(p['y'] * h)
            r  = p['r']
            painter.drawEllipse(QRectF(px - r, py - r, r * 2, r * 2))

        # ── Scan line (600–1300 ms) ──────────────────────────────
        if 600 < t < 1300:
            sp = (t - 600) / 700
            sy = int(sp * h)
            grad = QLinearGradient(0, sy - 14, 0, sy + 14)
            c0 = QColor(0, 212, 255, 0)
            c1 = QColor(0, 212, 255, max(0, int(130 * oa)))
            grad.setColorAt(0.0, c0)
            grad.setColorAt(0.5, c1)
            grad.setColorAt(1.0, c0)
            painter.setBrush(grad)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(0, sy - 14, w, 28)
            painter.setPen(QPen(QColor(0, 212, 255, max(0, int(200 * oa))), 1))
            painter.drawLine(0, sy, w, sy)

        # ── Title ────────────────────────────────────────────────
        if self._title_chars > 0:
            fsize = max(26, min(52, w // 22))
            font  = QFont('Courier New', fsize, QFont.Weight.Bold)
            painter.setFont(font)
            fm    = painter.fontMetrics()
            disp  = _TITLE[:self._title_chars]
            fw    = fm.horizontalAdvance(_TITLE)
            tx    = (w - fw) // 2
            ty    = h // 2 - 22

            # Glow layers
            for gr in range(12, 1, -2):
                g = QColor(0, 212, 255, max(0, int(22 * oa * (1 - gr / 14))))
                painter.setPen(QPen(g, gr))
                painter.drawText(tx, ty, disp)

            # Solid text
            painter.setPen(QColor(0, 212, 255, max(0, int(255 * oa))))
            painter.drawText(tx, ty, disp)

            # Blinking cursor while typing
            if self._title_chars < len(_TITLE):
                cx = tx + fm.horizontalAdvance(disp)
                if int(t / 180) % 2 == 0:
                    lc = QColor(0, 212, 255, max(0, int(200 * oa)))
                    painter.setPen(QPen(lc, 2))
                    painter.drawLine(cx + 3, ty - fm.ascent() + 6, cx + 3, ty + 2)

        # ── Subtitle ────────────────────────────────────────────
        if t > 1500 and self._title_chars == len(_TITLE):
            sub_fade = min((t - 1500) / 420, 1.0)
            sfont = QFont('Courier New', max(9, w // 75))
            painter.setFont(sfont)
            fm   = painter.fontMetrics()
            sw   = fm.horizontalAdvance(_SUBTITLE)
            sx   = (w - sw) // 2
            sy2  = h // 2 + 24
            sc   = QColor(80, 90, 130, max(0, int(sub_fade * oa * 155)))
            painter.setPen(sc)
            painter.drawText(sx, sy2, _SUBTITLE)

        # ── Corner brackets ──────────────────────────────────────
        mc  = QColor(0, 212, 255, max(0, int(170 * oa)))
        pen = QPen(mc, 2)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        ml, m = 50, 22

        painter.drawLine(m, m, m + ml, m);  painter.drawLine(m, m, m, m + ml)          # TL
        painter.drawLine(w-m, m, w-m-ml, m);painter.drawLine(w-m, m, w-m, m+ml)        # TR
        painter.drawLine(m, h-m, m+ml, h-m);painter.drawLine(m, h-m, m, h-m-ml)        # BL
        painter.drawLine(w-m, h-m, w-m-ml, h-m);painter.drawLine(w-m, h-m, w-m, h-m-ml) # BR
