"""Reusable widgets: animated sprites, HP bars, chips, cards and team strips."""
from __future__ import annotations

from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt, QTimer, pyqtProperty
from PyQt5.QtGui import QColor, QFont, QMovie, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout,
                             QWidget)

from ui import gamedata as gd
from ui import theme


def set_prop(widget, name: str, value) -> None:
    """Sets a dynamic property and refreshes the stylesheet."""
    widget.setProperty(name, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class AnimatedSprite(QLabel):
    """Animated GIF that keeps its aspect ratio; the movie is only recreated when the path changes."""

    def __init__(self, path: str | None = None, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self._movie = None
        self._path = None
        self._native = QSize(1, 1)
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity)
        if path:
            self.set_sprite(path)

    def set_sprite(self, path: str | None) -> None:
        if path == self._path:
            return
        self._path = path
        if self._movie is not None:
            self._movie.stop()
            self._movie.deleteLater()
            self._movie = None
        if not path:
            self.clear()
            return
        movie = QMovie(path)
        movie.setCacheMode(QMovie.CacheAll)
        movie.jumpToFrame(0)
        self._native = movie.currentImage().size()
        self._movie = movie
        self.setMovie(movie)
        self._rescale()
        movie.start()

    def set_faded(self, faded: bool) -> None:
        self._stop_blink()
        self._faded = faded
        self._opacity.setOpacity(0.28 if faded else 1.0)

    def blink(self, times: int = 3, interval: int = 70, end_faded: bool = False) -> None:
        """Hit feedback: the sprite flickers and ends visible (or faded if it fainted)."""
        self._stop_blink()
        self._blink_left = times * 2
        self._blink_end = end_faded
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink_step)
        self._blink_timer.start(max(20, interval))

    def _blink_step(self):
        self._blink_left -= 1
        if self._blink_left <= 0:
            self.set_faded(self._blink_end)
            return
        self._opacity.setOpacity(0.15 if self._blink_left % 2 else 1.0)

    def _stop_blink(self):
        timer = getattr(self, "_blink_timer", None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()
            self._blink_timer = None

    def _rescale(self):
        if self._movie is None or self._native.isEmpty():
            return
        size = self._native.scaled(self.size(), Qt.KeepAspectRatio)
        if size.width() > 0 and size.height() > 0:
            self._movie.setScaledSize(size)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


def static_pixmap(path: str, size: int) -> QPixmap:
    pix = QPixmap(path)
    if pix.isNull():
        return pix
    return pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class HPBar(QWidget):
    """Rounded HP bar with animated changes and green / yellow / red colour."""

    def __init__(self, parent=None, show_text: bool = True):
        super().__init__(parent)
        self._value = 100.0
        self.show_text = show_text
        self.setMinimumHeight(16)
        self._anim = QPropertyAnimation(self, b"value", self)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def get_value(self) -> float:
        return self._value

    def set_raw_value(self, v: float) -> None:
        self._value = max(0.0, min(100.0, float(v)))
        self.update()

    value = pyqtProperty(float, fget=get_value, fset=set_raw_value)

    def set_value(self, v: float, animate: bool = True, duration: int = 450) -> None:
        self._anim.stop()
        if not animate or duration <= 0:
            self.set_raw_value(v)
            return
        self._anim.setDuration(duration)
        self._anim.setStartValue(self._value)
        self._anim.setEndValue(float(v))
        self._anim.start()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        radius = r.height() / 2
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.PANEL_3))
        p.drawRoundedRect(r, radius, radius)
        if self._value > 0:
            fill = QRectF(r.x(), r.y(), max(r.height(), r.width() * self._value / 100.0), r.height())
            p.setBrush(QColor(theme.hp_color(self._value)))
            p.drawRoundedRect(fill, radius, radius)
        if self.show_text:
            p.setPen(QColor("#0b1020") if self._value > 55 else QColor(theme.TEXT))
            f = QFont(self.font())
            f.setBold(True)
            f.setPointSizeF(max(7.0, r.height() * 0.5))
            p.setFont(f)
            p.drawText(r, Qt.AlignCenter, f"{round(self._value)} %")


class Chip(QLabel):
    def __init__(self, text: str = "", variant: str = "neutral", parent=None):
        super().__init__(text, parent)
        self.setProperty("chip", variant)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

    def set(self, text: str, variant: str | None = None):
        self.setText(text)
        if variant is not None and variant != self.property("chip"):
            set_prop(self, "chip", variant)


class Card(QFrame):
    def __init__(self, title: str | None = None, parent=None, margins=(14, 12, 14, 12), spacing=8):
        super().__init__(parent)
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(*margins)
        self.body.setSpacing(spacing)
        if title:
            label = QLabel(title.upper())
            label.setProperty("role", "section")
            self.body.addWidget(label)


class StatTile(QFrame):
    """Big value with a caption below."""

    def __init__(self, caption: str, value: str = "-", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(0)
        self.value = QLabel(value)
        self.value.setProperty("role", "big")
        self.caption = QLabel(caption)
        self.caption.setProperty("role", "muted")
        lay.addWidget(self.value)
        lay.addWidget(self.caption)

    def set(self, value: str, color: str | None = None):
        self.value.setText(value)
        self.value.setStyleSheet(f"color: {color};" if color else "")


class PokeballRow(QWidget):
    """One small ball per team slot: alive, fainted or empty."""

    def __init__(self, parent=None, size: int = 12):
        super().__init__(parent)
        self.ball = size
        self.total = 6
        self.alive = 0
        self.fainted = set()
        self.setFixedHeight(size + 4)
        self.setMinimumWidth(6 * (size + 4))
        self.setAttribute(Qt.WA_TranslucentBackground)

    def set_state(self, total: int, fainted):
        """``fainted``: set of fainted positions (or how many, counting from the first)."""
        self.total = total
        self.fainted = set(fainted) if not isinstance(fainted, int) else set(range(fainted))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        d = self.ball
        for i in range(6):
            x = 2 + i * (d + 4)
            r = QRectF(x, 2, d, d)
            if i >= self.total:
                p.setPen(QPen(QColor(theme.BORDER), 1.2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(r)
                continue
            fainted = i in self.fainted
            top = QColor("#5b6478") if fainted else QColor(theme.ACCENT)
            bottom = QColor("#3a4152") if fainted else QColor("#f1f5f9")
            path = QPainterPath()
            path.addEllipse(r)
            p.setPen(Qt.NoPen)
            p.setClipPath(path)
            p.fillRect(QRectF(x, 2, d, d / 2), top)
            p.fillRect(QRectF(x, 2 + d / 2, d, d / 2), bottom)
            p.setClipping(False)
            p.setPen(QPen(QColor("#0b1020"), 1.2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(r)
            p.drawLine(int(x), int(2 + d / 2), int(x + d), int(2 + d / 2))


class TeamStrip(QWidget):
    """Row of Pokémon sprites (animated or static) with optional level and fainted state."""

    def __init__(self, size: int = 48, animated: bool = True, parent=None, show_names: bool = False):
        super().__init__(parent)
        self.size_px = size
        self.animated = animated
        self.show_names = show_names
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(4)
        self.lay.addStretch(1)
        self._slots = []
        self.setAttribute(Qt.WA_TranslucentBackground)

    def clear_team(self):
        for w in self._slots:
            w.setParent(None)
            w.deleteLater()
        self._slots = []

    def set_team(self, ids: list, levels: list | None = None, fainted: set | None = None, tooltips: list | None = None):
        self.clear_team()
        fainted = fainted or set()
        for i, pkm_id in enumerate(ids):
            box = QWidget()
            box.setAttribute(Qt.WA_TranslucentBackground)
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(0)
            path = gd.sprite_path(pkm_id, "front")
            if self.animated:
                sprite = AnimatedSprite(path)
                sprite.setFixedSize(self.size_px, self.size_px)
            else:
                sprite = QLabel()
                sprite.setFixedSize(self.size_px, self.size_px)
                sprite.setAlignment(Qt.AlignCenter)
                sprite.setStyleSheet("background: transparent;")
                sprite.setPixmap(static_pixmap(path, self.size_px))
            v.addWidget(sprite, 0, Qt.AlignHCenter)
            if self.show_names or levels:
                text = gd.pkm_name(pkm_id) if self.show_names else ""
                if levels:
                    text = (text + " " if text else "") + f"Nv {levels[i]}"
                cap = QLabel(text)
                cap.setAlignment(Qt.AlignCenter)
                cap.setStyleSheet(f"color: {theme.MUTED}; font-size: 10px; background: transparent;")
                v.addWidget(cap)
            tip = tooltips[i] if tooltips and i < len(tooltips) else gd.pkm_name(pkm_id)
            box.setToolTip(tip)
            effect = QGraphicsOpacityEffect(box)
            effect.setOpacity(0.28 if i in fainted else 1.0)
            box.setGraphicsEffect(effect)
            self.lay.insertWidget(self.lay.count() - 1, box)
            self._slots.append(box)

    def set_fainted(self, indices: set):
        for i, box in enumerate(self._slots):
            effect = box.graphicsEffect()
            if effect is not None:
                effect.setOpacity(0.28 if i in indices else 1.0)


def hline() -> QFrame:
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet(f"background-color: {theme.BORDER};")
    return line


def section_label(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setProperty("role", "section")
    return label
