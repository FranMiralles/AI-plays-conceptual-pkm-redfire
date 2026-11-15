"""Move animations by elemental type (sheets in images/move_animations)."""
from __future__ import annotations

import json

from PyQt5.QtCore import QRect, Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel

from route_data.move_types import MOVE_TYPES
from ui import gamedata as gd

ANIMATIONS_DIR = gd.IMAGES / "move_animations"

TYPE_NAMES = {
    "normal": "Normal", "fighting": "Lucha", "flying": "Volador", "poison": "Veneno", "ground": "Tierra",
    "rock": "Roca", "bug": "Bicho", "ghost": "Fantasma", "steel": "Acero", "fire": "Fuego", "water": "Agua",
    "grass": "Planta", "electric": "Eléctrico", "psychic": "Psíquico", "ice": "Hielo", "dragon": "Dragón",
    "dark": "Siniestro",
}
TYPE_COLORS = {
    "normal": "#a8a77a", "fighting": "#c22e28", "flying": "#a98ff3", "poison": "#a33ea1", "ground": "#e2bf65",
    "rock": "#b6a136", "bug": "#a6b91a", "ghost": "#735797", "steel": "#b7b7ce", "fire": "#ee8130",
    "water": "#6390f0", "grass": "#7ac74c", "electric": "#f7d02c", "psychic": "#f95587", "ice": "#96d9d6",
    "dragon": "#6f35fc", "dark": "#705746",
}


def move_type(move: str) -> str | None:
    return MOVE_TYPES.get(move)


class EffectLibrary:
    """Loads the sheets lazily and cuts the frames listed in animations.json."""

    def __init__(self, directory=ANIMATIONS_DIR):
        self.directory = directory
        index_path = directory / "animations.json"
        self.index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"types": {}}
        self.cell = self.index.get("cell_size", 192)
        self._frames: dict[str, list] = {}

    def frames(self, type_name: str | None) -> list:
        if not type_name or type_name not in self.index["types"]:
            return []
        if type_name not in self._frames:
            spec = self.index["types"][type_name]
            sheet = QPixmap(str(self.directory / spec["file"]))
            cols, c = spec["columns"], self.cell
            frames = []
            if not sheet.isNull():
                for cell in spec["cells"]:
                    row, col = divmod(cell, cols)
                    frames.append(sheet.copy(QRect(col * c, row * c, c, c)))
            self._frames[type_name] = frames
        return self._frames[type_name]


_library = None


def library() -> EffectLibrary:
    global _library
    if _library is None:
        _library = EffectLibrary()
    return _library


class EffectOverlay(QLabel):
    """Plays a frame sequence over a widget area (pixel art scaled without smoothing)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.setAlignment(Qt.AlignCenter)
        self.hide()
        self._frames = []
        self._index = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next)

    def play(self, frames: list, center, size: int, duration_ms: int):
        self.stop()
        if not frames or size <= 0:
            return
        cell = frames[0].width()
        size = cell * max(1, round(size / cell))  # integer zoom: every pixel becomes an equal block
        self._frames = [f.scaled(size, size, Qt.KeepAspectRatio, Qt.FastTransformation) for f in frames]
        self._index = 0
        self.setGeometry(int(center.x() - size / 2), int(center.y() - size / 2), size, size)
        self.setPixmap(self._frames[0])
        self.show()
        self.raise_()
        self._timer.start(max(16, int(duration_ms / len(self._frames))))

    def _next(self):
        self._index += 1
        if self._index >= len(self._frames):
            self.stop()
            return
        self.setPixmap(self._frames[self._index])

    def stop(self):
        self._timer.stop()
        self.hide()

    def is_playing(self) -> bool:
        return self._timer.isActive()
