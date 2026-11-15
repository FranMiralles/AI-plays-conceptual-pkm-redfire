"""Individual viewer: summary, capture map and step-by-step battles."""
from __future__ import annotations

import ast
import json

import numpy as np

from PyQt5.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QMovie, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QFrame, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout,
                             QHeaderView, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMenu, QMessageBox,
                             QPushButton, QSlider, QStackedWidget, QTableWidget,
                             QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget)

import battle_simulator as bs
from ga import operators as ops
from ga.engine import individual_to_json, load_individual
from ga.fitness import Evaluator, simulate_detailed
from ui import charts
from ui import effects
from ui.effects import EffectOverlay
from ui.individual_help import HowItWorksPanel
from ui import gamedata as gd
from ui import theme
from ui.widgets import AnimatedSprite, Card, Chip, HPBar, PokeballRow, StatTile, TeamStrip, section_label

# --------------------------------------------------------------------------- map


class _MovieItem(QGraphicsPixmapItem):
    """Animated GIF inside a QGraphicsScene, scaled to fit a box keeping the aspect ratio."""

    def __init__(self, path: str, box_w: float, box_h: float):
        super().__init__()
        self.setTransformationMode(Qt.FastTransformation)
        self.movie = QMovie(path)
        self.movie.setCacheMode(QMovie.CacheAll)
        self.movie.jumpToFrame(0)
        native = self.movie.currentImage().size()
        if not native.isEmpty():
            self.movie.setScaledSize(native.scaled(int(box_w), int(box_h), Qt.KeepAspectRatio))
        self.movie.frameChanged.connect(self._frame)
        self._frame()
        self.movie.start()

    def _frame(self, *_):
        self.setPixmap(self.movie.currentPixmap())

    def stop(self):
        self.movie.stop()


class MapView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(f"background: {theme.PANEL}; border-radius: 10px;")
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.map_pix = QPixmap(str(gd.IMAGES / "map.png"))
        self.map_item = self.scene_.addPixmap(self.map_pix)
        self.map_item.setTransformationMode(Qt.SmoothTransformation)
        self.W, self.H = self.map_pix.width(), self.map_pix.height()
        self.scene_.setSceneRect(0, 0, self.W, self.H)
        self.items_ = []

    def set_individual(self, genome, cut: int):
        for it in self.items_:
            if isinstance(it, _MovieItem):
                it.stop()
            self.scene_.removeItem(it)
        self.items_ = []
        catches = genome[0]
        W, H = self.W, self.H
        font = QFont("Segoe UI")
        font.setPixelSize(int(H * 0.0175))
        font.setBold(True)
        for i, pkm in enumerate(catches):
            if i >= cut:
                path, kind = gd.NOT_CAPTURED_YET, "ball"
            elif pkm is None:
                path, kind = gd.NOT_CAPTURED, "ball"
            else:
                path, kind = gd.sprite_path(pkm, "front"), "pkm"
            px, py = gd.ROUTE_POSITIONS[i]
            x = W * px
            y = H * (py + (0.05 if kind == "ball" else 0))
            # inside the original marker box (0.1 W x 0.15 H): Pokémon standing on its bottom
            # edge (labels placed "up" stay above it), balls centred
            if kind == "pkm":
                sprite = _MovieItem(path, W * 0.058, H * 0.068)
                rect = sprite.boundingRect()
                sprite.setPos(x + W * 0.05 - rect.width() / 2, y + H * 0.148 - rect.height())
            else:
                sprite = _MovieItem(path, W * 0.06, H * 0.09)
                rect = sprite.boundingRect()
                sprite.setPos(x + W * 0.05 - rect.width() / 2, y + H * 0.075 - rect.height() / 2)
            sprite.setZValue(2)
            options = ", ".join(gd.pkm_name(p) for p in gd.route_options(i))
            chosen = "sin captura" if pkm is None else gd.pkm_name(pkm)
            if i >= cut:
                chosen = "aún no visitada"
            sprite.setToolTip(f"<b>{gd.ROUTE_NAMES[i]}</b><br>Captura: {chosen}<br>"
                              f"<span style='color:{theme.MUTED}'>Opciones: {options}</span>")
            self.scene_.addItem(sprite)
            self.items_.append(sprite)
            text = QGraphicsSimpleTextItem(gd.ROUTE_NAMES[i])
            text.setFont(font)
            text.setBrush(QBrush(QColor("white")))
            br = text.boundingRect()
            lw, lh = br.width() + 10, br.height() + 4
            d = gd.label_direction(i)
            dy = {"pkm": {"left": 0.15, "right": 0.15, "down": 0.18, "up": 0.08},
                  "ball": {"left": 0.10, "right": 0.10, "down": 0.13, "up": 0.03}}[kind][d]
            if d == "left":
                lx = x - lw + W * 0.02
            elif d == "right":
                lx = x + W * 0.078
            else:
                lx = x + W * 0.025
            ly = y - lh + H * dy
            bg = QGraphicsRectItem(QRectF(0, 0, lw, lh))
            bg.setBrush(QBrush(QColor(10, 16, 30, 190)))
            bg.setPen(QPen(Qt.NoPen))
            bg.setPos(lx, ly)
            text.setParentItem(bg)
            text.setPos(5, 2)
            bg.setZValue(1)
            self.scene_.addItem(bg)
            self.items_.append(bg)
        self._fit()

    def _fit(self):
        self.fitInView(self.scene_.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def showEvent(self, event):
        super().showEvent(event)
        self._fit()


class MapPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        row = QHBoxLayout()
        row.addWidget(QLabel("Capturas disponibles:"))
        self.stage = QComboBox()
        for label, _cut in gd.STAGES:
            self.stage.addItem(label)
        row.addWidget(self.stage)
        row.addStretch(1)
        self.summary = QLabel("")
        self.summary.setProperty("role", "muted")
        row.addWidget(self.summary)
        lay.addLayout(row)
        self.view = MapView()
        lay.addWidget(self.view, 1)
        self.genome = None
        self.stage.currentIndexChanged.connect(self._refresh)

    def set_individual(self, genome):
        self.genome = genome
        self._refresh()

    def _refresh(self, *_):
        if self.genome is None:
            return
        cut = gd.STAGES[self.stage.currentIndex()][1]
        self.view.set_individual(self.genome, cut)
        caught = sum(1 for p in self.genome[0][:cut] if p is not None)
        self.summary.setText(f"{caught} capturas en {cut} zonas visitadas · pasa el ratón por una zona para ver sus opciones")


# --------------------------------------------------------------------------- battle scene


class InfoPlate(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet(f"QFrame#card {{ background-color: rgba(20, 30, 51, 225); border: 1px solid {theme.BORDER};"
                           f" border-radius: 12px; }} QLabel {{ background: transparent; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.name = QLabel("-")
        self.name.setStyleSheet("font-weight: 800; font-size: 15px;")
        self.level = QLabel("")
        self.level.setStyleSheet(f"color: {theme.MUTED}; font-weight: 700;")
        self.status = QLabel("")
        self.status.hide()
        top.addWidget(self.name)
        top.addWidget(self.status)
        top.addStretch(1)
        top.addWidget(self.level)
        lay.addLayout(top)
        self.hp = HPBar()
        self.hp.setFixedHeight(16)
        lay.addWidget(self.hp)
        self.balls = PokeballRow(size=11)
        lay.addWidget(self.balls)

    def set_status(self, code: str):
        """Status condition badge as in the game (PAR, QUE, CON, ENV, DOR)."""
        if not code:
            self.status.hide()
            return
        text, color = STATUS_BADGES[code]
        self.status.setText(text)
        self.status.setStyleSheet(f"background-color: {color}; color: white; font-weight: 800; font-size: 10px;"
                                  f" border-radius: 6px; padding: 1px 6px;")
        self.status.show()


STATUS_BADGES = {"par": ("PAR", "#c8a400"), "brn": ("QUE", "#e0552d"), "frz": ("CON", "#4fb3d9"),
                 "psn": ("ENV", "#9b4fc4"), "tox": ("ENV", "#7a2fa8"), "slp": ("DOR", "#7d869c")}
STATUS_TEXT = {"par": "paralizado", "brn": "quemado", "frz": "congelado", "psn": "envenenado",
               "tox": "gravemente envenenado", "slp": "dormido"}
STAT_TEXT = {"atk": "Ataque", "def": "Defensa", "spa": "At. Esp.", "spd": "Def. Esp.", "spe": "Velocidad",
             "acc": "precisión", "eva": "evasión"}
ABILITY_TEXT = {"intimidate": "Intimidación", "synchronize": "Sincronía", "static": "Elec. Estática",
                "flame-body": "Cuerpo Llama", "poison-point": "Punto Tóxico", "effect-spore": "Efecto Espora"}
EFFECTIVENESS_TEXT = {0: "", 1: " No es muy eficaz…", 2: " No es muy eficaz…", 8: " ¡Es muy eficaz!",
                      16: " ¡Es muy eficaz!"}


class BattleScene(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 360)
        self.bg = QPixmap(str(gd.IMAGES / "forest.png"))
        self.rival_sprite = AnimatedSprite(parent=self)
        self.player_sprite = AnimatedSprite(parent=self)
        self.rival_plate = InfoPlate(self)
        self.player_plate = InfoPlate(self)
        self.effect = EffectOverlay(self)
        self.message = QLabel("", self)
        self.message.setProperty("role", "message")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.type_badge = QLabel("", self)
        self.type_badge.setAlignment(Qt.AlignCenter)
        self.type_badge.hide()
        self.result_badge = QLabel("", self)
        self.result_badge.setAlignment(Qt.AlignCenter)
        self.result_badge.hide()
        self.turn_badge = QLabel("", self)
        self.turn_badge.setAlignment(Qt.AlignCenter)
        self.turn_badge.setStyleSheet(f"background-color: rgba(10,16,30,210); color: {theme.TEXT}; font-weight: 800;"
                                      f" font-size: 13px; letter-spacing: 1px; border: 1px solid {theme.BORDER};"
                                      f" border-radius: 13px;")
        self.turn_badge.hide()

    def set_turn(self, turn: int):
        if turn > 0:
            self.turn_badge.setText(f"TURNO {turn}")
            self.turn_badge.show()
            self.turn_badge.raise_()
        else:
            self.turn_badge.hide()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if not self.bg.isNull():
            scaled = self.bg.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
        p.fillRect(self.rect(), QColor(5, 10, 20, 70))
        w, h = self.width(), self.height()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 70))
        p.drawEllipse(QRectF(w * 0.58, h * 0.40, w * 0.32, h * 0.08))
        p.drawEllipse(QRectF(w * 0.06, h * 0.78, w * 0.40, h * 0.09))

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        self.rival_sprite.setGeometry(int(w * 0.57), int(h * 0.02), int(w * 0.34), int(h * 0.44))
        self.player_sprite.setGeometry(int(w * 0.04), int(h * 0.30), int(w * 0.44), int(h * 0.56))
        self.rival_plate.setGeometry(int(w * 0.04), int(h * 0.05), int(min(w * 0.42, 360)), 86)
        pw = int(min(w * 0.42, 360))
        self.player_plate.setGeometry(int(w * 0.96) - pw, int(h * 0.55), pw, 86)
        self.message.setGeometry(int(w * 0.03), int(h * 0.86), int(w * 0.94), int(h * 0.12))
        self.type_badge.setGeometry(int(w * 0.03) + 14, int(h * 0.86) - 13, 104, 24)
        self.result_badge.setGeometry(int(w * 0.30), int(h * 0.36), int(w * 0.40), 60)
        self.turn_badge.setGeometry(w - 126, 12, 112, 28)
        self.effect.stop()
        super().resizeEvent(event)

    def sprite(self, side: str) -> AnimatedSprite:
        return self.player_sprite if side == "player" else self.rival_sprite

    def play_effect(self, type_name: str | None, target: str, duration_ms: int):
        """Animation of the move type over the target Pokémon."""
        frames = effects.library().frames(type_name)
        if not frames:
            return
        rect = self.sprite(target).geometry()
        size = int(min(rect.width(), rect.height()) * (0.9 if target == "player" else 1.1))
        center = rect.center()
        center.setY(center.y() + int(rect.height() * 0.15))  # normalized sprites stand at the bottom of their canvas
        self.effect.play(frames, center, size, duration_ms)

    def set_move_type(self, type_name: str | None):
        """Type badge over the message box and message border in the type colour."""
        if not type_name:
            self.type_badge.hide()
            self.message.setStyleSheet("")
            return
        color = effects.TYPE_COLORS.get(type_name, theme.BORDER)
        self.type_badge.setText(effects.TYPE_NAMES.get(type_name, type_name).upper())
        self.type_badge.setStyleSheet(f"background-color: {color}; color: white; font-weight: 800; font-size: 11px;"
                                      f" border-radius: 12px; letter-spacing: 1px;")
        self.type_badge.show()
        self.type_badge.raise_()
        self.message.setStyleSheet(f"border-color: {color};")


class BattlePanel(QWidget):
    EFFECT_MS = 560  # duration of a move animation at x1

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)
        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(section_label("Escenario mostrado"))
        self.scenario = QComboBox()
        self.scenario.setToolTip(
            "Cada combate se juega en 100 escenarios aleatorios fijos (los mismos del entrenamiento).\n"
            "Peor / mediano / mejor: escenario con el fitness total más alto, mediano o más bajo de los 100.\n"
            "Determinista: el simulador de la memoria (tirada media, sin fallos ni efectos).")
        left.addWidget(self.scenario)
        self.list = QListWidget()
        self.list.setFixedWidth(240)
        left.addWidget(self.list, 1)
        lay.addLayout(left)
        right = QVBoxLayout()
        right.setSpacing(8)
        teams = QHBoxLayout()
        self.player_team = TeamStrip(size=40, animated=False)
        self.rival_team = TeamStrip(size=40, animated=False)
        teams.addWidget(section_label("Tu equipo"))
        teams.addWidget(self.player_team, 1)
        vs = QLabel("VS")
        vs.setStyleSheet(f"color: {theme.ACCENT}; font-weight: 900; font-size: 16px; padding: 0 10px;")
        teams.addWidget(vs)
        teams.addWidget(self.rival_team, 1)
        teams.addWidget(section_label("Rival"))
        right.addLayout(teams)
        self.plan_label = QLabel("")
        self.plan_label.setProperty("role", "muted")
        self.plan_label.setWordWrap(True)
        right.addWidget(self.plan_label)
        self.scene = BattleScene()
        right.addWidget(self.scene, 1)
        ctrl = QHBoxLayout()
        self.btn_first = QPushButton("⏮")
        self.btn_prev = QPushButton("◀")
        self.btn_play = QPushButton("▶  Reproducir")
        self.btn_play.setProperty("kind", "primary")
        self.btn_next = QPushButton("▶")
        self.btn_last = QPushButton("⏭")
        for b in (self.btn_first, self.btn_prev, self.btn_play, self.btn_next, self.btn_last):
            ctrl.addWidget(b)
        self.slider = QSlider(Qt.Horizontal)
        ctrl.addWidget(self.slider, 1)
        self.speed = QComboBox()
        for label, factor in (("x0.5", 0.5), ("x1", 1.0), ("x2", 2.0), ("x4", 4.0)):
            self.speed.addItem(label, factor)
        self.speed.setCurrentIndex(1)
        ctrl.addWidget(self.speed)
        self.step_label = QLabel("")
        self.step_label.setProperty("role", "muted")
        self.step_label.setMinimumWidth(130)
        ctrl.addWidget(self.step_label)
        right.addLayout(ctrl)
        lay.addLayout(right, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.battles = []
        self.genome = None
        self.current = 0
        self.step = 0
        self._token = 0  # invalidates delayed hit callbacks when the user jumps elsewhere
        self.list.currentRowChanged.connect(self.select_battle)
        self.btn_first.clicked.connect(lambda: self.goto(0))
        self.btn_prev.clicked.connect(lambda: self.goto(self.step - 1))
        self.btn_next.clicked.connect(lambda: self.goto(self.step + 1, animate=True))
        self.btn_last.clicked.connect(lambda: self.goto(len(self._events()) - 1))
        self._steps = []
        self.btn_play.clicked.connect(self._toggle_play)
        self.slider.valueChanged.connect(lambda v: self.goto(v) if v != self.step else None)

    def set_battles(self, genome, battles: list, keep_battle: bool = False):
        previous = self.current if keep_battle else 0
        self.stop()
        self.genome = genome
        self.battles = battles
        self.list.blockSignals(True)
        self.list.clear()
        for i, b in enumerate(battles):
            extra = f" · {b.switches} {'cambio' if b.switches == 1 else 'cambios'}" if b.switches else ""
            text = f"{'✔' if b.won else '✖'}  {gd.TRAINER_NAMES[i]}\n      {b.hp_lost} % HP · {b.turns} turnos{extra}"
            item = QListWidgetItem(text)
            item.setForeground(QColor(theme.TEXT if b.won else "#ff8a8a"))
            self.list.addItem(item)
        self.list.blockSignals(False)
        self.list.setCurrentRow(previous)
        self.select_battle(previous)

    def _events(self):
        """Playback steps: every event except the turn markers (the turn is shown in the scene),
        keeping the first one to show the initial state."""
        return self._steps

    @staticmethod
    def _fainted(ev: dict):
        return ({i for i, hp in enumerate(ev["p_hps"]) if hp <= 0},
                {i for i, hp in enumerate(ev["r_hps"]) if hp <= 0})

    def _plan_text(self, index: int, b) -> str:
        plan = self.genome[2][index] if len(self.genome) > 2 else None
        base_team = self.genome[1][index]
        rival = gd.rival_team(index, self.genome[0][0])
        parts = []
        for j, pkm in enumerate(plan or []):
            if pkm is not None and pkm in base_team and j < len(rival):
                evolved = b.team[base_team.index(pkm)]
                parts.append(f"{rival[j][2]} → {gd.pkm_name(evolved)}")
        if not parts:
            return "Sin plan de cambios: cada Pokémon combate hasta debilitarse y sale el siguiente en orden."
        return "Plan de cambios: " + " · ".join(parts)

    def _speed(self) -> float:
        return self.speed.currentData() or 1.0

    def select_battle(self, index: int):
        if index < 0 or index >= len(self.battles):
            return
        self.stop()
        self.current = index
        b = self.battles[index]
        self.player_team.set_team(b.team, levels=[b.level_cap] * len(b.team))
        rival = gd.rival_team(index, self.genome[0][0])
        self.rival_team.set_team([r[0] for r in rival], levels=[r[1] for r in rival])
        self.plan_label.setText(self._plan_text(index, b))
        self._steps = [e for i, e in enumerate(b.events) if e["type"] != "turn" or i == 0]
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(self._steps) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.goto(0)

    def goto(self, step: int, animate: bool = False):
        events = self._events()
        if not events:
            return
        step = max(0, min(step, len(events) - 1))
        self.step = step
        self.slider.blockSignals(True)
        self.slider.setValue(step)
        self.slider.blockSignals(False)
        self._render(events[step], animate)
        self.scene.set_turn(events[step].get("turn", 0))
        self.step_label.setText(f"Paso {step + 1}/{len(events)}")

    @staticmethod
    def _is_attack(ev: dict) -> bool:
        return ev["type"] == "action" and ev["kind"] == "attack"

    def _render(self, ev: dict, animate: bool):
        self._token += 1
        b = self.battles[self.current]
        sc = self.scene
        sc.effect.stop()
        duration = int(450 / self._speed())
        if ev["type"] == "end":
            sc.set_move_type(None)
            last = next((e for e in reversed(b.events) if e["type"] != "end"), None)
            if last:
                self._render_state(last, b, animate=False, duration=duration)
            p_fainted, r_fainted = self._fainted(ev)
            sc.player_plate.balls.set_state(len(b.team), p_fainted)
            sc.rival_plate.balls.set_state(len(b.rival_team), r_fainted)
            self.player_team.set_fainted(p_fainted)
            self.rival_team.set_fainted(r_fainted)
            if ev["won"]:
                txt, color = "¡VICTORIA!", theme.GREEN
                msg = f"¡Has ganado! Tu equipo ha perdido {ev['hp_lost']} % de HP en total."
            elif ev["timed_out"]:
                txt, color = "TIEMPO AGOTADO", theme.ORANGE
                msg = f"El combate superó {bs.MAX_TURNS} turnos sin terminar: cuenta como derrota."
            else:
                txt, color = "DERROTA", theme.RED
                msg = "Todo tu equipo se ha debilitado."
            sc.message.setText(msg)
            sc.result_badge.setText(txt)
            sc.result_badge.setStyleSheet(f"background-color: rgba(10,16,30,220); color: {color}; font-size: 26px;"
                                          f" font-weight: 900; border: 2px solid {color}; border-radius: 14px;")
            sc.result_badge.show()
            return
        sc.result_badge.hide()
        if ev["type"] == "switch" and animate:
            sc.set_move_type(None)
            token = self._token
            sc.player_sprite.set_sprite(gd.sprite_path(ev["from_id"], "back", normalized=True))
            sc.player_sprite.blink(times=2, interval=max(30, int(60 / self._speed())), end_faded=True)

            def bring_in():
                if token == self._token:
                    self._render_state(ev, b, animate=True, duration=duration)

            QTimer.singleShot(int(380 / self._speed()), bring_in)
            sc.message.setText(self._describe(ev))
            return
        if ev["type"] in ("residual", "confusion_hit", "recoil") and animate:
            side = ev["side"]
            self._render_state(ev, b, animate=True, duration=duration, target=side, hit=True)
            sc.set_move_type(None)
            sc.message.setText(self._describe(ev))
            return
        attack = self._is_attack(ev)
        mtype = effects.move_type(ev["move"]) if attack else None
        sc.set_move_type(mtype)
        if attack and animate:
            # 1) the type animation plays over the target, 2) the hit lands: HP drops and the target blinks
            target = "rival" if ev["side"] == "player" else "player"
            effect_ms = int(self.EFFECT_MS / self._speed())
            delay = 0
            if not ev.get("missed"):
                sc.play_effect(mtype, target, effect_ms)
                delay = int(effect_ms * 0.55)
            self._render_state(ev, b, animate=True, duration=duration, hit_delay=delay, target=target,
                               hit=ev["damage"] > 0)
        else:
            self._render_state(ev, b, animate, duration)
        sc.message.setText(self._describe(ev, mtype))

    def _render_state(self, ev: dict, b, animate: bool, duration: int, hit_delay: int = 0, target: str | None = None,
                      hit: bool = False):
        sc = self.scene
        p_id = ev["p_id"]
        r_id = gd.NAME_TO_ID.get(ev["r_species"], 0)
        r_level = b.rival_team[ev["r_idx"]]["level"]
        sc.player_sprite.set_sprite(gd.sprite_path(p_id, "back", normalized=True))
        sc.rival_sprite.set_sprite(gd.sprite_path(r_id, "front", normalized=True))
        sc.player_plate.name.setText(gd.pkm_name(p_id))
        sc.player_plate.level.setText(f"Nv {b.level_cap}")
        sc.rival_plate.name.setText(ev["r_species"])
        sc.rival_plate.level.setText(f"Nv {r_level}")
        hp = {"player": ev["p_hp"], "rival": ev["r_hp"]}
        sc.player_plate.set_status(ev.get("p_status", "") if hp["player"] > 0 else "")
        sc.rival_plate.set_status(ev.get("r_status", "") if hp["rival"] > 0 else "")
        token = self._token

        def land():
            if token != self._token:
                return
            sc.player_plate.hp.set_value(hp["player"], animate=animate, duration=duration)
            sc.rival_plate.hp.set_value(hp["rival"], animate=animate, duration=duration)
            p_fainted, r_fainted = self._fainted(ev)
            sc.player_plate.balls.set_state(len(b.team), p_fainted)
            sc.rival_plate.balls.set_state(len(b.rival_team), r_fainted)
            self.player_team.set_fainted(p_fainted)
            self.rival_team.set_fainted(r_fainted)
            for side in ("player", "rival"):
                if side == target and hit:
                    sc.sprite(side).blink(times=3, interval=max(30, int(70 / self._speed())),
                                          end_faded=hp[side] <= 0)
                else:
                    sc.sprite(side).set_faded(hp[side] <= 0)

        if hit_delay > 0:
            sc.sprite(target).set_faded(False)  # still standing until the attack lands
            QTimer.singleShot(hit_delay, land)
        else:
            land()

    def _describe(self, ev: dict, mtype: str | None = None) -> str:
        if ev["type"] == "turn":
            return f"¡Comienza el combate contra {gd.TRAINER_NAMES[self.current]}!"
        if ev["type"] == "switch":
            return f"¡{gd.pkm_name(ev['from_id'])}, vuelve! ¡Adelante, {gd.pkm_name(ev['to_id'])}!"
        if ev["type"] == "send_out":
            if ev["side"] == "player":
                return f"¡Adelante, {gd.pkm_name(ev['p_id'])}!"
            return f"El rival envía a {ev['r_species']}."
        if ev["type"] == "heal":
            return f"{ev['pokemon']} recupera {ev['amount']} % de HP."
        text = self._describe_effect(ev)
        if text is not None:
            return text
        if ev["type"] == "end":
            return "¡Victoria!" if ev.get("won") else "Derrota."
        who = ev["attacker"]
        move = gd.format_move(ev["move"])
        kind = ev["kind"]
        if kind == "recharge":
            return f"{who} necesita recargarse tras Hyper Beam."
        if kind == "solar-beam":
            return f"{who} está absorbiendo luz (Solar Beam)…"
        if kind == "dig":
            return f"{who} se ha escondido bajo tierra (Dig)…"
        if kind == "fly":
            return f"{who} ha volado muy alto (Fly)…"
        if kind == "dive":
            return f"{who} se ha sumergido (Dive)…"
        if kind in ("bounce", "sky-attack", "razor-wind"):
            return f"{who} se prepara para {move}…"
        if ev.get("failed"):
            return f"{who} usa {move}… ¡pero falla!"
        if ev.get("missed"):
            if "crit" in ev:  # random battles: a real miss
                return f"{who} usa {move}… ¡pero falla!"
            return f"{who} usa {move}… ¡pero {ev['defender']} no está a su alcance!"
        if ev["power"] == 0:
            return f"{who} usa {move}… ¡no afecta a {ev['defender']}!"
        text = f"{who} usa {move}: {ev['damage']} % de daño."
        if ev.get("hits", 1) > 1:
            text += f" ¡{ev['hits']} golpes!"
        if ev.get("crit"):
            text += " ¡Golpe crítico!"
        text += EFFECTIVENESS_TEXT.get(ev.get("effectiveness", 4), "")
        if ev["fainted"]:
            text += f" ¡{ev['defender']} se debilita!"
        return text

    @staticmethod
    def _describe_effect(ev: dict):
        """Messages of the random battles (status, stats, abilities...); None for other events."""
        t, who = ev["type"], ev.get("pokemon", "")
        if t == "status":
            return f"¡{who} está {STATUS_TEXT[ev['status']]}!"
        if t == "stat":
            stat, delta = STAT_TEXT[ev["stat"]], ev["delta"]
            if ev.get("blocked"):
                return f"¡La habilidad de {who} impide que baje su {stat}!"
            if not ev.get("applied"):
                return f"¡{stat} de {who} no puede {'subir' if delta > 0 else 'bajar'} más!"
            how = {1: "sube", 2: "sube mucho", -1: "baja", -2: "baja mucho"}.get(delta, "cambia")
            return f"¡{stat} de {who} {how}!"
        if t == "ability":
            name = ABILITY_TEXT.get(ev["ability"], ev["ability"])
            if ev["ability"] == "intimidate":
                if not ev.get("effective", True):
                    return f"¡{name} de {who} no afecta a {ev['target']}!"
                return f"¡{name} de {who} baja el Ataque de {ev['target']}!"
            return f"¡{name} de {who}!"
        if t == "cant_move":
            return {"sleep": f"{who} está dormido.", "frozen": f"¡{who} está congelado!",
                    "paralysis": f"¡{who} está paralizado! No se puede mover.",
                    "flinch": f"¡{who} retrocede y no ataca!",
                    "recharge": f"{who} necesita recargarse."}.get(ev["reason"], f"{who} no puede moverse.")
        if t == "wake":
            return f"¡{who} se ha despertado!"
        if t == "thaw":
            return f"¡{who} se ha descongelado!"
        if t == "confuse":
            return f"¡{who} está confuso!" + (" (agotado tras el ataque)" if ev.get("cause") == "fatigue" else "")
        if t == "confused":
            return f"{who} está confuso…"
        if t == "confusion_hit":
            text = f"¡Está tan confuso que se hiere a sí mismo! ({ev['damage']} %)"
            return text + (f" ¡{who} se debilita!" if ev.get("fainted") else "")
        if t == "confusion_end":
            return f"¡{who} ya no está confuso!"
        if t == "residual":
            cause = {"brn": "la quemadura", "psn": "el veneno", "tox": "el veneno", "trap": "el atrapamiento"}.get(
                ev["cause"], ev["cause"])
            text = f"{who} sufre por {cause} ({ev['amount']} %)."
            return text + (f" ¡{who} se debilita!" if ev.get("fainted") else "")
        if t == "trap":
            return f"¡{who} ha quedado atrapado por {gd.format_move(ev['move'])}!"
        if t == "trap_end":
            return f"¡{who} se ha liberado!"
        if t == "cure":
            return {"shed-skin": f"¡{who} se cura con Mudar!", "natural-cure": f"¡{who} se cura con Cura Natural!",
                    "smelling-salt": f"¡{who} ya no está paralizado!"}.get(ev["cause"], f"¡{who} se ha curado!")
        if t == "recoil":
            return f"¡{who} se ha hecho daño con el retroceso!"
        return None

    def _interval(self) -> int:
        """Time the current step stays on screen during autoplay (attacks need room for the animation)."""
        events = self._events()
        ev = events[self.step] if events else None
        kind = ev["type"] if ev is not None else None
        base = {"switch": 1100, "send_out": 800, "turn": 900, "stat": 800, "status": 900, "ability": 1000}.get(kind, 900)
        if ev is not None and self._is_attack(ev):
            base = 1250
        return int(base / self._speed())

    def _toggle_play(self):
        if self.timer.isActive():
            self.stop()
        else:
            if self.step >= len(self._events()) - 1:
                self.goto(0)
            self.timer.start(self._interval())
            self.btn_play.setText("⏸  Pausa")

    def stop(self):
        self.timer.stop()
        self.btn_play.setText("▶  Reproducir")

    def _tick(self):
        if self.step >= len(self._events()) - 1:
            self.stop()
            return
        self.goto(self.step + 1, animate=True)
        self.timer.setInterval(self._interval())


# --------------------------------------------------------------------------- summary

RANDOM_SCENARIOS = 100  # the viewer uses the same scenarios as training (100, seed 0)
RANDOM_SEED = 0


def _thousands(x) -> str:
    return f"{float(x):,.0f}".replace(",", ".")


def _pct(p: float) -> str:
    if p >= 1:
        return "100 %"
    return f"{p * 100:.1f} %".replace(".", ",")


class RandomScorer:
    """Random battles of an individual in the 100 scenarios of training (lazy: loads the move
    table the first time)."""

    def __init__(self):
        self.error = None

    def score(self, genome):
        try:
            import stochastic_battle as sb

            return sb.score_genomes([genome], RANDOM_SCENARIOS, seed=RANDOM_SEED, chunk=RANDOM_SCENARIOS)
        except (FileNotFoundError, KeyError) as exc:
            self.error = str(exc).strip("'\"")
            return None

    def replay(self, genome, scenario: int):
        """The 21 battles of one scenario with every event."""
        import stochastic_battle as sb

        return sb.replay(genome, scenario, seed=RANDOM_SEED)


class SummaryPanel(QWidget):
    battle_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)
        self.bars = charts.BattleBars()
        self.bars.setMinimumWidth(380)
        lay.addWidget(self.bars, 2)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Combate", "Resultado", "HP perdido", "Turnos", "Cambios",
                                              "P(victoria)", "Equipo (orden de salida)"])
        self.table.horizontalHeaderItem(5).setToolTip(
            f"Probabilidad de ganar el combate en los {RANDOM_SCENARIOS} escenarios aleatorios (los del entrenamiento).\n"
            "HP perdido, turnos y cambios son la media de esos escenarios.")
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(6, QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(40)
        lay.addWidget(self.table, 3)
        self.table.cellDoubleClicked.connect(lambda r, _c: self.battle_selected.emit(r))

    def set_data(self, genome, battles: list, per_battle_values: list, scores=None):
        self.bars.set_values(per_battle_values)
        self.table.setRowCount(len(battles))
        for i, b in enumerate(battles):
            p = None if scores is None else float(scores.win_rates[0, i])
            if scores is None:
                result = "Victoria" if b.won else ("Tiempo agotado" if b.timed_out else "Derrota")
                hp, turns, sw = f"{b.hp_lost} %", str(b.turns), str(b.switches) if b.switches else "-"
            else:
                result = "Victoria" if p >= 0.995 else ("Victoria casi siempre" if p >= 0.5 else "Derrota casi siempre")
                hp = f"{scores.battle_hp_lost[0, i]:.0f} %"
                turns = f"{scores.battle_turns[0, i]:.1f}".replace(".", ",")
                sw_mean = scores.battle_switches[0, i]
                sw = f"{sw_mean:.1f}".replace(".", ",") if sw_mean else "-"
            cells = [gd.TRAINER_NAMES[i], result, hp, turns, sw, "-" if p is None else _pct(p)]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 1:
                    good = b.won if p is None else p >= 0.5
                    item.setForeground(QColor(theme.GREEN if good else theme.RED))
                if col == 5 and p is not None:
                    item.setForeground(QColor(theme.GREEN if p >= 0.995 else (theme.YELLOW if p >= 0.9 else theme.RED)))
                if col in (2, 3, 4, 5):
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, col, item)
            strip = TeamStrip(size=34, animated=False)
            strip.lay.setContentsMargins(4, 0, 4, 0)
            strip.set_team(b.team, tooltips=[f"{gd.pkm_name(p)} (Nv {b.level_cap})" for p in b.team])
            self.table.setCellWidget(i, 6, strip)


# --------------------------------------------------------------------------- view


class IndividualView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.evaluator = Evaluator()
        self.random_scorer = RandomScorer()
        self.genome = None
        self.evaluation = None
        self.source = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        bar = QHBoxLayout()
        self.title = QLabel("Visor de individuos")
        self.title.setProperty("role", "title")
        bar.addWidget(self.title)
        self.subtitle = QLabel("")
        self.subtitle.setProperty("role", "muted")
        bar.addWidget(self.subtitle)
        bar.addStretch(1)
        self.btn_examples = QToolButton()
        self.btn_examples.setText("Ejemplos ▾")
        self.btn_examples.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self)
        for ex in gd.example_individuals():
            menu.addAction(ex["name"], lambda ex=ex: self.load_genome(ex["genome"], ex["name"]))
        self.btn_examples.setMenu(menu)
        self.btn_open = QPushButton("Abrir JSON…")
        self.btn_paste = QPushButton("Pegar genotipo…")
        self.btn_export = QPushButton("Exportar JSON…")
        self.btn_export.setEnabled(False)
        for b in (self.btn_examples, self.btn_open, self.btn_paste, self.btn_export):
            bar.addWidget(b)
        root.addLayout(bar)

        tiles = QHBoxLayout()
        self.tile_fit = StatTile("Fitness (menor es mejor)")
        self.tile_wins = StatTile("Combates ganados")
        self.tile_hp = StatTile("% HP perdido en total")
        self.tile_catch = StatTile("Capturas")
        for t in (self.tile_fit, self.tile_wins, self.tile_hp, self.tile_catch):
            tiles.addWidget(t, 1)
        status = Card(None)
        self.chip_valid = Chip("-")
        self.chip_game = Chip("-")
        self.chip_starter = Chip("-")
        status.body.addWidget(self.chip_valid)
        status.body.addWidget(self.chip_game)
        status.body.addWidget(self.chip_starter)
        self.chip_switch = Chip("-")
        status.body.addWidget(self.chip_switch)
        self.chip_random = Chip("-")
        status.body.addWidget(self.chip_random)
        tiles.addWidget(status, 1)
        root.addLayout(tiles)

        self.stack = QStackedWidget()
        empty = Card(None)
        msg = QLabel("Carga un individuo desde un entrenamiento, el Hall of Fame de una ejecución, un archivo JSON, "
                     "pegando su genotipo o con los ejemplos.")
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignCenter)
        msg.setProperty("role", "muted")
        empty.body.addStretch(1)
        empty.body.addWidget(msg)
        empty.body.addStretch(1)
        self.stack.addWidget(empty)
        self.tabs = QTabWidget()
        self.summary = SummaryPanel()
        self.map_panel = MapPanel()
        self.battle_panel = BattlePanel()
        self.tabs.addTab(self.summary, "Resumen")
        self.tabs.addTab(self.map_panel, "Mapa de capturas")
        self.tabs.addTab(self.battle_panel, "Combates")
        self.help_panel = HowItWorksPanel()
        self.tabs.addTab(self.help_panel, "¿Cómo funciona?")
        self.stack.addWidget(self.tabs)
        root.addWidget(self.stack, 1)

        self.btn_open.clicked.connect(self._open_file)
        self.btn_paste.clicked.connect(self._paste)
        self.btn_export.clicked.connect(self._export)
        self.summary.battle_selected.connect(self._goto_battle)
        self.battle_panel.scenario.currentIndexChanged.connect(self._show_scenario)
        self.scores, self._det_battles, self._replays = None, [], {}

    def _goto_battle(self, index: int):
        self.tabs.setCurrentWidget(self.battle_panel)
        self.battle_panel.list.setCurrentRow(index)

    # ------------------------------------------------------------------ loading

    def load_genome(self, genome, source: str = "") -> bool:
        try:
            genome = load_individual(genome)
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Individuo no válido", str(exc))
            return False
        try:
            evaluation = self.evaluator.evaluate(genome)
            battles = simulate_detailed(genome) if bs.teams_use_available_pokemon(genome) else []
        except bs.DamageServerError as exc:
            QMessageBox.critical(self, "Faltan datos de daño", str(exc))
            return False
        except (KeyError, IndexError) as exc:
            QMessageBox.warning(self, "Individuo no válido", f"El genotipo contiene datos desconocidos: {exc}")
            return False
        self.genome, self.evaluation, self.source = genome, evaluation, source
        self.subtitle.setText(source)
        scores = self.random_scorer.score(genome) if battles else None
        self.scores, self._det_battles, self._replays = scores, battles, {}
        self._fill_header(genome, evaluation)
        self._fill_random(scores, evaluation)
        self.map_panel.set_individual(genome)
        self.help_panel.set_individual(genome, scores)
        if battles:
            if scores is not None:
                values = [round(float(h)) if w >= 0.5 else -1
                          for h, w in zip(scores.battle_hp_lost[0], scores.win_rates[0])]
                self.summary.set_data(genome, battles, values, scores)
            else:
                self.summary.set_data(genome, battles, evaluation.per_battle_values())
            self._fill_scenarios(scores, evaluation)
            self._show_scenario()
        self.stack.setCurrentIndex(1)
        self.btn_export.setEnabled(True)
        return True

    def _fill_scenarios(self, scores, evaluation):
        """Scenario selector of the battle tab: worst, median and best of the 100 random scenarios
        (the same ones used in training) and the deterministic simulator."""
        combo = self.battle_panel.scenario
        combo.blockSignals(True)
        combo.clear()
        if scores is not None:
            fit = scores.fitness[0]
            order = np.argsort(fit, kind="stable")
            picks = (("Mediano", int(order[len(order) // 2])), ("Peor", int(order[-1])), ("Mejor", int(order[0])))
            for label, sc in picks:
                lost = [gd.TRAINER_NAMES[t] for t in range(ops.N_TRAINERS)
                        if fit[sc] >= bs.LOSS_PENALTY and self._battle_lost(sc, t)]
                extra = f" · pierde {', '.join(lost)}" if lost else ""
                combo.addItem(f"{label} · nº {sc + 1} · fitness {_thousands(fit[sc])}{extra}", sc)
        combo.addItem(f"Determinista (memoria) · fitness {_thousands(evaluation.raw_fitness)}", -1)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _battle_lost(self, scenario: int, trainer: int) -> bool:
        return not self._replay(scenario)[trainer].won

    def _replay(self, scenario: int):
        if scenario not in self._replays:
            self._replays[scenario] = self.random_scorer.replay(self.genome, scenario)
        return self._replays[scenario]

    def _show_scenario(self, *_):
        sc = self.battle_panel.scenario.currentData()
        if sc is None:
            return
        battles = self._det_battles if sc < 0 else self._replay(sc)
        self.battle_panel.set_battles(self.genome, battles, keep_battle=True)

    def _fill_header(self, genome, ev):
        sc = self.scores
        if sc is not None:  # random battles: the values of training (mean of the 100 scenarios)
            p = float(sc.p_complete[0])
            self.tile_fit.caption.setText(f"Fitness (media de {RANDOM_SCENARIOS} escenarios)")
            self.tile_fit.set(_thousands(sc.mean[0]), theme.YELLOW)
            self.tile_wins.caption.setText("Completa la ruta (21/21)")
            self.tile_wins.set(_pct(p), theme.GREEN if p >= 0.99 else (theme.ORANGE if p >= 0.8 else theme.RED))
            self.tile_hp.caption.setText("% HP perdido en total (media)")
            self.tile_hp.set(_thousands(sc.hp_lost[0].mean()))
        else:
            fit = ev.raw_fitness
            self.tile_fit.caption.setText("Fitness (menor es mejor)")
            self.tile_fit.set("∞" if fit == float("inf") else _thousands(fit), theme.YELLOW)
            self.tile_wins.caption.setText("Combates ganados")
            self.tile_hp.caption.setText("% HP perdido en total")
            if ev.per_battle:
                self.tile_wins.set(f"{ev.wins}/21", theme.GREEN if ev.wins == 21 else theme.RED)
                self.tile_hp.set(_thousands(ev.hp_lost))
            else:
                self.tile_wins.set("-")
                self.tile_hp.set("-")
        caught = sum(1 for p in genome[0] if p is not None)
        self.tile_catch.set(f"{caught}/{ops.N_ROUTES}")
        if ev.valid:
            self.chip_valid.set("✔ Cumple las reglas nuzlocke", "good")
            self.chip_valid.setToolTip("Sin capturas repetidas y equipos con Pokémon ya capturados.")
        else:
            self.chip_valid.set(f"✖ {len(ev.violations)} infracciones", "bad")
            self.chip_valid.setToolTip("\n".join(ev.violations[:20]))
        if ev.per_battle and sc is not None:
            self.chip_game.set(f"Determinista: fitness {_thousands(ev.raw_fitness)} · {ev.wins}/21", "neutral")
            self.chip_game.setToolTip("Simulador de la memoria: tirada media, sin fallos, críticos ni efectos.")
        elif ev.per_battle:
            self.chip_game.set("✔ Completa el juego" if ev.losses == 0 else f"✖ Pierde {ev.losses} combates",
                               "good" if ev.losses == 0 else "warn")
        else:
            self.chip_game.set("Sin simular (equipos no disponibles)", "bad")
        self.chip_starter.set(f"Inicial: {gd.pkm_name(genome[0][0])}", "accent")
        n_switch = sum(b.switches for b in ev.per_battle)
        self.chip_switch.set(f"{n_switch} cambios en 21 combates" if n_switch else "Sin cambios de Pokémon", "neutral")

    def _fill_random(self, scores, ev=None):
        if scores is None:
            self.chip_random.set("Combates aleatorios: no disponibles", "neutral")
            self.chip_random.setToolTip(self.random_scorer.error or "El individuo no se puede simular.")
            return
        fit = scores.fitness[0]
        self.chip_random.set(f"{RANDOM_SCENARIOS} escenarios · mejor {_thousands(fit.min())} · peor {_thousands(fit.max())}",
                             "accent")
        self.chip_random.setToolTip(
            f"Cada combate se juega en {RANDOM_SCENARIOS} escenarios aleatorios fijos (los mismos del entrenamiento):\n"
            "precisión, críticos, tiradas de daño, efectos secundarios, estados, cambios de características y\n"
            "habilidades de la 3.ª generación. 'Completa la ruta' = gana los 21 combates en el mismo escenario.\n"
            f"Fitness medio {scores.mean[0]:.0f} ± {scores.stderr[0]:.0f} (error estándar).")

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir individuo", str(bs.ROOT), "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(open(path, encoding="utf-8").read())
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "No se pudo leer el archivo", str(exc))
            return
        if isinstance(data, list) and data and isinstance(data[0], dict):  # hall_of_fame.json
            data = data[0]
        self.load_genome(data, path)

    def _paste(self):
        text, ok = QInputDialog.getMultiLineText(
            self, "Pegar genotipo", "Genotipo [[capturas], [equipos]] o JSON de un individuo:")
        if not ok or not text.strip():
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(text.strip())
            except (ValueError, SyntaxError) as exc:
                QMessageBox.warning(self, "Formato no reconocido", str(exc))
                return
        self.load_genome(data, "Genotipo pegado")

    def _export(self):
        if self.genome is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar individuo", str(bs.ROOT / "individuo.json"),
                                              "JSON (*.json)")
        if not path:
            return
        data = individual_to_json(self.genome, self.evaluation, source=self.source)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
