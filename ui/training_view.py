"""Training view: configure and run the GA, watch the evolution live."""
from __future__ import annotations

import os
import time

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                             QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                             QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

import battle_simulator as bs
from ga import damage_table
from ga.config import PRESET_LABELS, GAConfig, preset
from ui import charts
from ui import gamedata as gd
from ui import theme
from ui.widgets import Card, Chip, StatTile, TeamStrip
from ui.worker import GARunner

HELP = {
    "population": "POB · Número de individuos de la población (constante en cada generación).",
    "generations": "Número de generaciones a ejecutar.",
    "seed": "Semilla del generador aleatorio: con la misma semilla y configuración se obtiene el mismo resultado.",
    "selection": "Ruleta por rango: probabilidad proporcional a la posición (el mejor pesa N, el peor 1).\n"
                 "Torneo: cada padre es el mejor de k individuos al azar.",
    "tournament_k": "Tamaño del torneo: más grande = más presión selectiva.",
    "selected_pct": "SEL · Porcentaje de la población que se selecciona como padres.",
    "cross_pct": "CRUCE · De los padres seleccionados, porcentaje que se cruza; el resto se muta.",
    "mut_catches": "MUT_CAP · Probabilidad por ruta de cambiar la captura (los equipos se reparan).",
    "mut_team": "MUT_EQU · Probabilidad por hueco de cambiar un Pokémon del equipo.",
    "mut_order": "Probabilidad por equipo de intercambiar dos posiciones (orden de salida al combate).",
    "children_per_pair": "Hijos generados por cada pareja de padres.",
    "survivors": "SUP · Individuos únicos (sin clones) que sobreviven a la siguiente generación (élite).",
    "fill_strategy": "Cómo se completa la población tras los supervivientes:\n"
                     "· Aleatorios (memoria): individuos nuevos al azar.\n"
                     "· Hijos + inmigrantes: los mejores hijos y un pequeño porcentaje de aleatorios.",
    "immigrants_pct": "Porcentaje de individuos aleatorios nuevos por generación (estrategia 'hijos').",
    "smart_init": "Los equipos contra los líderes de gimnasio de los individuos aleatorios favorecen tipos con ventaja.",
    "adaptive_modes": "Alterna exploración/explotación según haya mejora (experimento de la memoria).",
    "graded_loss_penalty": "Una derrota suma 10000 + el HP que le queda al rival (en vez de 10000 fijo):\n"
                           "da gradiente entre derrotas ajustadas y derrotas claras.",
    "workers": "Procesos que evalúan la población en paralelo (1 = sin paralelismo).",
    "switching": "Cada individuo lleva un plan de cambios por combate: para cada Pokémon rival, qué miembro\n"
                 "de su equipo debe enfrentarse a él. Cuando sale ese rival, el individuo gasta el turno en\n"
                 "cambiar (el rival golpea al que entra). Sin esta opción no hay cambios, como en la memoria.",
    "mut_plan": "Probabilidad por entrada del plan de cambiar el Pokémon asignado a un rival (o quitar la asignación).",
    "scenarios": "Combates aleatorios: cada combate se juega una vez por escenario y el fitness es la media.\n"
                 "Se simula la 3.ª generación: precisión, críticos, tiradas de daño, golpes múltiples, efectos\n"
                 "secundarios (parálisis, quemadura, congelación, veneno, confusión, retroceso, bajadas de\n"
                 "características), estados, fases de características y habilidades.\n"
                 "Los escenarios son siempre los mismos (los 100 que usa también el visor), así que el fitness\n"
                 "sigue siendo determinista. 0 = simulador determinista de la memoria.\n"
                 "Con 100 escenarios cuesta unas 15-20 veces más que el determinista: usa varios procesos.",
}


def _pct_spin(value=0.0):
    spin = QDoubleSpinBox()
    spin.setRange(0, 100)
    spin.setDecimals(0)
    spin.setSingleStep(5)
    spin.setSuffix(" %")
    spin.setValue(value * 100)
    return spin


class ConfigPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self._loading = False

        top = Card("Configuración")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(7)
        self.algorithm = QComboBox()
        self.algorithm.addItem("Algoritmo genético")
        self.algorithm.addItem("Enfriamiento simulado (experimental)")
        self.algorithm.model().item(1).setEnabled(False)
        self.algorithm.setToolTip("El enfriamiento simulado está en desarrollo (experimental/simulated_annealing.py).")
        form.addRow("Algoritmo", self.algorithm)
        self.preset = QComboBox()
        for key, label in PRESET_LABELS.items():
            self.preset.addItem(label, key)
        self.preset.addItem("Personalizado", "personalizado")
        form.addRow("Preset", self.preset)
        top.body.addLayout(form)
        lay.addWidget(top)

        self.fields = {}

        def group(title):
            card = Card(title)
            f = QFormLayout()
            f.setHorizontalSpacing(10)
            f.setVerticalSpacing(6)
            card.body.addLayout(f)
            lay.addWidget(card)
            return f

        def add(f, key, label, widget):
            widget.setToolTip(HELP.get(key, ""))
            lbl = QLabel(label)
            lbl.setToolTip(HELP.get(key, ""))
            f.addRow(lbl, widget)
            self.fields[key] = widget
            for sig in ("valueChanged", "currentIndexChanged", "toggled"):
                if hasattr(widget, sig):
                    getattr(widget, sig).connect(self._on_edit)
                    break

        f = group("Población")
        pop = QSpinBox()
        pop.setRange(4, 20000)
        pop.setSingleStep(50)
        add(f, "population", "Individuos (POB)", pop)
        gens = QSpinBox()
        gens.setRange(1, 100000)
        gens.setSingleStep(10)
        add(f, "generations", "Generaciones", gens)
        seed_row = QWidget()
        sr = QHBoxLayout(seed_row)
        sr.setContentsMargins(0, 0, 0, 0)
        self.seed_random = QCheckBox("Aleatoria")
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.seed.setToolTip(HELP["seed"])
        self.seed_random.setToolTip(HELP["seed"])
        self.seed_random.toggled.connect(lambda on: self.seed.setEnabled(not on))
        sr.addWidget(self.seed, 1)
        sr.addWidget(self.seed_random)
        f.addRow("Semilla", seed_row)

        f = group("Selección")
        sel = QComboBox()
        sel.addItem("Ruleta por rango", "roulette_rank")
        sel.addItem("Torneo", "tournament")
        add(f, "selection", "Método", sel)
        k = QSpinBox()
        k.setRange(2, 20)
        add(f, "tournament_k", "Tamaño torneo (k)", k)
        add(f, "selected_pct", "Padres (SEL)", _pct_spin())

        f = group("Operadores")
        add(f, "cross_pct", "Cruce (CRUCE)", _pct_spin())
        add(f, "mut_catches", "Mut. capturas", _pct_spin())
        add(f, "mut_team", "Mut. equipos", _pct_spin())
        add(f, "mut_order", "Mut. orden", _pct_spin())
        add(f, "mut_plan", "Mut. plan", _pct_spin())
        cpp = QComboBox()
        cpp.addItem("1", 1)
        cpp.addItem("2", 2)
        add(f, "children_per_pair", "Hijos por pareja", cpp)

        f = group("Reemplazo")
        sup = QSpinBox()
        sup.setRange(1, 20000)
        add(f, "survivors", "Supervivientes (SUP)", sup)
        fill = QComboBox()
        fill.addItem("Aleatorios (memoria)", "random")
        fill.addItem("Hijos + inmigrantes", "offspring")
        add(f, "fill_strategy", "Completar con", fill)
        add(f, "immigrants_pct", "Inmigrantes", _pct_spin())

        f = group("Opciones")
        for key, text in (("switching", "Permitir cambios de Pokémon"), ("smart_init", "Inicialización por tipos"),
                          ("adaptive_modes", "Modos exploración/explotación"),
                          ("graded_loss_penalty", "Penalización graduada")):
            cb = QCheckBox(text)
            cb.setToolTip(HELP[key])
            f.addRow(cb)
            self.fields[key] = cb
            cb.toggled.connect(self._on_edit)
        scen = QSpinBox()
        scen.setRange(0, 20000)
        scen.setSingleStep(100)
        scen.setSpecialValueText("Determinista")
        add(f, "scenarios", "Escenarios aleatorios", scen)
        workers = QSpinBox()
        workers.setRange(1, max(1, os.cpu_count() or 1))
        add(f, "workers", "Procesos", workers)
        lay.addStretch(1)

        # Combo boxes must not ask for the width of their longest option
        for combo in self.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        self.preset.currentIndexChanged.connect(self._on_preset)
        self.fields["selection"].currentIndexChanged.connect(self._sync_enabled)
        self.fields["fill_strategy"].currentIndexChanged.connect(self._sync_enabled)
        self.fields["switching"].toggled.connect(self._sync_enabled)
        self.set_config(preset("memoria"))

    def _sync_enabled(self, *_):
        self.fields["tournament_k"].setEnabled(self.fields["selection"].currentData() == "tournament")
        self.fields["immigrants_pct"].setEnabled(self.fields["fill_strategy"].currentData() == "offspring")
        self.fields["mut_plan"].setEnabled(self.fields["switching"].isChecked())

    def _on_preset(self, _index):
        key = self.preset.currentData()
        if key in PRESET_LABELS and not self._loading:
            self.set_config(preset(key))

    def _on_edit(self, *_):
        if self._loading:
            return
        self._loading = True
        self.preset.setCurrentIndex(self.preset.findData("personalizado"))
        self._loading = False
        self.changed.emit()

    def set_config(self, cfg: GAConfig):
        self._loading = True
        fl = self.fields
        fl["population"].setValue(cfg.population)
        fl["generations"].setValue(cfg.generations)
        for key in ("selected_pct", "cross_pct", "mut_catches", "mut_team", "mut_order", "immigrants_pct", "mut_plan"):
            fl[key].setValue(getattr(cfg, key) * 100)
        fl["tournament_k"].setValue(cfg.tournament_k)
        fl["survivors"].setValue(cfg.survivors)
        fl["selection"].setCurrentIndex(fl["selection"].findData(cfg.selection))
        fl["fill_strategy"].setCurrentIndex(fl["fill_strategy"].findData(cfg.fill_strategy))
        fl["children_per_pair"].setCurrentIndex(fl["children_per_pair"].findData(cfg.children_per_pair))
        for key in ("switching", "smart_init", "adaptive_modes", "graded_loss_penalty"):
            fl[key].setChecked(getattr(cfg, key))
        fl["workers"].setValue(cfg.workers)
        fl["scenarios"].setValue(cfg.scenarios)
        self.seed_random.setChecked(cfg.seed is None)
        if cfg.seed is not None:
            self.seed.setValue(cfg.seed)
        idx = self.preset.findData(cfg.preset)
        self.preset.setCurrentIndex(idx if idx >= 0 else self.preset.findData("personalizado"))
        self._sync_enabled()
        self._loading = False

    def get_config(self) -> GAConfig:
        fl = self.fields
        key = self.preset.currentData()
        base = preset(key) if key in PRESET_LABELS else GAConfig()
        cfg = base.with_overrides(
            population=fl["population"].value(),
            generations=fl["generations"].value(),
            selected_pct=fl["selected_pct"].value() / 100,
            cross_pct=fl["cross_pct"].value() / 100,
            mut_catches=fl["mut_catches"].value() / 100,
            mut_team=fl["mut_team"].value() / 100,
            mut_order=fl["mut_order"].value() / 100,
            immigrants_pct=fl["immigrants_pct"].value() / 100,
            mut_plan=fl["mut_plan"].value() / 100,
            tournament_k=fl["tournament_k"].value(),
            survivors=min(fl["survivors"].value(), fl["population"].value()),
            selection=fl["selection"].currentData(),
            fill_strategy=fl["fill_strategy"].currentData(),
            children_per_pair=fl["children_per_pair"].currentData(),
            workers=fl["workers"].value(),
        )
        cfg.scenarios = fl["scenarios"].value()
        cfg.smart_init = fl["smart_init"].isChecked()
        cfg.switching = fl["switching"].isChecked()
        cfg.adaptive_modes = fl["adaptive_modes"].isChecked()
        cfg.graded_loss_penalty = fl["graded_loss_penalty"].isChecked()
        cfg.seed = None if self.seed_random.isChecked() else self.seed.value()
        cfg.preset = key
        return cfg.validate()


def _fmt_time(seconds) -> str:
    if seconds is None:
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _fmt_fit(value) -> str:
    return "∞" if value is None else f"{value:,.0f}".replace(",", ".")


class TrainingView(QWidget):
    open_individual = pyqtSignal(object, str)
    run_saved = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.runner = GARunner(self)
        self.runner.started.connect(self._on_started)
        self.runner.generation.connect(self._on_generation)
        self.runner.hall_of_fame.connect(self._on_hof)
        self.runner.finished.connect(self._on_finished)
        self.runner.failed.connect(self._on_failed)
        self.runner.state_changed.connect(self._on_state)
        self.run_dir = None
        self.best_genome = None
        self.best_stats = None
        self.hof = []
        self._t_start = None

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        # ---- left: configuration + controls
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(10)
        controls = Card("Ejecución")
        btns = QHBoxLayout()
        self.btn_start = QPushButton("▶  Entrenar")
        self.btn_start.setProperty("kind", "primary")
        self.btn_pause = QPushButton("⏸  Pausar")
        self.btn_stop = QPushButton("■  Detener")
        for b in (self.btn_start, self.btn_pause, self.btn_stop):
            btns.addWidget(b)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        controls.body.addLayout(btns)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        controls.body.addWidget(self.progress)
        self.lbl_gen = QLabel("Listo para entrenar")
        self.lbl_gen.setProperty("role", "h2")
        self.lbl_time = QLabel("")
        self.lbl_time.setProperty("role", "muted")
        self.lbl_speed = QLabel("")
        self.lbl_speed.setProperty("role", "muted")
        self.lbl_dir = QLabel("")
        self.lbl_dir.setProperty("role", "muted")
        self.lbl_dir.setWordWrap(True)
        for w in (self.lbl_gen, self.lbl_time, self.lbl_speed, self.lbl_dir):
            controls.body.addWidget(w)
        left_lay.addWidget(controls)
        self.config = ConfigPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.config)
        left_lay.addWidget(scroll, 1)
        left.setMinimumWidth(370)
        left.setMaximumWidth(410)
        splitter.addWidget(left)

        # ---- center: charts
        center = QWidget()
        c_lay = QVBoxLayout(center)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(10)
        fit_card = Card(None, margins=(6, 6, 6, 6))
        top_row = QHBoxLayout()
        top_row.addStretch(1)
        self.log_check = QCheckBox("Escala logarítmica")
        self.log_check.setChecked(True)
        top_row.addWidget(self.log_check)
        fit_card.body.addLayout(top_row)
        self.fitness_chart = charts.FitnessChart()
        fit_card.body.addWidget(self.fitness_chart, 1)
        self.log_check.toggled.connect(self.fitness_chart.set_log)
        c_lay.addWidget(fit_card, 5)
        self.tabs = QTabWidget()
        self.heatmap = charts.BattleHeatmap()
        self.scatter = charts.PopulationScatter()
        div_widget = QWidget()
        dl = QGridLayout(div_widget)
        dl.setContentsMargins(4, 4, 4, 4)
        self.diversity = charts.DiversityChart()
        self.wins = charts.WinsHistogram()
        dl.addWidget(self.diversity, 0, 0)
        dl.addWidget(self.wins, 0, 1)
        self.tabs.addTab(self.heatmap, "Combates del mejor")
        self.tabs.addTab(self.scatter, "Población")
        self.tabs.addTab(div_widget, "Diversidad y victorias")
        c_lay.addWidget(self.tabs, 4)
        splitter.addWidget(center)

        # ---- right: champion, records, hall of fame
        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(10)
        champ = Card("Campeón actual")
        tiles = QHBoxLayout()
        self.tile_fit = StatTile("Fitness")
        self.tile_wins = StatTile("Victorias")
        tiles.addWidget(self.tile_fit)
        tiles.addWidget(self.tile_wins)
        champ.body.addLayout(tiles)
        chips = QHBoxLayout()
        self.chip_gen = Chip("Gen -")
        self.chip_origin = Chip("Origen -")
        self.chip_hp = Chip("HP perdido -")
        for c in (self.chip_gen, self.chip_origin, self.chip_hp):
            chips.addWidget(c)
        chips.addStretch(1)
        champ.body.addLayout(chips)
        lbl = QLabel("Equipo de la Liga Pokémon")
        lbl.setProperty("role", "muted")
        champ.body.addWidget(lbl)
        self.league_strip = TeamStrip(size=54, animated=True)
        self.league_strip.setMinimumHeight(60)
        champ.body.addWidget(self.league_strip)
        self.btn_open_best = QPushButton("Abrir en el visor")
        self.btn_open_best.setEnabled(False)
        champ.body.addWidget(self.btn_open_best)
        r_lay.addWidget(champ)

        rec = Card("Récords")
        self.records = QListWidget()
        self.records.setMinimumHeight(110)
        rec.body.addWidget(self.records)
        r_lay.addWidget(rec, 2)

        hof = Card("Hall of Fame (doble clic para abrir)")
        self.hof_table = QTableWidget(0, 5)
        self.hof_table.setHorizontalHeaderLabels(["#", "Fitness", "Victorias", "Gen", "Origen"])
        self.hof_table.verticalHeader().setVisible(False)
        self.hof_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.hof_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.hof_table.setAlternatingRowColors(True)
        self.hof_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hof_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hof.body.addWidget(self.hof_table)
        r_lay.addWidget(hof, 3)
        right.setMinimumWidth(330)
        right.setMaximumWidth(420)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        self.btn_start.clicked.connect(self.start)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop.clicked.connect(self.runner.stop)
        self.btn_open_best.clicked.connect(self._open_best)
        self.hof_table.cellDoubleClicked.connect(self._open_hof_row)

    # ------------------------------------------------------------------ control

    def _preflight(self) -> bool:
        present, required = damage_table.coverage(bs.dataset)
        if present >= required:
            return True
        if damage_table.server_available():
            return True
        answer = QMessageBox.warning(
            self, "Tabla de daños incompleta",
            f"La tabla de daños tiene {present} de {required} entradas y el servidor de daños no responde.\n\n"
            "Arranca 'node damage_calculator/servidor.js' y precalcula la tabla desde la barra superior "
            "(o con 'python tools/precompute_damage.py'). ¿Quieres intentarlo igualmente?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    def _reset_views(self):
        for chart in (self.fitness_chart, self.heatmap, self.scatter, self.diversity, self.wins):
            chart.reset()
        self.records.clear()
        self.hof_table.setRowCount(0)
        self.hof = []
        self.best_genome = None
        self.best_stats = None
        self.league_strip.clear_team()
        self.btn_open_best.setEnabled(False)
        self.tile_fit.set("-")
        self.tile_wins.set("-")

    def start(self):
        if self.runner.is_active():
            return
        try:
            cfg = self.config.get_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Configuración no válida", str(exc))
            return
        if not self._preflight():
            return
        self._reset_views()
        self._t_start = time.time()
        self.lbl_gen.setText("Generando población inicial…")
        self.runner.start(config=cfg.to_dict())

    def resume_run(self, run_dir: str, generations: int | None = None):
        if self.runner.is_active():
            QMessageBox.information(self, "Entrenamiento en curso", "Detén el entrenamiento actual antes de reanudar otro.")
            return
        if not self._preflight():
            return
        self._reset_views()
        self._t_start = time.time()
        self.lbl_gen.setText("Reanudando…")
        self.runner.start(resume_dir=run_dir, generations=generations)

    def _toggle_pause(self):
        if self.runner.state == "running":
            self.runner.pause()
        elif self.runner.state == "paused":
            self.runner.resume()

    def _on_state(self, state: str):
        running = state in ("running", "paused", "stopping")
        self.btn_start.setEnabled(not running)
        self.btn_pause.setEnabled(state in ("running", "paused"))
        self.btn_stop.setEnabled(state in ("running", "paused"))
        self.btn_pause.setText("▶  Continuar" if state == "paused" else "⏸  Pausar")
        self.config.setEnabled(not running)
        if state == "stopping":
            self.lbl_gen.setText("Deteniendo (guardando checkpoint)…")
        elif state == "paused":
            self.lbl_gen.setText(self.lbl_gen.text().split(" · ")[0] + " · en pausa")

    def shutdown(self):
        self.runner.shutdown()

    # ------------------------------------------------------------------ messages

    def _on_started(self, payload: dict):
        self.run_dir = payload["run_dir"]
        self.lbl_dir.setText(f"Guardando en: {os.path.relpath(self.run_dir, bs.ROOT)}")
        cfg = GAConfig.from_dict(payload["config"])
        self.config.set_config(cfg)
        history = payload.get("history") or []
        if history:
            for chart in (self.fitness_chart, self.heatmap, self.scatter, self.diversity, self.wins):
                chart.load(history)
            for st in history:
                self._add_record(st)
        if payload.get("hall_of_fame"):
            self._on_hof(payload["hall_of_fame"])
            top = payload["hall_of_fame"][0]
            self._set_champion(top["genome"], top["fitness"], top["wins"], top["hp_lost"], top["gen_found"], top["origin"],
                               top.get("p_complete"))

    def _add_record(self, st: dict):
        if not st.get("improved") or st["gen"] == 0:
            return
        origin = theme.ORIGIN_LABELS.get(st.get("best_origin"), st.get("best_origin"))
        prev = self.fitness_chart.best[-2] if len(self.fitness_chart.best) > 1 else None
        delta = f"  ({st['best'] - prev:+,.0f})".replace(",", ".") if prev else ""
        item = QListWidgetItem(f"Gen {st['gen']:>4} · {_fmt_fit(st['best'])}{delta} · {origin} · {st['best_wins']}/21")
        item.setForeground(pg_color(theme.ORIGIN_COLORS.get(st.get("best_origin"), theme.TEXT)))
        self.records.insertItem(0, item)

    def _on_generation(self, st: dict):
        for chart in (self.fitness_chart, self.heatmap, self.scatter, self.diversity, self.wins):
            chart.add(st)
        self._add_record(st)
        g, total = st["gen"], st["generations"]
        self.progress.setValue(int(100 * g / max(1, total)))
        self.lbl_gen.setText(f"Generación {g} / {total}" + (" · en pausa" if self.runner.state == "paused" else ""))
        elapsed = st.get("elapsed")
        eta = (elapsed / g) * (total - g) if g and elapsed else None
        self.lbl_time.setText(f"Tiempo {_fmt_time(elapsed)} · restante ≈ {_fmt_time(eta)}")
        hit = st.get("battle_hit_rate")
        hit_txt = f" · caché de combates {hit:.0%}" if hit is not None else ""
        self.lbl_speed.setText(f"{st.get('evals_per_s') or 0:,.0f} evaluaciones/s{hit_txt} · únicos "
                               f"{st['unique_ratio']:.0%}".replace(",", "."))
        if st.get("improved") and st.get("best_genome"):
            self._set_champion(st["best_genome"], st["best"], st["best_wins"], st["best_hp_lost"],
                               st["best_found_gen"], st["best_origin"], st.get("best_p_complete"))
        self.best_stats = st

    def _set_champion(self, genome, fitness, wins, hp_lost, gen, origin, p_complete=None):
        self.best_genome = genome
        self.tile_fit.set(_fmt_fit(fitness), theme.YELLOW)
        if p_complete is None:
            self.tile_wins.caption.setText("Victorias")
            self.tile_wins.set(f"{wins}/21", theme.GREEN if wins == 21 else theme.RED)
        else:  # random battles: probability of winning the 21 battles in the training scenarios
            self.tile_wins.caption.setText("Completa la ruta")
            text = "100 %" if p_complete >= 1 else f"{p_complete * 100:.1f} %".replace(".", ",")
            self.tile_wins.set(text, theme.GREEN if p_complete >= 0.99 else (theme.ORANGE if p_complete >= 0.8 else theme.RED))
        self.chip_gen.set(f"Gen {gen}")
        self.chip_origin.set(theme.ORIGIN_LABELS.get(origin, origin), "accent")
        self.chip_hp.set(f"HP perdido {hp_lost}", "neutral")
        league = gd.evolved_team(genome, 16)
        self.league_strip.set_team(league, tooltips=[gd.pkm_name(p) for p in league])
        self.btn_open_best.setEnabled(True)

    def _on_hof(self, entries: list):
        self.hof = entries
        self.hof_table.setRowCount(len(entries))
        for row, e in enumerate(entries):
            values = [str(row + 1), _fmt_fit(e["fitness"]), f"{e['wins']}/21", str(e["gen_found"]),
                      theme.ORIGIN_LABELS.get(e["origin"], e["origin"])]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 4:
                    item.setForeground(pg_color(theme.ORIGIN_COLORS.get(e["origin"], theme.TEXT)))
                self.hof_table.setItem(row, col, item)

    def _on_finished(self, result: dict):
        stopped = result.get("stopped")
        best = result.get("best") or {}
        self.lbl_gen.setText(("Detenido" if stopped else "Completado") + f" · generación {result.get('generations_done')}")
        if best:
            self._set_champion(best["genome"], best["fitness"], best["wins"], best["hp_lost"], best["gen_found"],
                               best["origin"], best.get("p_complete"))
        if result.get("run_dir"):
            self.run_saved.emit(result["run_dir"])

    def _on_failed(self, tb: str):
        self.lbl_gen.setText("Error en el entrenamiento")
        last = tb.strip().splitlines()[-1] if tb.strip() else tb
        box = QMessageBox(QMessageBox.Critical, "Error en el algoritmo", last, parent=self)
        box.setDetailedText(tb)
        box.exec_()

    # ------------------------------------------------------------------ open

    def _open_best(self):
        if self.best_genome:
            self.open_individual.emit(self.best_genome, "Campeón del entrenamiento actual")

    def _open_hof_row(self, row: int, _col: int):
        if 0 <= row < len(self.hof):
            e = self.hof[row]
            self.open_individual.emit(e["genome"], f"Hall of Fame #{row + 1} (gen {e['gen_found']})")


def pg_color(hex_color: str):
    from PyQt5.QtGui import QColor

    return QColor(hex_color)
