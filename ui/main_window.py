"""Main window: navigation, status bar of the damage data and the three views."""
from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                             QStackedWidget, QVBoxLayout, QWidget)

import battle_simulator as bs
from ga import damage_table
from ui import gamedata as gd
from ui import theme
from ui.individual_view import IndividualView
from ui.runs_view import RunsView
from ui.training_view import TrainingView
from ui.widgets import Chip


class PrecomputeThread(QThread):
    progress = pyqtSignal(int, int)
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def run(self):
        try:
            self.done.emit(damage_table.precompute(bs.dataset, progress=lambda d, t: self.progress.emit(d, t)))
        except Exception as exc:  # shown to the user
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nuzlocke GA · Rojo Fuego")
        self.setWindowIcon(QIcon(gd.NOT_CAPTURED_YET))
        self.resize(1560, 940)
        self.setMinimumSize(1200, 760)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        h = QHBoxLayout(header)
        h.setContentsMargins(18, 10, 18, 10)
        logo = QLabel("◓")
        logo.setStyleSheet(f"color: {theme.ACCENT}; font-size: 26px; font-weight: 900;")
        h.addWidget(logo)
        title = QLabel("Nuzlocke GA")
        title.setProperty("role", "title")
        h.addWidget(title)
        sub = QLabel("Optimización metaheurística de una ruta nuzlocke en Rojo Fuego")
        sub.setProperty("role", "muted")
        h.addWidget(sub)
        h.addStretch(1)
        self.chip_table = Chip("Tabla de daños…")
        self.chip_server = Chip("Servidor…")
        self.btn_precompute = QPushButton("Precalcular tabla")
        self.btn_precompute.setProperty("kind", "ghost")
        self.btn_check = QPushButton("Comprobar")
        self.btn_check.setProperty("kind", "ghost")
        for w in (self.chip_table, self.chip_server, self.btn_precompute, self.btn_check):
            h.addWidget(w)
        outer.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        nav = QFrame()
        nav.setObjectName("nav")
        nav.setFixedWidth(190)
        nv = QVBoxLayout(nav)
        nv.setContentsMargins(10, 16, 10, 16)
        nv.setSpacing(4)
        self.nav_group = QButtonGroup(self)
        self.stack = QStackedWidget()
        self.training = TrainingView()
        self.viewer = IndividualView()
        self.runs = RunsView()
        for i, (text, view) in enumerate((("▶   Entrenar", self.training), ("◉   Visor", self.viewer),
                                          ("☰   Ejecuciones", self.runs))):
            btn = QPushButton(text)
            btn.setProperty("kind", "nav")
            btn.setCheckable(True)
            self.nav_group.addButton(btn, i)
            nv.addWidget(btn)
            self.stack.addWidget(view)
        self.nav_group.button(0).setChecked(True)
        self.nav_group.buttonClicked[int].connect(self.stack.setCurrentIndex)
        nv.addStretch(1)
        body.addWidget(nav)
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)

        self.training.open_individual.connect(self.show_individual)
        self.runs.open_individual.connect(self.show_individual)
        self.runs.resume_requested.connect(self._resume)
        self.training.run_saved.connect(lambda path: self.runs.refresh(select=path))
        self.btn_check.clicked.connect(self.check_data)
        self.btn_precompute.clicked.connect(self._precompute)
        self._thread = None
        self.check_data()
        examples = gd.example_individuals()
        if examples:
            self.viewer.load_genome(examples[0]["genome"], examples[0]["name"])

    def navigate(self, index: int):
        self.nav_group.button(index).setChecked(True)
        self.stack.setCurrentIndex(index)

    def show_individual(self, genome, source: str):
        if self.viewer.load_genome(genome, source):
            self.navigate(1)

    def _resume(self, path: str, generations):
        self.navigate(0)
        self.training.resume_run(path, generations)

    # ------------------------------------------------------------------ damage data

    def check_data(self):
        present, required = damage_table.coverage(bs.dataset)
        complete = present >= required
        self.chip_table.set(f"Tabla de daños {present / max(1, required):.0%}", "good" if complete else "warn")
        self.chip_table.setToolTip(f"{present} de {required} combinaciones precalculadas en "
                                   f"{bs.DAMAGE_DICTIONARY_PATH.name}")
        up = damage_table.server_available(timeout=0.4)
        self.chip_server.set("Servidor activo" if up else "Servidor apagado",
                             "good" if up else ("neutral" if complete else "bad"))
        self.chip_server.setToolTip("Servidor de daños @smogon/calc (node damage_calculator/servidor.js). "
                                    "No hace falta si la tabla está completa.")
        self.btn_precompute.setVisible(not complete)
        self.btn_precompute.setEnabled(up)

    def _precompute(self):
        if self._thread is not None:
            return
        self._thread = PrecomputeThread(self)
        self._thread.progress.connect(lambda d, t: self.chip_table.set(f"Precalculando {d / max(1, t):.0%}", "warn"))
        self._thread.done.connect(self._precompute_done)
        self._thread.failed.connect(lambda msg: QMessageBox.critical(self, "Error al precalcular", msg))
        self._thread.finished.connect(self._precompute_finished)
        self.btn_precompute.setEnabled(False)
        self._thread.start()

    def _precompute_done(self, summary: dict):
        QMessageBox.information(self, "Tabla de daños",
                                f"Se han calculado {summary['computed_keys']} combinaciones "
                                f"({summary['calcs']} cálculos en {summary['seconds']} s).")

    def _precompute_finished(self):
        self._thread = None
        self.check_data()

    def closeEvent(self, event):
        if self.training.runner.is_active():
            answer = QMessageBox.question(self, "Entrenamiento en curso",
                                          "Hay un entrenamiento en curso. Se detendrá y se guardará un checkpoint "
                                          "para poder reanudarlo. ¿Salir?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        self.training.shutdown()
        event.accept()
