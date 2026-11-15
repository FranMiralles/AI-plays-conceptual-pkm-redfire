"""Saved runs: replay their charts, compare them, open individuals and resume them."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices
from PyQt5.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMessageBox,
                             QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from ga.config import PRESET_LABELS
from ga.engine import RUNS_DIR, list_runs, read_history
from ui import charts
from ui import theme
from ui.widgets import Card


def _fmt(v):
    return "-" if v is None else f"{v:,.0f}".replace(",", ".")


def run_label(name: str) -> str:
    """'20261114-234148_mejorado' -> '14/11 23:41 · mejorado'."""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})_(.+)$", name)
    if m:
        return f"{m.group(3)}/{m.group(2)} {m.group(4)}:{m.group(5)} · {m.group(7)}"
    m = re.match(r"^\d{8}-(.+)$", name)
    return m.group(1) if m else name


class RunsView(QWidget):
    open_individual = pyqtSignal(object, str)
    resume_requested = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.runs = []
        self.current = None
        self.hof = []
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = Card("Ejecuciones guardadas")
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Ejecución", "Preset", "Pob.", "Gen.", "Mejor", "Victorias"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        left.body.addWidget(self.table, 1)
        hint = QLabel("Selecciona varias con Ctrl para compararlas.")
        hint.setProperty("role", "muted")
        left.body.addWidget(hint)
        btns = QHBoxLayout()
        self.btn_refresh = QPushButton("Actualizar")
        self.btn_folder = QPushButton("Abrir carpeta")
        self.btn_compare = QPushButton("Comparar")
        self.btn_resume = QPushButton("Reanudar…")
        self.btn_resume.setProperty("kind", "primary")
        for b in (self.btn_refresh, self.btn_folder, self.btn_compare, self.btn_resume):
            btns.addWidget(b)
        left.body.addLayout(btns)
        left.setMinimumWidth(430)
        splitter.addWidget(left)

        self.tabs = QTabWidget()
        evo = QWidget()
        ev_lay = QVBoxLayout(evo)
        ev_lay.setContentsMargins(6, 6, 6, 6)
        self.fitness = charts.FitnessChart()
        self.heatmap = charts.BattleHeatmap()
        ev_lay.addWidget(self.fitness, 3)
        ev_lay.addWidget(self.heatmap, 2)
        self.tabs.addTab(evo, "Evolución")
        pop = QWidget()
        pop_lay = QVBoxLayout(pop)
        pop_lay.setContentsMargins(6, 6, 6, 6)
        self.scatter = charts.PopulationScatter()
        self.diversity = charts.DiversityChart()
        pop_lay.addWidget(self.scatter, 3)
        pop_lay.addWidget(self.diversity, 2)
        self.tabs.addTab(pop, "Población")
        hof = QWidget()
        hof_lay = QVBoxLayout(hof)
        hof_lay.setContentsMargins(6, 6, 6, 6)
        self.hof_table = QTableWidget(0, 6)
        self.hof_table.setHorizontalHeaderLabels(["#", "Fitness", "Victorias", "HP perdido", "Gen", "Origen"])
        self.hof_table.verticalHeader().setVisible(False)
        self.hof_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.hof_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.hof_table.setAlternatingRowColors(True)
        self.hof_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        hof_lay.addWidget(QLabel("Doble clic en un individuo para abrirlo en el visor."))
        hof_lay.addWidget(self.hof_table, 1)
        self.tabs.addTab(hof, "Hall of Fame")
        self.compare = charts.CompareChart()
        self.tabs.addTab(self.compare, "Comparativa")
        self.config_text = QPlainTextEdit()
        self.config_text.setReadOnly(True)
        self.tabs.addTab(self.config_text, "Configuración")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)

        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_folder.clicked.connect(self._open_folder)
        self.btn_compare.clicked.connect(self._compare)
        self.btn_resume.clicked.connect(self._resume)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.hof_table.cellDoubleClicked.connect(self._open_hof)
        self.refresh()

    def refresh(self, select: str | None = None):
        self.runs = list_runs()
        self.table.setRowCount(len(self.runs))
        for row, run in enumerate(self.runs):
            cfg, last = run["config"], run["last"] or {}
            preset = PRESET_LABELS.get(cfg.get("preset"), cfg.get("preset", "-")).split(" (")[0]
            values = [run_label(run["name"]), preset.capitalize(), str(cfg.get("population")),
                      f"{last.get('gen', 0)}/{cfg.get('generations')}", _fmt(last.get("best")),
                      f"{last.get('best_wins', '-')}/21"]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col >= 2:
                    item.setTextAlignment(Qt.AlignCenter)
                if col == 0:
                    item.setToolTip(run["path"])
                if col == 4:
                    item.setForeground(QColor(theme.YELLOW))
                self.table.setItem(row, col, item)
        if self.runs:
            idx = next((i for i, r in enumerate(self.runs) if select and r["path"] == select), 0)
            self.table.selectRow(idx)

    def _selected_rows(self):
        return sorted({i.row() for i in self.table.selectedIndexes()})

    def _selection_changed(self):
        rows = self._selected_rows()
        if not rows:
            return
        run = self.runs[rows[0]]
        if self.current == run["path"]:
            return
        self.current = run["path"]
        history = read_history(run["path"])
        for chart in (self.fitness, self.heatmap, self.scatter, self.diversity):
            chart.load(history)
        self.config_text.setPlainText(json.dumps(run["config"], indent=2, ensure_ascii=False))
        hof_path = Path(run["path"]) / "hall_of_fame.json"
        self.hof = json.loads(hof_path.read_text(encoding="utf-8")) if hof_path.exists() else []
        self.hof_table.setRowCount(len(self.hof))
        for r, e in enumerate(self.hof):
            vals = [str(r + 1), _fmt(e["fitness"]), f"{e['wins']}/21", str(e["hp_lost"]), str(e["gen_found"]),
                    theme.ORIGIN_LABELS.get(e["origin"], e["origin"])]
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if c == 5:
                    item.setForeground(QColor(theme.ORIGIN_COLORS.get(e["origin"], theme.TEXT)))
                self.hof_table.setItem(r, c, item)

    def _compare(self):
        rows = self._selected_rows()
        if not rows:
            return
        self.compare.set_runs([(run_label(self.runs[r]["name"]), read_history(self.runs[r]["path"])) for r in rows])
        self.tabs.setCurrentWidget(self.compare)

    def _open_folder(self):
        target = self.current or str(RUNS_DIR)
        os.makedirs(target, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    def _open_hof(self, row: int, _col: int):
        if 0 <= row < len(self.hof):
            name = Path(self.current).name if self.current else ""
            self.open_individual.emit(self.hof[row]["genome"], f"{name} · Hall of Fame #{row + 1}")

    def _resume(self):
        rows = self._selected_rows()
        if not rows:
            return
        run = self.runs[rows[0]]
        if not run["resumable"]:
            QMessageBox.information(self, "Sin checkpoint", "Esta ejecución no tiene checkpoint para reanudarla.")
            return
        done = (run["last"] or {}).get("gen", 0)
        target = max(run["config"].get("generations", done), done + 50)
        gens, ok = QInputDialog.getInt(self, "Reanudar ejecución",
                                       f"Generación actual: {done}. ¿Hasta qué generación quieres continuar?",
                                       target, done + 1, 1_000_000, 10)
        if ok:
            self.resume_requested.emit(run["path"], gens)
