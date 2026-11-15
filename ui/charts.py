"""Live charts (pyqtgraph) of the training and of an individual."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from ga.engine import ORIGINS
from ui import gamedata as gd
from ui import theme

N_BATTLES = len(gd.TRAINER_NAMES)


class FitnessAxis(pg.AxisItem):
    """In log mode only labels 1, 2 and 5 times a power of ten, written as plain numbers."""

    def tickStrings(self, values, scale, spacing):
        if not self.logMode:
            return super().tickStrings(values, scale, spacing)
        out = []
        for v in values:
            x = 10 ** v
            mantissa = round(x / 10 ** math.floor(v + 1e-9), 3)
            out.append(f"{x:,.0f}".replace(",", ".") if mantissa in (1.0, 2.0, 5.0) else "")
        return out


def _axes():
    return {"left": FitnessAxis(orientation="left")}


LEGEND_BRUSH = (20, 30, 51, 225)


class FitnessChart(pg.PlotWidget):
    """Best / median fitness with the p10-p90 band of the population and record markers."""

    def __init__(self, parent=None, title="Evolución del fitness"):
        super().__init__(parent, axisItems=_axes())
        self.plot_item = self.getPlotItem()
        theme.style_plot(self.plot_item, title, "Fitness (HP perdido + 10000 por derrota)", "Generación")
        self.plot_item.addLegend(offset=(-10, 10), labelTextColor=theme.MUTED, brush=pg.mkBrush(*LEGEND_BRUSH))
        self.band_low = pg.PlotDataItem(pen=pg.mkPen(None))
        self.band_high = pg.PlotDataItem(pen=pg.mkPen(None))
        self.band = pg.FillBetweenItem(self.band_low, self.band_high, brush=pg.mkBrush(79, 143, 230, 45))
        self.plot_item.addItem(self.band_low)
        self.plot_item.addItem(self.band_high)
        self.plot_item.addItem(self.band)
        self.median_curve = self.plot(pen=pg.mkPen(theme.BLUE, width=2), name="Mediana población")
        self.best_gen_curve = self.plot(pen=pg.mkPen(theme.GREY, width=1, style=Qt.DashLine), name="Mejor de la generación")
        self.best_curve = self.plot(pen=pg.mkPen(theme.YELLOW, width=3), name="Mejor global")
        self.records = pg.ScatterPlotItem(size=9, pen=pg.mkPen(theme.BG, width=1.5), brush=pg.mkBrush(theme.ACCENT))
        self.plot_item.addItem(self.records)
        self.band_legend = pg.PlotDataItem(pen=pg.mkPen((79, 143, 230, 120), width=8))
        self.plot_item.legend.addItem(self.band_legend, "Banda p10-p90")
        self.log_y = True
        self.plot_item.setLogMode(y=True)
        self.regions = []
        self.reset()

    def reset(self):
        self.gens, self.best, self.median, self.p10, self.p90, self.gen_best = [], [], [], [], [], []
        self.record_pts = []
        self.modes = []
        for r in self.regions:
            self.plot_item.removeItem(r)
        self.regions = []
        self._redraw()

    def set_log(self, on: bool):
        self.log_y = on
        self.plot_item.setLogMode(y=on)
        self._redraw()

    def add(self, stats: dict, redraw: bool = True):
        if stats.get("best") is None:
            return
        self.gens.append(stats["gen"])
        self.best.append(stats["best"])
        self.gen_best.append(stats.get("gen_best") or stats["best"])
        self.median.append(stats.get("median") or stats["best"])
        self.p10.append(stats.get("p10") or stats["best"])
        self.p90.append(stats.get("p90") or stats["best"])
        if stats.get("improved") and stats["gen"] > 0:
            self.record_pts.append((stats["gen"], stats["best"]))
        mode = stats.get("mode")
        if mode:
            self._add_mode(stats["gen"], mode)
        if redraw:
            self._redraw()

    def _add_mode(self, gen, mode):
        if self.modes and self.modes[-1][1] == mode:
            self.modes[-1][2] = gen
            self.regions[-1].setRegion((self.modes[-1][0] - 0.5, gen + 0.5))
            return
        self.modes.append([gen, mode, gen])
        color = (255, 203, 5, 22) if mode == "exploitation" else (167, 139, 250, 22)
        region = pg.LinearRegionItem((gen - 0.5, gen + 0.5), movable=False, brush=pg.mkBrush(*color), pen=pg.mkPen(None))
        region.setZValue(-10)
        self.plot_item.addItem(region)
        self.regions.append(region)

    def load(self, history: list):
        self.reset()
        for st in history:
            self.add(st, redraw=False)
        self._redraw()

    def _redraw(self):
        x = np.array(self.gens, dtype=float)
        self.best_curve.setData(x, np.array(self.best, dtype=float))
        self.best_gen_curve.setData(x, np.array(self.gen_best, dtype=float))
        self.median_curve.setData(x, np.array(self.median, dtype=float))
        self.band_low.setData(x, np.array(self.p10, dtype=float))
        self.band_high.setData(x, np.array(self.p90, dtype=float))
        if self.record_pts:
            pts = np.array(self.record_pts, dtype=float)
            self.records.setData(pts[:, 0], np.log10(pts[:, 1]) if self.log_y else pts[:, 1])
        else:
            self.records.setData([], [])


class BattleHeatmap(pg.PlotWidget):
    """HP lost by the best individual in each of the 21 battles, generation by generation."""

    CAP = 300

    def __init__(self, parent=None, title="Combates del mejor individuo"):
        super().__init__(parent)
        self.plot_item = self.getPlotItem()
        theme.style_plot(self.plot_item, title, None, "Generación")
        self.plot_item.showGrid(x=False, y=False)
        self.image = pg.ImageItem()
        self.plot_item.addItem(self.image)
        positions = [0.0, 0.004, 0.005, 0.25, 0.6, 1.0]
        colors = [(40, 12, 18), (40, 12, 18), (56, 211, 159), (180, 220, 90), (245, 165, 36), (239, 68, 68)]
        self.lut = pg.ColorMap(positions, colors).getLookupTable(0.0, 1.0, 256)
        left = self.plot_item.getAxis("left")
        left.setTicks([[(i + 0.5, name) for i, name in enumerate(gd.TRAINER_SHORT_NAMES)]])
        left.setStyle(tickFont=pg.QtGui.QFont("Segoe UI", 8))
        self.plot_item.invertY(True)
        self.plot_item.setMouseEnabled(x=True, y=False)
        self.columns = []
        self.gens = []
        self.setToolTip("")
        self.scene().sigMouseMoved.connect(self._hover)
        self.setToolTip("Verde: poco HP perdido · rojo: mucho · negro: derrota")

    def reset(self):
        self.columns, self.gens = [], []
        self.image.clear()

    def _encode(self, per_battle: list) -> list:
        out = []
        for v in per_battle:
            if v is None or v < 0:
                out.append(0.0)
            else:
                out.append(0.006 + 0.994 * min(v, self.CAP) / self.CAP)
        return out

    def add(self, stats: dict, redraw: bool = True):
        per = stats.get("best_per_battle")
        if not per or len(per) != N_BATTLES:
            return
        self.gens.append(stats["gen"])
        self.columns.append((self._encode(per), per))
        if redraw:
            self._redraw()

    def load(self, history: list):
        self.reset()
        for st in history:
            self.add(st, redraw=False)
        self._redraw()

    def _redraw(self):
        if not self.columns:
            self.image.clear()
            return
        data = np.array([c[0] for c in self.columns], dtype=float)  # (gens, battles)
        self.image.setImage(data, levels=(0.0, 1.0), lut=self.lut, autoLevels=False)
        g0 = self.gens[0]
        self.image.setRect(pg.QtCore.QRectF(g0 - 0.5, 0, len(self.gens), N_BATTLES))

    def _hover(self, pos):
        if not self.columns or not self.plot_item.sceneBoundingRect().contains(pos):
            return
        p = self.plot_item.vb.mapSceneToView(pos)
        col = int(round(p.x() - (self.gens[0] - 0.5) - 0.5))
        row = int(p.y())
        if 0 <= col < len(self.columns) and 0 <= row < N_BATTLES:
            v = self.columns[col][1][row]
            text = "derrota" if v < 0 else f"{v} % HP perdido"
            self.setToolTip(f"Gen {self.gens[col]} · {gd.TRAINER_NAMES[row]}: {text}")


class PopulationScatter(pg.PlotWidget):
    """Every individual of each generation (sampled), coloured by its origin."""

    MAX_POINTS_PER_GEN = 120

    def __init__(self, parent=None):
        super().__init__(parent, axisItems=_axes())
        self._rng = np.random.default_rng(0)
        self.plot_item = self.getPlotItem()
        theme.style_plot(self.plot_item, "Población por generación", "Fitness (log)", "Generación")
        self.plot_item.setLogMode(y=True)
        self.scatter = pg.ScatterPlotItem(size=4, pen=None)
        self.plot_item.addItem(self.scatter)
        legend = self.plot_item.addLegend(offset=(-10, 10), labelTextColor=theme.MUTED,
                                          brush=pg.mkBrush(*LEGEND_BRUSH))
        for origin in ORIGINS:
            item = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush(theme.ORIGIN_COLORS[origin]))
            legend.addItem(item, theme.ORIGIN_LABELS[origin])
        self.brushes = [pg.mkBrush(QColor(theme.ORIGIN_COLORS[o]).red(), QColor(theme.ORIGIN_COLORS[o]).green(),
                                   QColor(theme.ORIGIN_COLORS[o]).blue(), 150) for o in ORIGINS]
        self.reset()

    def reset(self):
        self.xs, self.ys, self.bs = [], [], []
        self.scatter.setData([], [])

    def add(self, stats: dict, redraw: bool = True):
        pop = [p for p in stats.get("population", []) if p[0] is not None and p[0] > 0]
        if len(pop) > self.MAX_POINTS_PER_GEN:
            step = len(pop) / self.MAX_POINTS_PER_GEN
            pop = [pop[int(i * step)] for i in range(self.MAX_POINTS_PER_GEN)]
        g = stats["gen"]
        n = len(pop)
        jitter = self._rng.uniform(-0.38, 0.38, n)
        for (fit, origin), j in zip(pop, jitter):
            self.xs.append(g + j)
            self.ys.append(math.log10(fit))
            self.bs.append(self.brushes[origin])
        if redraw:
            self._redraw()

    def load(self, history: list):
        self.reset()
        for st in history:
            self.add(st, redraw=False)
        self._redraw()

    def _redraw(self):
        self.scatter.setData(self.xs, self.ys, brush=self.bs)


class DiversityChart(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_item = self.getPlotItem()
        theme.style_plot(self.plot_item, "Diversidad de la población", "0 - 1", "Generación")
        self.plot_item.addLegend(offset=(-10, 10), labelTextColor=theme.MUTED, brush=pg.mkBrush(*LEGEND_BRUSH))
        self.plot_item.setYRange(0, 1.02)
        self.unique_curve = self.plot(pen=pg.mkPen(theme.GREEN, width=2), name="Genotipos únicos")
        self.div_curve = self.plot(pen=pg.mkPen(theme.PURPLE, width=2), name="Distancia media de capturas")
        self.win_curve = self.plot(pen=pg.mkPen(theme.ORANGE, width=2), name="Completan el juego (21/21)")
        self.reset()

    def reset(self):
        self.g, self.u, self.d, self.w = [], [], [], []
        self._redraw()

    def add(self, stats: dict, redraw: bool = True):
        self.g.append(stats["gen"])
        self.u.append(stats.get("unique_ratio", 0))
        self.d.append(stats.get("diversity", 0))
        hist = stats.get("wins_hist") or []
        total = sum(hist) or 1
        self.w.append((hist[-1] / total) if hist else 0)
        if redraw:
            self._redraw()

    def load(self, history: list):
        self.reset()
        for st in history:
            self.add(st, redraw=False)
        self._redraw()

    def _redraw(self):
        self.unique_curve.setData(self.g, self.u)
        self.div_curve.setData(self.g, self.d)
        self.win_curve.setData(self.g, self.w)


class WinsHistogram(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_item = self.getPlotItem()
        theme.style_plot(self.plot_item, "Victorias en la población", "Individuos", "Combates ganados (de 21)")
        self.bars = pg.BarGraphItem(x=list(range(N_BATTLES + 1)), height=[0] * (N_BATTLES + 1), width=0.8,
                                    brush=pg.mkBrush(theme.BLUE), pen=pg.mkPen(None))
        self.plot_item.addItem(self.bars)
        self.plot_item.setXRange(-0.5, N_BATTLES + 0.5)

    def add(self, stats: dict, redraw: bool = True):
        hist = stats.get("wins_hist")
        if not hist:
            return
        brushes = [pg.mkBrush(theme.GREEN if i == N_BATTLES else theme.BLUE) for i in range(len(hist))]
        self.bars.setOpts(height=hist, brushes=brushes)

    def load(self, history: list):
        if history:
            self.add(history[-1])

    def reset(self):
        self.bars.setOpts(height=[0] * (N_BATTLES + 1))


class BattleBars(pg.PlotWidget):
    """Horizontal bars: HP lost in each battle by one individual (lost battles in red)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_item = self.getPlotItem()
        theme.style_plot(self.plot_item, "HP perdido por combate", None, "% HP perdido (suma del equipo)")
        self.plot_item.showGrid(x=True, y=False, alpha=0.12)
        left = self.plot_item.getAxis("left")
        left.setTicks([[(i, name) for i, name in enumerate(gd.TRAINER_NAMES)]])
        left.setStyle(tickFont=pg.QtGui.QFont("Segoe UI", 8))
        self.plot_item.invertY(True)
        self.plot_item.setMouseEnabled(x=False, y=False)
        self.bars = None
        self.labels = []

    def set_values(self, values: list):
        if self.bars is not None:
            self.plot_item.removeItem(self.bars)
        for t in self.labels:
            self.plot_item.removeItem(t)
        self.labels = []
        widths, brushes = [], []
        top = max([v for v in values if v is not None and v >= 0] + [100])
        for v in values:
            if v is None or v < 0:
                widths.append(top)
                brushes.append(pg.mkBrush(239, 68, 68, 170))
            else:
                widths.append(max(v, top * 0.004))
                brushes.append(pg.mkBrush(theme.hp_color(100 - min(100, v / 3))))
        y = list(range(len(values)))
        self.bars = pg.BarGraphItem(x0=[0] * len(values), y=y, height=0.7, width=widths, brushes=brushes,
                                    pen=pg.mkPen(None))
        self.plot_item.addItem(self.bars)
        for i, v in enumerate(values):
            text = "DERROTA" if v is None or v < 0 else f"{v}"
            item = pg.TextItem(text, color=theme.TEXT, anchor=(0, 0.5))
            item.setPos(widths[i] + top * 0.01, i)
            self.plot_item.addItem(item)
            self.labels.append(item)
        self.plot_item.setXRange(0, top * 1.15)
        self.plot_item.setYRange(-0.6, len(values) - 0.4)


class CompareChart(pg.PlotWidget):
    PALETTE = [theme.YELLOW, theme.BLUE, theme.GREEN, theme.PURPLE, theme.ORANGE, theme.ACCENT, "#22d3ee", "#f472b6"]

    def __init__(self, parent=None):
        super().__init__(parent, axisItems=_axes())
        self.plot_item = self.getPlotItem()
        theme.style_plot(self.plot_item, "Comparativa: mejor fitness por generación", "Fitness (log)", "Generación")
        self.plot_item.setLogMode(y=True)
        self.legend = self.plot_item.addLegend(offset=(-10, 10), labelTextColor=theme.MUTED,
                                               brush=pg.mkBrush(*LEGEND_BRUSH))

    def set_runs(self, runs: list):
        """``runs``: list of ``(label, history)``."""
        self.plot_item.clear()
        self.legend.clear()
        for i, (label, history) in enumerate(runs):
            pts = [(h["gen"], h["best"]) for h in history if h.get("best")]
            if not pts:
                continue
            x, y = zip(*pts)
            self.plot(x, y, pen=pg.mkPen(self.PALETTE[i % len(self.PALETTE)], width=2), name=label)
