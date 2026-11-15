"""Colours, stylesheet and chart defaults of the interface."""
import pyqtgraph as pg

BG = "#0d1422"
PANEL = "#141e33"
PANEL_2 = "#1b2842"
PANEL_3 = "#22325a"
BORDER = "#2a3b62"
TEXT = "#e8eef8"
MUTED = "#8fa0bf"
ACCENT = "#e3350d"  # Pokédex red
ACCENT_HOVER = "#f24d27"
YELLOW = "#ffcb05"
BLUE = "#4f8fe6"
GREEN = "#38d39f"
ORANGE = "#f5a524"
PURPLE = "#a78bfa"
GREY = "#6b7a96"
RED = "#ef4444"

ORIGIN_COLORS = {
    "initial": "#94a3b8",
    "elite": YELLOW,
    "crossover": BLUE,
    "mutation": PURPLE,
    "immigrant": "#5b6b88",
}
ORIGIN_LABELS = {
    "initial": "Inicial",
    "elite": "Superviviente",
    "crossover": "Cruce",
    "mutation": "Mutación",
    "immigrant": "Aleatorio",
}


def hp_color(hp: float) -> str:
    if hp > 50:
        return "#4ade80"
    if hp > 20:
        return "#facc15"
    return RED


STYLE_SHEET = f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 13px;
}}
QToolTip {{
    background-color: {PANEL_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px; border-radius: 6px;
}}
QFrame#card {{
    background-color: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
}}
QFrame#card QLabel, QFrame#card QCheckBox {{ background: transparent; }}
QFrame#nav {{ background-color: {PANEL}; border-right: 1px solid {BORDER}; }}
QFrame#nav QLabel {{ background: transparent; }}
QFrame#header {{ background-color: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QFrame#header QLabel {{ background: transparent; }}
QLabel[role="title"] {{ font-size: 20px; font-weight: 700; }}
QLabel[role="h2"] {{ font-size: 15px; font-weight: 700; }}
QLabel[role="section"] {{ color: {MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QLabel[role="muted"] {{ color: {MUTED}; }}
QLabel[role="big"] {{ font-size: 28px; font-weight: 800; }}
QLabel[role="message"] {{
    background-color: rgba(10, 16, 30, 215); color: {TEXT}; border: 2px solid {BORDER};
    border-radius: 12px; padding: 10px 14px; font-size: 15px; font-weight: 600;
}}
QLabel[chip] {{ border-radius: 10px; padding: 3px 10px; font-weight: 700; font-size: 12px; }}
QLabel[chip="neutral"] {{ background-color: {PANEL_3}; color: {TEXT}; }}
QLabel[chip="good"] {{ background-color: rgba(56, 211, 159, 40); color: {GREEN}; border: 1px solid rgba(56,211,159,90); }}
QLabel[chip="bad"] {{ background-color: rgba(239, 68, 68, 40); color: #ff8a8a; border: 1px solid rgba(239,68,68,90); }}
QLabel[chip="warn"] {{ background-color: rgba(245, 165, 36, 40); color: {ORANGE}; border: 1px solid rgba(245,165,36,90); }}
QLabel[chip="accent"] {{ background-color: rgba(255, 203, 5, 35); color: {YELLOW}; border: 1px solid rgba(255,203,5,90); }}

QPushButton {{
    background-color: {PANEL_3}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 7px 14px; font-weight: 600;
}}
QPushButton:hover {{ background-color: #2b3f6e; }}
QPushButton:pressed {{ background-color: #1a2949; }}
QPushButton:disabled {{ color: #56647f; background-color: {PANEL_2}; border-color: {PANEL_3}; }}
QPushButton[kind="primary"] {{ background-color: {ACCENT}; border: 1px solid {ACCENT}; color: white; }}
QPushButton[kind="primary"]:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton[kind="primary"]:disabled {{ background-color: #6d2a1c; border-color: #6d2a1c; color: #c9a59c; }}
QPushButton[kind="ghost"] {{ background-color: transparent; border: 1px solid {BORDER}; }}
QPushButton[kind="ghost"]:hover {{ background-color: {PANEL_2}; }}
QPushButton[kind="nav"] {{
    background-color: transparent; border: none; border-radius: 10px; text-align: left;
    padding: 10px 14px; font-size: 14px; color: {MUTED};
}}
QPushButton[kind="nav"]:hover {{ background-color: {PANEL_2}; color: {TEXT}; }}
QPushButton[kind="nav"]:checked {{ background-color: rgba(227, 53, 13, 38); color: white; border-left: 3px solid {ACCENT}; }}
QToolButton {{
    background-color: {PANEL_3}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 12px;
    font-weight: 600;
}}
QToolButton:hover {{ background-color: #2b3f6e; }}
QToolButton::menu-indicator {{ image: none; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit {{
    background-color: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 7px; padding: 5px 8px;
    selection-background-color: {ACCENT};
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: #3b5287; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {PANEL_2}; border: 1px solid {BORDER}; selection-background-color: {PANEL_3}; outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {BORDER}; background: {PANEL_2}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 10px; top: -1px; background: {PANEL}; }}
QTabBar::tab {{
    background: transparent; color: {MUTED}; padding: 8px 18px; margin-right: 4px; border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QTableWidget, QListWidget {{
    background-color: {PANEL}; alternate-background-color: {PANEL_2}; border: 1px solid {BORDER};
    border-radius: 10px; gridline-color: transparent; outline: none;
}}
QTableWidget::item, QListWidget::item {{ padding: 4px; border: none; }}
QTableWidget::item:selected, QListWidget::item:selected {{ background-color: {PANEL_3}; color: {TEXT}; }}
QListWidget::item:hover, QTableWidget::item:hover {{ background-color: {PANEL_2}; }}
QHeaderView::section {{
    background-color: {PANEL}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
    padding: 6px; font-weight: 700; font-size: 11px;
}}
QTableCornerButton::section {{ background-color: {PANEL}; border: none; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {PANEL_3}; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {PANEL_3}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QProgressBar {{
    background-color: {PANEL_2}; border: none; border-radius: 6px; height: 12px; text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT}, stop:1 {YELLOW}); border-radius: 6px; }}
QSlider::groove:horizontal {{ height: 6px; background: {PANEL_3}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: white; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QSplitter::handle {{ background-color: transparent; }}
QStatusBar {{ background-color: {PANEL}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ background: transparent; }}
QMenu {{ background-color: {PANEL_2}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background-color: {PANEL_3}; }}
QMessageBox {{ background-color: {PANEL}; }}
"""


def setup_pyqtgraph():
    pg.setConfigOptions(antialias=True, background=PANEL, foreground=MUTED)


def style_plot(plot_item, title: str = None, left: str = None, bottom: str = None):
    """Common look for a pyqtgraph PlotItem."""
    if title:
        plot_item.setTitle(f"<span style='color:{TEXT}; font-size:11pt; font-weight:700'>{title}</span>")
    for name in ("left", "bottom"):
        axis = plot_item.getAxis(name)
        axis.setPen(pg.mkPen(BORDER))
        axis.setTextPen(pg.mkPen(MUTED))
    if left:
        plot_item.setLabel("left", left, color=MUTED)
    if bottom:
        plot_item.setLabel("bottom", bottom, color=MUTED)
    plot_item.showGrid(x=True, y=True, alpha=0.12)
