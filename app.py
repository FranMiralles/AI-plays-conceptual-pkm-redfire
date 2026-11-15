"""Interfaz gráfica: entrenar el algoritmo genético, ver su evolución y explorar individuos.

Uso:
    python app.py                      # abre la aplicación
    python app.py runs/<run>/best.json # abre directamente un individuo en el visor
"""
import multiprocessing
import sys


def main():
    import json

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)

    from ui import theme
    from ui.main_window import MainWindow

    theme.setup_pyqtgraph()
    app.setStyleSheet(theme.STYLE_SHEET)
    window = MainWindow()
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as fh:
            window.show_individual(json.load(fh), sys.argv[1])
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
