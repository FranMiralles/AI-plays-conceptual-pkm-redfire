"""Runs the genetic algorithm in a separate process and relays its progress as Qt signals."""
from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from ga.worker_process import run_ga_process


class GARunner(QObject):
    started = pyqtSignal(dict)
    generation = pyqtSignal(dict)
    hall_of_fame = pyqtSignal(list)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)
    message = pyqtSignal(str)
    state_changed = pyqtSignal(str)  # idle | running | paused | stopping

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._queue = None
        self._stop = None
        self._pause = None
        self.state = "idle"
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)

    def is_active(self) -> bool:
        return self._proc is not None

    def _set_state(self, state: str):
        self.state = state
        self.state_changed.emit(state)

    def start(self, config: dict | None = None, resume_dir: str | None = None, generations: int | None = None):
        if self._proc is not None:
            return
        self._queue = self._ctx.Queue()
        self._stop = self._ctx.Event()
        self._pause = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=run_ga_process,
            args=(config, None, resume_dir, generations, self._queue, self._stop, self._pause),
            daemon=True,
        )
        self._proc.start()
        self._timer.start()
        self._set_state("running")

    def pause(self):
        if self._proc is not None and self.state == "running":
            self._pause.set()
            self._set_state("paused")

    def resume(self):
        if self._proc is not None and self.state == "paused":
            self._pause.clear()
            self._set_state("running")

    def stop(self):
        if self._proc is not None:
            self._stop.set()
            self._pause.clear()
            self._set_state("stopping")

    def shutdown(self, timeout: float = 5.0):
        """Stops the run (it saves a checkpoint) and waits for the process."""
        if self._proc is None:
            return
        self.stop()
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.terminate()
        self._cleanup()

    def _cleanup(self):
        self._timer.stop()
        self._proc = None
        self._set_state("idle")

    def _poll(self):
        batch = []
        try:
            for _ in range(200):
                batch.append(self._queue.get_nowait())
        except queue_mod.Empty:
            pass
        # Many generations may arrive at once with fast presets: emit all of them (charts
        # redraw per signal but stay cheap) except the oldest population samples.
        for kind, payload in batch:
            if kind == "generation":
                self.generation.emit(payload)
            elif kind == "hof":
                self.hall_of_fame.emit(payload)
            elif kind == "started":
                self.started.emit(payload)
            elif kind == "finished":
                self.finished.emit(payload)
                self._finish()
                return
            elif kind == "error":
                self.failed.emit(payload)
                self._finish()
                return
            elif kind == "log":
                self.message.emit(payload)
        if self._proc is not None and not self._proc.is_alive() and not batch:
            self.failed.emit("El proceso del algoritmo terminó inesperadamente.")
            self._finish()

    def _finish(self):
        if self._proc is not None:
            self._proc.join(2)
        self._cleanup()
