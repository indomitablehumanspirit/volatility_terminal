"""QRunnable workers for off-UI-thread fetch / compute."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot


class WorkerSignals(QObject):
    finished = pyqtSignal(object)   # payload
    failed = pyqtSignal(str)        # error message
    progress = pyqtSignal(int, int, str)  # current, total, message


class Worker(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, progress_cb=self.signals.progress.emit,
                             **self.kwargs) if self._accepts_progress() \
                else self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            import traceback
            self.signals.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def _accepts_progress(self) -> bool:
        try:
            import inspect
            return "progress_cb" in inspect.signature(self.fn).parameters
        except (TypeError, ValueError):
            return False
