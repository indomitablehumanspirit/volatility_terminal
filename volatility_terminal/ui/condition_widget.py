"""One row of an entry/exit rule: signal | op | threshold | remove."""
from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QPushButton, QWidget,
)

from ..analytics.signals import ConditionConfig
from .signal_library import SignalLibrary


OP_CHOICES = [
    (">", ">"), ("<", "<"), ("≥", ">="), ("≤", "<="),
    ("cross↑", "cross_up"), ("cross↓", "cross_down"),
]


class ConditionWidget(QWidget):
    """A single condition row. Emits ``changed`` when any control changes;
    ``selected`` when the user clicks anywhere on this row (used to drive the
    signal-preview chart in the parent tab).
    """
    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)   # emits self
    selected = pyqtSignal(object)           # emits self

    def __init__(self, library: SignalLibrary, parent=None):
        super().__init__(parent)
        self._library = library

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        self.signal_combo = QComboBox()
        self.signal_combo.setMinimumWidth(220)
        self._refresh_library()
        self.signal_combo.currentIndexChanged.connect(self._on_changed)

        self.op_combo = QComboBox()
        for label, val in OP_CHOICES:
            self.op_combo.addItem(label, val)
        self.op_combo.currentIndexChanged.connect(self._on_changed)

        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setDecimals(4)
        self.thr_spin.setRange(-1e6, 1e6)
        self.thr_spin.setSingleStep(0.01)
        self.thr_spin.valueChanged.connect(self._on_changed)

        self.remove_btn = QPushButton("✕")
        self.remove_btn.setFixedWidth(28)
        self.remove_btn.setToolTip("Remove condition")
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))

        layout.addWidget(self.signal_combo, 3)
        layout.addWidget(self.op_combo, 1)
        layout.addWidget(self.thr_spin, 1)
        layout.addWidget(self.remove_btn)

    def mousePressEvent(self, ev):
        self.selected.emit(self)
        return super().mousePressEvent(ev)

    def _refresh_library(self):
        prev = self.signal_combo.currentText()
        self.signal_combo.blockSignals(True)
        self.signal_combo.clear()
        for name in self._library.names():
            self.signal_combo.addItem(name)
        # Restore previous selection if still present
        idx = self.signal_combo.findText(prev)
        if idx >= 0:
            self.signal_combo.setCurrentIndex(idx)
        self.signal_combo.blockSignals(False)

    def refresh_library(self):
        self._refresh_library()

    def _on_changed(self, *_):
        self.changed.emit()
        self.selected.emit(self)

    # -- public --
    def to_config(self) -> ConditionConfig | None:
        name = self.signal_combo.currentText()
        entry = self._library.get(name)
        if entry is None:
            return None
        return ConditionConfig(
            signal=entry.signal.to_dict(),
            op=self.op_combo.currentData(),
            threshold=float(self.thr_spin.value()),
        )

    def set_from_config(self, cfg: ConditionConfig) -> None:
        # Try to find a library entry matching the signal dict; fallback: leave
        # the dropdown alone but set op + threshold.
        target = cfg.signal
        for name in self._library.names():
            e = self._library.get(name)
            if e is not None and e.signal.to_dict() == target:
                idx = self.signal_combo.findText(name)
                if idx >= 0:
                    self.signal_combo.setCurrentIndex(idx)
                break
        for i in range(self.op_combo.count()):
            if self.op_combo.itemData(i) == cfg.op:
                self.op_combo.setCurrentIndex(i); break
        self.thr_spin.setValue(cfg.threshold)

    def selected_signal_name(self) -> str:
        return self.signal_combo.currentText()

    def selected_threshold(self) -> float:
        return float(self.thr_spin.value())
