"""Container widget for a list of conditions joined by AND/OR."""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup, QGroupBox, QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QVBoxLayout, QWidget,
)

from ..analytics.signals import RuleConfig
from .condition_widget import ConditionWidget
from .signal_builder import SignalBuilderDialog
from .signal_library import SignalLibrary


class RuleWidget(QGroupBox):
    """Holds a list of ``ConditionWidget`` rows + AND/OR + add buttons."""

    changed = pyqtSignal()
    condition_selected = pyqtSignal(object)   # ConditionWidget

    def __init__(self, title: str, library: SignalLibrary,
                 get_ticker=None, parent=None):
        super().__init__(title, parent)
        self._library = library
        # Optional callable returning the active ticker for live signal preview
        self._get_ticker = get_ticker

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # AND/OR radio
        combine_row = QHBoxLayout()
        combine_row.addWidget(QLabel("Combine:"))
        self.rb_and = QRadioButton("AND")
        self.rb_or = QRadioButton("OR")
        self.rb_and.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_and); bg.addButton(self.rb_or)
        self.rb_and.toggled.connect(lambda _: self.changed.emit())
        self.rb_or.toggled.connect(lambda _: self.changed.emit())
        combine_row.addWidget(self.rb_and)
        combine_row.addWidget(self.rb_or)
        combine_row.addStretch(1)

        self.add_cond_btn = QPushButton("+ Condition")
        self.add_cond_btn.clicked.connect(lambda: self._add_condition())
        self.add_signal_btn = QPushButton("+ Custom signal…")
        self.add_signal_btn.clicked.connect(self._on_add_custom_signal)
        combine_row.addWidget(self.add_cond_btn)
        combine_row.addWidget(self.add_signal_btn)
        root.addLayout(combine_row)

        self._cond_box = QWidget()
        self._cond_layout = QVBoxLayout(self._cond_box)
        self._cond_layout.setContentsMargins(0, 0, 0, 0)
        self._cond_layout.setSpacing(2)
        self._cond_layout.setAlignment(Qt.AlignTop)
        root.addWidget(self._cond_box)

        self._conditions: list[ConditionWidget] = []

    def _add_condition(self) -> ConditionWidget:
        w = ConditionWidget(self._library, self)
        w.changed.connect(self.changed.emit)
        w.remove_requested.connect(self._remove_condition)
        w.selected.connect(self._on_condition_selected)
        self._cond_layout.addWidget(w)
        self._conditions.append(w)
        self.changed.emit()
        return w

    def _remove_condition(self, w: ConditionWidget) -> None:
        if w in self._conditions:
            self._conditions.remove(w)
        self._cond_layout.removeWidget(w)
        w.setParent(None)
        w.deleteLater()
        self.changed.emit()

    def _on_condition_selected(self, w: ConditionWidget):
        self.condition_selected.emit(w)

    def _on_add_custom_signal(self):
        ticker = self._get_ticker() if self._get_ticker else None
        dlg = SignalBuilderDialog(self._library, ticker=ticker, parent=self)
        if dlg.exec_() == dlg.Accepted:
            entry = dlg.result_entry()
            if entry is not None:
                try:
                    self._library.add(entry)
                except ValueError:
                    return
                # Refresh dropdowns in all condition rows
                for c in self._conditions:
                    c.refresh_library()

    # -- API --
    def refresh_library(self):
        for c in self._conditions:
            c.refresh_library()

    def to_config(self) -> RuleConfig:
        conds = []
        for c in self._conditions:
            cfg = c.to_config()
            if cfg is not None:
                conds.append(cfg)
        return RuleConfig(conditions=conds,
                          combine="AND" if self.rb_and.isChecked() else "OR")

    def set_from_config(self, cfg: RuleConfig) -> None:
        # Clear existing
        for w in list(self._conditions):
            self._remove_condition(w)
        self.rb_and.setChecked(cfg.combine != "OR")
        self.rb_or.setChecked(cfg.combine == "OR")
        for c in cfg.conditions:
            row = self._add_condition()
            row.set_from_config(c)

    def conditions(self) -> list[ConditionWidget]:
        return list(self._conditions)
