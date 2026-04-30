"""Modal dialog to add a new entry to the SignalLibrary.

Two modes:
  - Primitive: pick a primitive type and edit its params.
  - Composite: pick two primitive types, each with its own params, and an
    operator. Optional smoother on the result of either mode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QSpinBox, QStackedWidget,
    QVBoxLayout, QWidget,
)

from ..analytics.signals import (
    AtmIvSignal, ButterflySignal, CompositeSignal, ForwardVolSignal,
    IvRvRatioSignal, RealizedVolSignal, RiskReversalSignal, Signal,
    SmoothedSignal, TransformedSignal, VrpSignal,
)
from .signal_library import LibraryEntry, SignalLibrary


PRIMITIVE_TYPES = [
    ("ATM IV", "atm_iv"),
    ("Realized vol", "realized_vol"),
    ("IV/RV ratio", "iv_rv_ratio"),
    ("VRP (IV - RV)", "vrp"),
    ("Risk reversal", "risk_reversal"),
    ("Butterfly", "butterfly"),
    ("Forward vol", "forward_vol"),
]

COMPOSITE_OPS = [("Difference (a − b)", "diff"), ("Ratio (a / b)", "ratio"),
                 ("Sum (a + b)", "sum"), ("Product (a · b)", "product")]
SMOOTH_KINDS = [
    ("None", ""),
    ("SMA (mean)", "SMA"),
    ("EMA (exponential mean)", "EMA"),
    ("Median (drops single spikes)", "MEDIAN"),
    ("Hampel (rejects outliers, k=3)", "HAMPEL"),
]

TRANSFORM_KINDS = [
    ("None", ""),
    ("Rank 0–100 (min/max)", "RANK"),
    ("Percentile 0–100", "PERCENTILE"),
    ("Z-score (σ-units)", "ZSCORE"),
]


class PrimitiveSignalPicker(QGroupBox):
    """Self-contained primitive picker: type combo + per-type params.

    Shows only the params relevant to the selected primitive type. ``build()``
    returns a fresh ``Signal`` instance. Emits ``changed`` whenever any input
    is modified — used by the dialog to drive a live preview.
    """

    changed = pyqtSignal()

    def __init__(self, title: str = "", parent=None):
        super().__init__(title, parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 6)

        self.type_combo = QComboBox()
        for label, kind in PRIMITIVE_TYPES:
            self.type_combo.addItem(label, kind)
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        type_row.addWidget(self.type_combo, 1)
        outer.addLayout(type_row)

        # Params row — show/hide as needed
        params_widget = QWidget()
        self._form = QFormLayout(params_widget)
        self._form.setContentsMargins(0, 0, 0, 0)

        self.dte = QSpinBox(); self.dte.setRange(1, 730); self.dte.setValue(30)
        self.window = QSpinBox(); self.window.setRange(2, 500); self.window.setValue(30)
        self.delta = QDoubleSpinBox()
        self.delta.setRange(0.01, 0.50); self.delta.setSingleStep(0.05)
        self.delta.setDecimals(2); self.delta.setValue(0.25)
        self.dte1 = QSpinBox(); self.dte1.setRange(1, 730); self.dte1.setValue(30)
        self.dte2 = QSpinBox(); self.dte2.setRange(1, 730); self.dte2.setValue(60)

        self._rows = {
            "dte":    (QLabel("DTE:"),     self.dte),
            "window": (QLabel("Window:"),  self.window),
            "delta":  (QLabel("|Δ|:"),     self.delta),
            "dte1":   (QLabel("DTE 1:"),   self.dte1),
            "dte2":   (QLabel("DTE 2:"),   self.dte2),
        }
        for lbl, w in self._rows.values():
            self._form.addRow(lbl, w)

        outer.addWidget(params_widget)

        self.type_combo.currentIndexChanged.connect(self._refresh_params)
        self.type_combo.currentIndexChanged.connect(self.changed.emit)
        for w in (self.dte, self.window, self.dte1, self.dte2):
            w.valueChanged.connect(self.changed.emit)
        self.delta.valueChanged.connect(self.changed.emit)
        self._refresh_params()

    def _refresh_params(self):
        for lbl, w in self._rows.values():
            lbl.setVisible(False); w.setVisible(False)
        kind = self.type_combo.currentData()
        show_keys = {
            "atm_iv":        ["dte"],
            "realized_vol":  ["window"],
            "iv_rv_ratio":   ["dte", "window"],
            "vrp":           ["dte", "window"],
            "risk_reversal": ["dte", "delta"],
            "butterfly":     ["dte", "delta"],
            "forward_vol":   ["dte1", "dte2"],
        }.get(kind, [])
        for key in show_keys:
            lbl, w = self._rows[key]
            lbl.setVisible(True); w.setVisible(True)

    def build(self) -> Signal:
        kind = self.type_combo.currentData()
        if kind == "atm_iv":
            return AtmIvSignal(self.dte.value())
        if kind == "realized_vol":
            return RealizedVolSignal(self.window.value())
        if kind == "iv_rv_ratio":
            return IvRvRatioSignal(self.dte.value(), self.window.value())
        if kind == "vrp":
            return VrpSignal(self.dte.value(), self.window.value())
        if kind == "risk_reversal":
            return RiskReversalSignal(self.dte.value(), self.delta.value())
        if kind == "butterfly":
            return ButterflySignal(self.dte.value(), self.delta.value())
        if kind == "forward_vol":
            return ForwardVolSignal(self.dte1.value(), self.dte2.value())
        raise ValueError(kind)


class SignalBuilderDialog(QDialog):
    """Compose a new library entry. Returns the created LibraryEntry on accept."""

    def __init__(self, library: SignalLibrary, ticker: str | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Build signal")
        self.setMinimumWidth(720)
        # Cap height so the dialog always fits even on smaller screens; the
        # body scrolls if its natural height exceeds this.
        self.resize(760, 600)
        self._library = library
        self._result: LibraryEntry | None = None
        self._initial_ticker = (ticker or "").upper().strip() or None

        # Debounce timer for preview refresh
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(250)
        self._refresh_timer.timeout.connect(self._refresh_preview)

        # The dialog itself just stacks: scrollable body + always-visible buttons
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self._scroll, 1)

        body = QWidget()
        self._scroll.setWidget(body)
        root = QVBoxLayout(body)
        root.setContentsMargins(8, 8, 8, 8)

        # Mode radio
        mode_row = QHBoxLayout()
        self.rb_prim = QRadioButton("Primitive")
        self.rb_comp = QRadioButton("Composite (a ⊙ b)")
        self.rb_prim.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_prim); bg.addButton(self.rb_comp)
        mode_row.addWidget(self.rb_prim)
        mode_row.addWidget(self.rb_comp)
        mode_row.addStretch(1)
        root.addLayout(mode_row)

        self.stack = QStackedWidget()
        root.addWidget(self.stack)

        # Page 0: primitive
        pg_prim = QWidget(); pg_prim_layout = QVBoxLayout(pg_prim)
        pg_prim_layout.setContentsMargins(0, 0, 0, 0)
        self.prim_picker = PrimitiveSignalPicker("Signal")
        pg_prim_layout.addWidget(self.prim_picker)
        pg_prim_layout.addStretch(1)
        self.stack.addWidget(pg_prim)

        # Page 1: composite — two primitive pickers side by side + op
        pg_comp = QWidget(); pg_comp_layout = QVBoxLayout(pg_comp)
        pg_comp_layout.setContentsMargins(0, 0, 0, 0)

        self.left_picker = PrimitiveSignalPicker("Left signal (a)")
        self.right_picker = PrimitiveSignalPicker("Right signal (b)")

        # Default the right picker to a different primitive (Forward vol 60/90)
        # so the very first composite isn't a × b with identical inputs.
        self.right_picker.type_combo.setCurrentIndex(
            self.right_picker.type_combo.findData("forward_vol"))
        self.right_picker.dte1.setValue(60)
        self.right_picker.dte2.setValue(90)
        # And bump the left to forward_vol too so the canonical FW30/FW60 ratio
        # case is one click away.
        self.left_picker.type_combo.setCurrentIndex(
            self.left_picker.type_combo.findData("forward_vol"))
        self.left_picker.dte1.setValue(30)
        self.left_picker.dte2.setValue(60)

        sides_row = QHBoxLayout()
        sides_row.addWidget(self.left_picker, 1)
        sides_row.addWidget(self.right_picker, 1)
        pg_comp_layout.addLayout(sides_row)

        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Operation:"))
        self.op_combo = QComboBox()
        for label, kind in COMPOSITE_OPS:
            self.op_combo.addItem(label, kind)
        self.op_combo.setCurrentIndex(self.op_combo.findData("ratio"))
        op_row.addWidget(self.op_combo)
        op_row.addStretch(1)
        pg_comp_layout.addLayout(op_row)
        pg_comp_layout.addStretch(1)
        self.stack.addWidget(pg_comp)

        self.rb_prim.toggled.connect(lambda ok: ok and self.stack.setCurrentIndex(0))
        self.rb_comp.toggled.connect(lambda ok: ok and self.stack.setCurrentIndex(1))

        # Transform (relative-to-history wrapper) — applied before smoothing
        tx_row = QHBoxLayout()
        tx_row.addWidget(QLabel("Transform:"))
        self.transform_combo = QComboBox()
        for label, kind in TRANSFORM_KINDS:
            self.transform_combo.addItem(label, kind)
        tx_row.addWidget(self.transform_combo)
        tx_row.addWidget(QLabel("Lookback:"))
        self.transform_lookback = QSpinBox()
        self.transform_lookback.setRange(20, 2520)
        self.transform_lookback.setValue(252)
        self.transform_lookback.setSuffix(" d")
        tx_row.addWidget(self.transform_lookback)
        tx_row.addStretch(1)
        root.addLayout(tx_row)

        # Smoother (applied last)
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Smooth:"))
        self.smooth_combo = QComboBox()
        for label, kind in SMOOTH_KINDS:
            self.smooth_combo.addItem(label, kind)
        smooth_row.addWidget(self.smooth_combo)
        smooth_row.addWidget(QLabel("Window:"))
        self.smooth_window = QSpinBox()
        self.smooth_window.setRange(2, 500); self.smooth_window.setValue(5)
        smooth_row.addWidget(self.smooth_window)
        smooth_row.addStretch(1)
        root.addLayout(smooth_row)

        # Name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. FW30/FW60 ratio")
        name_row.addWidget(self.name_edit, 1)
        root.addLayout(name_row)

        # ---- Preview ----
        preview_box = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_box)
        ticker_row = QHBoxLayout()
        ticker_row.addWidget(QLabel("Ticker:"))
        self.preview_ticker = QLineEdit()
        self.preview_ticker.setPlaceholderText("e.g. AAPL")
        if self._initial_ticker:
            self.preview_ticker.setText(self._initial_ticker)
        self.preview_ticker.textChanged.connect(self._schedule_refresh)
        ticker_row.addWidget(self.preview_ticker, 1)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_preview)
        ticker_row.addWidget(self.refresh_btn)
        preview_layout.addLayout(ticker_row)

        self.preview_plot = pg.PlotWidget()
        self.preview_plot.setBackground("#111")
        self.preview_plot.showGrid(x=True, y=True, alpha=0.3)
        self.preview_plot.setAxisItems({"bottom": pg.DateAxisItem()})
        self.preview_plot.setLabel("bottom", "Date")
        self.preview_plot.setMinimumHeight(160)
        preview_layout.addWidget(self.preview_plot, 1)

        self.preview_label = QLabel(
            "Enter a ticker to preview the signal series.")
        self.preview_label.setStyleSheet("color: #aaa;")
        self.preview_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_label)
        root.addWidget(preview_box, 1)

        # Wire all inputs to debounced refresh
        self.rb_prim.toggled.connect(self._schedule_refresh)
        self.rb_comp.toggled.connect(self._schedule_refresh)
        self.prim_picker.changed.connect(self._schedule_refresh)
        self.left_picker.changed.connect(self._schedule_refresh)
        self.right_picker.changed.connect(self._schedule_refresh)
        self.op_combo.currentIndexChanged.connect(self._schedule_refresh)
        self.transform_combo.currentIndexChanged.connect(self._schedule_refresh)
        self.transform_lookback.valueChanged.connect(self._schedule_refresh)
        self.smooth_combo.currentIndexChanged.connect(self._schedule_refresh)
        self.smooth_window.valueChanged.connect(self._schedule_refresh)

        # Buttons — pinned outside the scroll area so they're always visible
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        # Initial preview
        if self._initial_ticker:
            self._schedule_refresh()

    # -- preview --
    def _schedule_refresh(self, *_):
        self._refresh_timer.start()

    def _refresh_preview(self):
        self.preview_plot.clear()
        ticker = self.preview_ticker.text().strip().upper()
        if not ticker:
            self.preview_label.setText(
                "Enter a ticker to preview the signal series.")
            return
        try:
            sig = self._build_signal()
        except Exception as e:
            self.preview_label.setText(f"Build error: {e}")
            return
        if sig is None:
            self.preview_label.setText("Could not build signal from inputs.")
            return
        try:
            series = sig.series(ticker)
        except Exception as e:
            self.preview_label.setText(f"Compute error: {e}")
            return
        if series is None or series.empty:
            self.preview_label.setText(
                f"No data for {ticker}. (Run backfill or pick another ticker.)")
            return

        idx = pd.to_datetime(series.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        ys = series.astype(float).to_numpy()
        valid = np.isfinite(ys)
        if not valid.any():
            self.preview_label.setText(
                f"Signal series has no valid points on {ticker}.")
            return
        xs = np.asarray(idx.view("int64") // 1_000_000_000)[valid]
        ys = ys[valid]

        label = sig.label()
        self.preview_plot.plot(
            xs, ys, pen=pg.mkPen("#4aa3ff", width=2), name=label)
        self.preview_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen("#555", style=Qt.DashLine)))
        self.preview_plot.setLabel("left", label)
        self.preview_label.setText(
            f"{label}  •  {len(ys):,} pts  •  "
            f"range [{ys.min():+.4f}, {ys.max():+.4f}]  on {ticker}"
        )

    def _build_signal(self) -> Signal | None:
        if self.rb_prim.isChecked():
            sig = self.prim_picker.build()
        else:
            left = self.left_picker.build()
            right = self.right_picker.build()
            op = self.op_combo.currentData()
            sig = CompositeSignal(left, right, op)
        transform_kind = self.transform_combo.currentData()
        if transform_kind:
            sig = TransformedSignal(sig, transform_kind,
                                    self.transform_lookback.value())
        smooth_kind = self.smooth_combo.currentData()
        if smooth_kind:
            sig = SmoothedSignal(sig, smooth_kind, self.smooth_window.value())
        return sig

    def _on_accept(self):
        name = self.name_edit.text().strip()
        if not name:
            self.name_edit.setStyleSheet("border: 1px solid #c33;")
            return
        existing = self._library.get(name)
        if existing is not None and existing.builtin:
            self.name_edit.setStyleSheet("border: 1px solid #c33;")
            return
        sig = self._build_signal()
        if sig is None:
            return
        self._result = LibraryEntry(name=name, signal=sig, builtin=False)
        self.accept()

    def result_entry(self) -> LibraryEntry | None:
        return self._result
