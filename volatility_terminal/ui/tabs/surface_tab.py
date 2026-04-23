"""3D IV surface (log-moneyness x tau x IV) via pyqtgraph.opengl."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QVBoxLayout, QWidget

try:
    import pyqtgraph.opengl as gl
    _HAS_GL = True
except Exception:
    _HAS_GL = False

import pyqtgraph as pg

from ...pricing.surface import build_surface


class SurfaceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _HAS_GL:
            from PyQt5.QtWidgets import QLabel
            layout.addWidget(QLabel(
                "3D surface requires PyOpenGL. Install with `pip install PyOpenGL`."
            ))
            self.view = None
            return

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("#111")
        self.view.opts["distance"] = 3.0
        layout.addWidget(self.view)

        grid = gl.GLGridItem()
        grid.setSize(2, 2, 0)
        grid.setSpacing(0.1, 0.1, 0.1)
        self.view.addItem(grid)

        self.surface_item = None
        self.scatter_item = None

    def set_chain(self, chain: pd.DataFrame):
        if self.view is None:
            return
        for item in (self.surface_item, self.scatter_item):
            if item is not None:
                self.view.removeItem(item)
        self.surface_item = None
        self.scatter_item = None

        grid = build_surface(chain)
        k = grid["k_grid"]; tau = grid["tau_grid"]; Z = grid["iv_grid"]
        if k.size == 0 or tau.size == 0 or Z.size == 0:
            return

        # Normalize axes to -1..1 for nice viewing; keep IV in 0..1 roughly.
        x = np.linspace(-1, 1, len(k))
        y = np.linspace(-1, 1, len(tau))
        z = np.where(np.isfinite(Z), Z, np.nanmean(Z))
        z = np.clip(z, 0, 2.0)

        cmap = pg.colormap.get("viridis")
        z_norm = (z - np.nanmin(z)) / max(np.nanmax(z) - np.nanmin(z), 1e-9)
        colors = cmap.map(z_norm.flatten(), mode="float").reshape(*z.shape, 4)

        self.surface_item = gl.GLSurfacePlotItem(
            x=x, y=y, z=z, colors=colors.reshape(-1, 4),
            shader="shaded", smooth=True,
        )
        self.surface_item.translate(0, 0, -0.5)
        self.view.addItem(self.surface_item)
