import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import Button, LassoSelector
from matplotlib.lines import Line2D


class AdataLassoSelector:
    """
    Interactive lasso selector for AnnData spatial coordinates in Jupyter.

    Recommended Jupyter backend:
        %matplotlib widget

    Basic usage:
        selector = AdataLassoSelector(adata, spatial_key="spatial")
        selector.show()
        adata_selected = selector.to_adata()
    """

    def __init__(
        self,
        adata,
        spatial_key="spatial",
        color=None,
        color_map=None,
        cmap="viridis",
        na_color="#D9D9D9",
        point_size=2,
        alpha=0.75,
        selected_color="#E64B35",
        selected_edge_color="#111111",
        selected_linewidth=0.8,
        selected_size=None,
        unselected_color="#B8B8B8",
        invert_y=False,
        title=None,
        legend=True,
        max_legend_categories=30,
    ):
        if spatial_key not in adata.obsm:
            raise KeyError(f"adata.obsm does not contain spatial_key={spatial_key!r}")

        coords = np.asarray(adata.obsm[spatial_key])
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"adata.obsm[{spatial_key!r}] must be an n_cells x >=2 array.")
        if coords.shape[0] != adata.n_obs:
            raise ValueError("Spatial coordinates rows must match adata.n_obs.")

        self.adata = adata
        self.spatial_key = spatial_key
        self.coords = coords[:, :2].astype(float)
        self.color = color
        self.color_map = color_map or {}
        self.cmap = cmap
        self.na_color = na_color
        self.point_size = point_size
        self.alpha = alpha
        self.selected_color = selected_color
        self.selected_edge_color = selected_edge_color
        self.selected_linewidth = selected_linewidth
        self.selected_size = selected_size if selected_size is not None else max(point_size * 5, 8)
        self.unselected_color = unselected_color
        self.invert_y = invert_y
        self.title = title or f"Lasso cells from adata.obsm[{spatial_key!r}]"
        self.legend = legend
        self.max_legend_categories = max_legend_categories
        self._color_is_numeric = False
        self._category_color_map = {}

        self.selected_mask = np.zeros(adata.n_obs, dtype=bool)
        self.last_mask = np.zeros(adata.n_obs, dtype=bool)
        self.mode = "add"

        self.fig = None
        self.ax = None
        self.scatter = None
        self.selected_scatter = None
        self.lasso = None
        self.status_text = None

    def _base_colors(self):
        if self.color is None:
            self._color_is_numeric = False
            self._category_color_map = {}
            return np.full(self.adata.n_obs, self.unselected_color, dtype=object)

        if self.color not in self.adata.obs:
            raise KeyError(f"adata.obs does not contain color={self.color!r}")

        values = self.adata.obs[self.color]
        if pd.api.types.is_numeric_dtype(values) and not self.color_map:
            self._color_is_numeric = True
            self._category_color_map = {}
            return values.to_numpy()

        self._color_is_numeric = False
        labels = values.astype("string").fillna("NA").astype(str)
        categories = list(pd.unique(labels))

        if self.color_map:
            mapping = {str(k): v for k, v in self.color_map.items()}
            colors = labels.map(lambda x: mapping.get(str(x), self.na_color)).to_numpy()
            self._category_color_map = {
                cat: mapping.get(str(cat), self.na_color) for cat in categories
            }
            return colors

        tab = plt.get_cmap("tab20")
        mapping = {cat: tab(i % 20) for i, cat in enumerate(categories)}
        self._category_color_map = mapping
        return labels.map(mapping).to_numpy()

    def _draw_colors(self):
        return self._base_colors()

    def _update_status(self):
        n_selected = int(self.selected_mask.sum())
        n_last = int(self.last_mask.sum())
        msg = (
            f"mode: {self.mode} | selected: {n_selected:,}/{self.adata.n_obs:,} "
            f"| last lasso: {n_last:,}"
        )
        if self.status_text is not None:
            self.status_text.set_text(msg)
        print(msg)

    def _refresh(self):
        if self.scatter is None:
            return
        if not self._color_is_numeric:
            self.scatter.set_color(self._draw_colors())
        if self.selected_scatter is not None:
            pts = self.coords[self.selected_mask]
            if len(pts) == 0:
                pts = np.empty((0, 2))
            self.selected_scatter.set_offsets(pts)
        self._update_status()
        self.fig.canvas.draw_idle()

    def _add_legend(self):
        if not self.legend or not self._category_color_map:
            return
        if len(self._category_color_map) > self.max_legend_categories:
            print(
                f"Skip legend: {len(self._category_color_map)} categories "
                f"> max_legend_categories={self.max_legend_categories}"
            )
            return

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=6,
                label=str(label),
            )
            for label, color in self._category_color_map.items()
        ]
        self.ax.legend(
            handles=handles,
            title=self.color,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )

    def _on_select(self, verts):
        path = Path(verts)
        mask = path.contains_points(self.coords)
        self.last_mask = mask

        if self.mode == "add":
            self.selected_mask |= mask
        elif self.mode == "remove":
            self.selected_mask &= ~mask
        elif self.mode == "replace":
            self.selected_mask = mask.copy()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self._refresh()

    def _set_mode(self, mode):
        self.mode = mode
        self._refresh()

    def _clear(self, _event=None):
        self.selected_mask[:] = False
        self.last_mask[:] = False
        self._refresh()

    def _undo_last(self, _event=None):
        if self.mode == "add":
            self.selected_mask &= ~self.last_mask
        elif self.mode == "remove":
            self.selected_mask |= self.last_mask
        elif self.mode == "replace":
            self.selected_mask[:] = False
        self.last_mask[:] = False
        self._refresh()

    def show(self, figsize=(7, 7)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(bottom=0.18)

        x = self.coords[:, 0]
        y = self.coords[:, 1]
        base_colors = self._draw_colors()
        scatter_kwargs = dict(
            x=x,
            y=y,
            s=self.point_size,
            c=base_colors,
            alpha=self.alpha,
            linewidths=0,
            rasterized=True,
        )
        if self._color_is_numeric:
            scatter_kwargs["cmap"] = self.cmap
        self.scatter = self.ax.scatter(**scatter_kwargs)

        self.selected_scatter = self.ax.scatter(
            [],
            [],
            s=self.selected_size,
            facecolors="none",
            edgecolors=self.selected_edge_color,
            linewidths=self.selected_linewidth,
            alpha=1.0,
        )

        self.ax.set_title(self.title)
        self.ax.set_xlabel("spatial_1")
        self.ax.set_ylabel("spatial_2")
        self.ax.set_aspect("equal", adjustable="box")
        if self.invert_y:
            self.ax.invert_yaxis()
        self._add_legend()

        self.status_text = self.ax.text(
            0.01,
            1.01,
            "",
            transform=self.ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#555555",
        )

        ax_add = self.fig.add_axes([0.12, 0.05, 0.13, 0.05])
        ax_remove = self.fig.add_axes([0.27, 0.05, 0.13, 0.05])
        ax_replace = self.fig.add_axes([0.42, 0.05, 0.13, 0.05])
        ax_undo = self.fig.add_axes([0.57, 0.05, 0.13, 0.05])
        ax_clear = self.fig.add_axes([0.72, 0.05, 0.13, 0.05])

        self.btn_add = Button(ax_add, "Add")
        self.btn_remove = Button(ax_remove, "Remove")
        self.btn_replace = Button(ax_replace, "Replace")
        self.btn_undo = Button(ax_undo, "Undo")
        self.btn_clear = Button(ax_clear, "Clear")

        self.btn_add.on_clicked(lambda event: self._set_mode("add"))
        self.btn_remove.on_clicked(lambda event: self._set_mode("remove"))
        self.btn_replace.on_clicked(lambda event: self._set_mode("replace"))
        self.btn_undo.on_clicked(self._undo_last)
        self.btn_clear.on_clicked(self._clear)

        self.lasso = LassoSelector(self.ax, onselect=self._on_select)
        self._update_status()
        plt.show()
        return self

    def selected_obs_names(self):
        return self.adata.obs_names[self.selected_mask].copy()

    def to_adata(self, copy=True):
        if copy:
            return self.adata[self.selected_mask].copy()
        return self.adata[self.selected_mask]

    def save_selected_obs_names(self, path):
        path = str(path)
        pd.Series(self.selected_obs_names(), name="obs_name").to_csv(path, index=False)
        return path

    def load_selected_obs_names(self, path):
        names = pd.read_csv(path)["obs_name"].astype(str)
        self.selected_mask = np.asarray(self.adata.obs_names.isin(names), dtype=bool)
        self._refresh()
        return self
