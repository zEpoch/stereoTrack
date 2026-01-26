"""
3D visualization using k3d and other advanced visualization engines.

This module provides high-quality 3D visualization for spatial transcriptomics data.
"""

import numpy as np
from typing import Optional, Union, List
import warnings

try:
    import k3d
    K3D_AVAILABLE = True
except ImportError:
    K3D_AVAILABLE = False
    warnings.warn("k3d not available. Install with: pip install k3d")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("plotly not available. Install with: pip install plotly")


def plot_3d_k3d(
    coords: np.ndarray,
    values: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    point_size: float = 0.5,
    color_map: str = 'viridis',
    title: str = '3D Spatial Visualization',
    show_plot: bool = True
):
    """
    Create interactive 3D visualization using k3d.
    
    Parameters
    ----------
    coords
        Spatial coordinates (n_cells, 3) - must be 3D.
    values
        Values to color points by (n_cells,).
        (Default: None)
    colors
        Direct color values (n_cells, 3) RGB or (n_cells, 4) RGBA.
        (Default: None)
    point_size
        Size of points.
        (Default: 0.5)
    color_map
        Colormap name if values are provided.
        (Default: 'viridis')
    title
        Plot title.
        (Default: '3D Spatial Visualization')
    show_plot
        Whether to display the plot immediately.
        (Default: True)
    
    Returns
    -------
    k3d.Plot
        k3d plot object.
    """
    if not K3D_AVAILABLE:
        raise ImportError("k3d is required. Install with: pip install k3d")
    
    if coords.shape[1] != 3:
        raise ValueError(f"Coordinates must be 3D, got {coords.shape[1]}D")
    
    # Create plot
    plot = k3d.plot(name=title)
    
    # Determine colors
    if colors is not None:
        if colors.shape[1] == 3:
            # RGB, add alpha channel
            colors_rgba = np.column_stack([colors, np.ones(len(colors))])
        elif colors.shape[1] == 4:
            colors_rgba = colors
        else:
            raise ValueError("colors must be RGB (n, 3) or RGBA (n, 4)")
    elif values is not None:
        # Map values to colors
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
        cmap = cm.get_cmap(color_map)
        colors_rgba = cmap(norm(values))
    else:
        # Default color (blue)
        colors_rgba = np.tile([0, 0, 1, 1], (len(coords), 1))
    
    # Convert colors to uint8
    colors_rgba = (colors_rgba * 255).astype(np.uint8)
    
    # Add points
    plot += k3d.points(
        coords.astype(np.float32),
        colors=colors_rgba,
        point_size=point_size,
        shader='3d'
    )
    
    if show_plot:
        plot.display()
    
    return plot


def plot_lap_3d_k3d(
    coords: np.ndarray,
    lap_path: np.ndarray,
    lap_values: Optional[np.ndarray] = None,
    point_size: float = 0.3,
    line_width: float = 2.0,
    color_map: str = 'hsv',
    title: str = '3D LAP Visualization',
    show_plot: bool = True
):
    """
    Visualize Least Action Path in 3D using k3d.
    
    Parameters
    ----------
    coords
        All cell coordinates (n_cells, 3).
    lap_path
        LAP path coordinates (n_points, 3).
    lap_values
        Values along LAP path for coloring (n_points,).
        (Default: None)
    point_size
        Size of cell points.
        (Default: 0.3)
    line_width
        Width of LAP path line.
        (Default: 2.0)
    color_map
        Colormap for LAP path.
        (Default: 'hsv')
    title
        Plot title.
        (Default: '3D LAP Visualization')
    show_plot
        Whether to display the plot immediately.
        (Default: True)
    
    Returns
    -------
    k3d.Plot
        k3d plot object.
    """
    if not K3D_AVAILABLE:
        raise ImportError("k3d is required. Install with: pip install k3d")
    
    plot = k3d.plot(name=title)
    
    # Plot all cells (light gray)
    plot += k3d.points(
        coords.astype(np.float32),
        colors=np.tile([200, 200, 200, 100], (len(coords), 1)).astype(np.uint8),
        point_size=point_size,
        shader='3d'
    )
    
    # Plot LAP path
    if lap_values is not None:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=lap_values.min(), vmax=lap_values.max())
        cmap = cm.get_cmap(color_map)
        lap_colors = cmap(norm(lap_values))
        lap_colors_rgba = (lap_colors * 255).astype(np.uint8)
    else:
        lap_colors_rgba = np.tile([255, 0, 0, 255], (len(lap_path), 1)).astype(np.uint8)
    
    # LAP path points
    plot += k3d.points(
        lap_path.astype(np.float32),
        colors=lap_colors_rgba,
        point_size=point_size * 2,
        shader='3d'
    )
    
    # LAP path line
    plot += k3d.line(
        lap_path.astype(np.float32),
        colors=lap_colors_rgba,
        width=line_width,
        shader='mesh'
    )
    
    if show_plot:
        plot.display()
    
    return plot


def plot_3d_plotly(
    coords: np.ndarray,
    values: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    point_size: float = 3,
    color_map: str = 'Viridis',
    title: str = '3D Spatial Visualization',
    show_plot: bool = True
):
    """
    Create interactive 3D visualization using Plotly.
    
    Parameters
    ----------
    coords
        Spatial coordinates (n_cells, 3) - must be 3D.
    values
        Values to color points by (n_cells,).
        (Default: None)
    colors
        Direct color values (n_cells,) for categorical coloring.
        (Default: None)
    labels
        Labels for hover text (n_cells,).
        (Default: None)
    point_size
        Size of points.
        (Default: 3)
    color_map
        Colormap name if values are provided.
        (Default: 'Viridis')
    title
        Plot title.
        (Default: '3D Spatial Visualization')
    show_plot
        Whether to display the plot immediately.
        (Default: True)
    
    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure object.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    if coords.shape[1] != 3:
        raise ValueError(f"Coordinates must be 3D, got {coords.shape[1]}D")
    
    # Determine color values
    if values is not None:
        color_values = values
        color_scale = color_map
    elif colors is not None:
        color_values = colors
        color_scale = None
    else:
        color_values = None
        color_scale = None
    
    # Create figure
    fig = go.Figure(data=go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=color_values,
            colorscale=color_scale,
            showscale=(color_values is not None),
            opacity=0.8
        ),
        text=labels,
        hovertemplate='<b>%{text}</b><br>' +
                      'X: %{x:.2f}<br>' +
                      'Y: %{y:.2f}<br>' +
                      'Z: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=800,
        height=600
    )
    
    if show_plot:
        fig.show()
    
    return fig


def plot_lap_3d_plotly(
    coords: np.ndarray,
    lap_path: np.ndarray,
    lap_values: Optional[np.ndarray] = None,
    point_size: float = 2,
    line_width: float = 3,
    color_map: str = 'Rainbow',
    title: str = '3D LAP Visualization',
    show_plot: bool = True
):
    """
    Visualize Least Action Path in 3D using Plotly.
    
    Parameters
    ----------
    coords
        All cell coordinates (n_cells, 3).
    lap_path
        LAP path coordinates (n_points, 3).
    lap_values
        Values along LAP path for coloring (n_points,).
        (Default: None)
    point_size
        Size of cell points.
        (Default: 2)
    line_width
        Width of LAP path line.
        (Default: 3)
    color_map
        Colormap for LAP path.
        (Default: 'Rainbow')
    title
        Plot title.
        (Default: '3D LAP Visualization')
    show_plot
        Whether to display the plot immediately.
        (Default: True)
    
    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure object.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    fig = go.Figure()
    
    # Plot all cells
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color='lightgray',
            opacity=0.3
        ),
        name='Cells',
        showlegend=False
    ))
    
    # Plot LAP path
    fig.add_trace(go.Scatter3d(
        x=lap_path[:, 0],
        y=lap_path[:, 1],
        z=lap_path[:, 2],
        mode='markers+lines',
        marker=dict(
            size=point_size * 2,
            color=lap_values if lap_values is not None else 'red',
            colorscale=color_map if lap_values is not None else None,
            showscale=(lap_values is not None),
            colorbar=dict(title="Action") if lap_values is not None else None
        ),
        line=dict(
            color=lap_values if lap_values is not None else 'red',
            colorscale=color_map if lap_values is not None else None,
            width=line_width
        ),
        name='LAP Path'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=900,
        height=700
    )
    
    if show_plot:
        fig.show()
    
    return fig
