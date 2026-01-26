from .velocity import (
    set_start_cells,
    get_ot_matrix,
    get_ot_matrix_gnn,
    get_ptime,
    get_velocity,
    Lasso,
    auto_get_start_cluster,
    get_velocity_grid,
    auto_estimate_para,
    calc_alpha_by_moransI,
)

from .lap import (
    least_action,
    map_cell_to_LAP,
    plot_least_action_path,
)

from .vectorfield import VectorField

from .Pgene import (
    filter_gene,
    ptime_gene_GAM,
    order_trajectory_genes,
    plot_trajectory_gene_heatmap,
    plot_trajectory_gene,
)

from .start_cluster import (

   assess_start_cluster,
   assess_start_cluster_plot
   )

from .utils import nearest_neighbors, build_spatial_graph, validate_spatial_coords

# GNN models (optional)
try:
    from .gnn_models import (
        SpatialGATAutoEncoder,
        get_gnn_embedding,
        get_imputed_expression,
        train_gnn_imputation,
        prepare_graph_data,
        run_3d_gnn_imputation,
    )
except ImportError:
    pass

# 3D visualization (optional)
try:
    from .visualization_3d import (
        plot_3d_k3d,
        plot_lap_3d_k3d,
        plot_3d_plotly,
        plot_lap_3d_plotly,
    )
except ImportError:
    pass

from .gene_regulation import Trainer

__version__ = '0.1.1'
