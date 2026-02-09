import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay

def contains_only_integers(arr):
    return np.all(arr % 1 == 0)

def construct_graph(adata,
                    spatial_key='spatial'
                    ):
    if spatial_key not in adata.obsm:
        raise ValueError(f"Spatial coordinates not found in adata.obsm['{spatial_key}']")
    data = adata.obsm[spatial_key]
    tri = Delaunay(data)
    indptr, indices = tri.vertex_neighbor_vertices
    adjacency_matrix = csr_matrix(
        (np.ones_like(indices, dtype=np.float64), indices, indptr),
        shape=(data.shape[0], data.shape[0]),
    )
    adata.obsm["adj"] = adjacency_matrix
    return adata

def preprocess_adj_sparse(adata):
    if  "adj" not in adata.obsm:
        raise ValueError("Adjacency matrix not found in adata.obsm['adj']")
    adj = sp.coo_matrix(adata.obsm["adj"])
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = (
        adj_.dot(degree_mat_inv_sqrt)
        .transpose()
        .dot(degree_mat_inv_sqrt)
        .tocoo()
    )
    adata.obsm[
        "adj_normalized"
    ] = adj_normalized  # sparse_mx_to_torch_sparse_tensor(adj_normalized)
    adata.obsm["adj_normalized"] = adata.obsm["adj_normalized"].tocsr()
    return adata

def get_spatial_input(adata):
    if isinstance(adata.X, np.ndarray):
        adata.obsm["spatial_input"]= csr_matrix(adata.X)
    else:
        adata.obsm["spatial_input"] = adata.X
    return adata
