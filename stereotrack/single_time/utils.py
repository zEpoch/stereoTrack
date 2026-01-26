from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
import warnings


def nearest_neighbors(coord, coords, n_neighbors=5):
    """
    Find nearest neighbors for a given coordinate.
    Supports both 2D and 3D coordinates.
    """
    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(coords)
    _, neighs = neigh.kneighbors(np.atleast_2d(coord))
    return neighs


def validate_spatial_coords(coords: np.ndarray, expected_dim: int = None) -> int:
    """
    Validate spatial coordinates dimension.
    
    Parameters
    ----------
    coords
        Spatial coordinates array (n_cells, n_dims).
    expected_dim
        Expected dimension (2 for 2D, 3 for 3D). If None, accepts both.
        (Default: None)
    
    Returns
    -------
    int
        The dimension of coordinates (2 or 3).
    
    Raises
    ------
    ValueError
        If coordinates have invalid dimension.
    """
    if coords.ndim != 2:
        raise ValueError(f"Coordinates must be 2D array (n_cells, n_dims), got {coords.ndim}D")
    
    dim = coords.shape[1]
    if dim not in [2, 3]:
        raise ValueError(f"Spatial coordinates must be 2D (x,y) or 3D (x,y,z), got {dim}D")
    
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(f"Expected {expected_dim}D coordinates, got {dim}D")
    
    return dim


def build_spatial_graph(coords: np.ndarray, method: str = 'auto', n_neighbors: int = 15):
    """
    Build spatial graph using Delaunay triangulation (for 3D) or k-nearest neighbors (for 2D).
    
    Parameters
    ----------
    coords
        Spatial coordinates array (n_cells, n_dims). Can be 2D (x,y) or 3D (x,y,z).
    method
        Method to build graph: 'auto' (Delaunay for 3D, kNN for 2D), 'delaunay', or 'knn'.
        (Default: 'auto')
    n_neighbors
        Number of neighbors for kNN method (only used when method='knn' or for 2D).
        (Default: 15)
    
    Returns
    -------
    scipy.sparse.csr_matrix
        Adjacency matrix of the spatial graph.
    """
    validate_spatial_coords(coords)
    dim = coords.shape[1]
    n_cells = coords.shape[0]
    
    if method == 'auto':
        method = 'delaunay' if dim == 3 else 'knn'
    
    if method == 'delaunay':
        if dim == 2:
            # 2D Delaunay triangulation
            tri = Delaunay(coords)
            edges = set()
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        edges.add((simplex[i], simplex[j]))
        elif dim == 3:
            # 3D Delaunay triangulation (tetrahedralization)
            tri = Delaunay(coords)
            edges = set()
            for simplex in tri.simplices:
                # Each simplex is a tetrahedron with 4 vertices
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        edges.add((simplex[i], simplex[j]))
        else:
            raise ValueError(f"Delaunay triangulation not supported for {dim}D coordinates")
        
        # Build adjacency matrix
        row_indices = []
        col_indices = []
        for edge in edges:
            row_indices.extend([edge[0], edge[1]])
            col_indices.extend([edge[1], edge[0]])
        
        adjacency = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), 
                              shape=(n_cells, n_cells))
        
    elif method == 'knn':
        # k-nearest neighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(coords)
        distances, indices = nn.kneighbors(coords)
        
        row_indices = []
        col_indices = []
        for i in range(n_cells):
            for j in indices[i][1:]:  # Skip self (first neighbor)
                row_indices.append(i)
                col_indices.append(j)
        
        adjacency = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), 
                              shape=(n_cells, n_cells))
        # Make symmetric
        adjacency = (adjacency + adjacency.T > 0).astype(float)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'auto', 'delaunay', or 'knn'")
    
    return adjacency


def kmeans_centers(coords: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    """
    Use kmeans to find cluster centers based on coordinates.
    Supports both 2D and 3D coordinates.

    Parameters
    ----------
    coords
        Coordinates stored in a 2D array (n_cells, n_dims). Can be 2D (x,y) or 3D (x,y,z).
    n_clusters
        The number of cluster centers.
        (Default: 2)

    Returns
    -------
    np.ndarray
        Centers of clusters.
    """
    validate_spatial_coords(coords)
    cell_coordinates = coords
    kmeans = KMeans(n_clusters, random_state=8)
    kmeans.fit(cell_coordinates)
    cluster_centers = kmeans.cluster_centers_
    print("kmeans cluster centers:")
    list(map(print, cluster_centers))
    return cluster_centers
