import numpy as np

from src.shape_utils import polygonize_edge

def estimate_curvature(p0, p1, p2):
    """
    p0, p1, p2: np.arrays of shape (2,) (y, x)
    return: curvature in point p1
    """
    eps = 1e-3
    det = np.linalg.det([
        [p0[0] - p1[0], p2[0] - p1[0]],
        [p0[1] - p1[1], p2[1] - p1[1]]
    ])
    norm01 = np.linalg.norm(p1 - p0)
    norm02 = np.linalg.norm(p2 - p0)
    norm12 = np.linalg.norm(p1 - p2)
    return -2 * det / (norm01 * norm02 * norm12 + eps)

def polygon2curvatures(polygon):
    curvatures = np.array([
        (estimate_curvature(polygon[i - 6], polygon[i], polygon[i + 6]) + 
        estimate_curvature(polygon[i - 3], polygon[i], polygon[i + 3]) +  
        estimate_curvature(polygon[i - 4], polygon[i], polygon[i + 4]) +  
         estimate_curvature(polygon[i - 2], polygon[i], polygon[i + 2]))/4
#          estimate_curvature(polygon[i - 1], polygon[i], polygon[i + 1])) / 4
        if i > 10 and i < polygon.shape[0] - 10
        else 0 
        for i in range(polygon.shape[0])
    ])
    return curvatures

def curvatures_on_edge(edge_coords, polygon, curvatures):
    """
    edge_coords: np.array, coordinates of each point on edge
    polygon: np.array, coordinates of points on polygonized edge
    curvatures: np.array, curvatures of points on polygonized edge
    return: np.array, curvatures of all points on edge
    """
    
    polygon = polygon[50:-50]
    curvatures = curvatures[50:-50]
    edge_curvatures = np.zeros((edge_coords.shape[0], 1))
    for point_idx, point in enumerate(edge_coords):
        argmin = np.argmin(np.linalg.norm(polygon - point, axis=1))
        edge_curvatures[point_idx] = np.mean(curvatures[max(argmin - 1, 0): argmin + 2])
    return edge_curvatures

def edge_coords2curvatures(edge_coords):
    polygon = polygonize_edge(edge_coords)
    curvatures = polygon2curvatures(polygon)
    return curvatures_on_edge(edge_coords, polygon, curvatures)