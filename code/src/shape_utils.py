
import numpy as np
import rdp

from skimage.feature import canny
from skimage.color import rgb2lab, lab2rgb
from scipy.ndimage import center_of_mass
from joblib import Parallel, delayed
from typing import List

from src.utils import Fragment

class ShapeDescriptor:
    def __init__(self, color_edge, edge_coords, curvatures):
        self.color_edge = color_edge
        self.edge_coords = edge_coords
        self.curvatures = curvatures
        
def choose_next(region):
    """ 
    region: 3x3 neighbourhood, np.array with dtype bool
    """
    order = [(1, 2), (0, 1), (1, 0), (0, 2), (0, 0), (2, 1), (2, 0), (2, 2)]
    for point in order:
        if region[point]:
            return point

def correct_edge(edge_1d):
    corrected_edge = []
    i = 0
    while i < edge_1d.shape[0] - 2:
        corrected_edge.append(edge_1d[i])
        move1 = edge_1d[i + 1] - edge_1d[i]
        move2 = edge_1d[i + 2] - edge_1d[i + 1]
#         print(move1, move2)
        if move1[0] == - move2[0] and move1[1] * move2[1] == 0:
            i += 2
        elif move1[1] == - move2[1] and move1[0] * move2[0] == 0: 
            i += 2
        elif move1[0] == 0 and move2[1] == 0:
            i += 2
        elif move1[1] == 0 and move2[0] == 0:
            i += 2
        else: 
            i += 1
    corrected_edge.append(edge_1d[-2])
    corrected_edge.append(edge_1d[-1])
    return np.array(corrected_edge)

def linearize_edge(mask):    
    center = center_of_mass(mask)
    center = int(center[0]), int(center[1])
    edge = canny(mask[:,:, 0], sigma=1)
    for col in range(center[1], mask.shape[1]):
        if edge[center[0], col]:
            break
            
    start_point = (center[0], col)
    current_point = start_point
    not_seen = edge.copy()
    
    edge_1d = [current_point]
    not_seen[start_point[0], start_point[1]] = False
    while not_seen.any():
        region = not_seen[
            current_point[0] - 1: current_point[0] + 2, 
            current_point[1] - 1: current_point[1] + 2,
        ]
        if len(edge_1d) > 5 and current_point[0] - 1 <= start_point[0] <= current_point[0] + 1 and current_point[1] - 1 <= start_point[1] <= current_point[1] + 1:
            break
        move = choose_next(region)
        if move is None:
            break
        current_point = current_point[0] + move[0] - 1, current_point[1] + move[1] - 1
        not_seen[current_point[0], current_point[1]] = False
        edge_1d.append(current_point)
    edge_1d = correct_edge(np.array(edge_1d + edge_1d))
    return edge_1d

def get_colorized_edge(palette, frag):
    edge_1d = linearize_edge(frag.mask)
    colorized_edge = np.zeros(frag.fragment.shape)
    colorized_edge_1d = np.zeros((edge_1d.shape[0], 3))
    
    lab_frag = rgb2lab(frag.fragment)
    for i, point in enumerate(edge_1d):
        color = lab_frag[point[0], point[1]]
        argmin = np.argmin(np.linalg.norm(palette - color, axis=1))
        q_color = palette[argmin]
        colorized_edge[point[0], point[1]] = color
        colorized_edge_1d[i] = color  #!
    
    return colorized_edge_1d, lab2rgb(colorized_edge)

def polygonize_edge(edge_1d):
    polygon = rdp.rdp(edge_1d, epsilon=1)    
    stop = False
    while not stop:
        stop = True
        new_polygon = []
        for i in range(polygon.shape[0] - 1):
            new_polygon.append(polygon[i])
            if np.linalg.norm(polygon[i] - polygon[i + 1]) > 5:
                new_point = (polygon[i] + polygon[i + 1]) / 2
                new_polygon.append(np.round(new_point).astype(int))
                stop = False
        polygon = np.array(new_polygon)
        
    sigma = 0.25
    kernel = np.array([np.exp(-(i - 2) ** 2 / (2 * sigma ** 2)) for i in range(5)])
    kernel /= kernel.sum()

    smoothed = polygon.copy()
    smoothed[2:-2, 0] = np.convolve(smoothed[:, 0], kernel, mode='valid')
    smoothed[2:-2, 1] = np.convolve(smoothed[:, 1], kernel, mode='valid')
    return smoothed

