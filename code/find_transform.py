import numpy as np
import random

def estimate_transform_params(start1, start2, end1, end2):
    """
    returns the transform parameters cos(alpha), sin(alpha), a, b (a, b - shift, alpha - rotating angle)
    """
    A = np.array([
        [start1[0], -start1[1], 1, 0],
        [start1[1], start1[0], 0, 1],
        [start2[0], -start2[1], 1, 0],
        [start2[1], start2[0], 0, 1],
    ])
    b = np.array([end1[0], end1[1], end2[0], end2[1]])
    params = np.linalg.solve(A, b)
    return params

def aligned_coords2line(alignment_part, edge_coords, left: bool):
    indices = alignment_part
    indices = [i if left else j for (i, j) in indices]

    line = []
    for k in indices:
        line.append(edge_coords[k - 1])
    return np.array(line)

def old_find_best_transform_ransac(line1, line2):
    length = min(line1.shape[0], line2.shape[0])
    min_error = 10000
    best_transform = None
    
    for i in range(1000):
        l, r = random.choice(range(length)), random.choice(range(length))
        if l == r: 
            continue
        try: 
            dist = np.linalg.norm(line1[l] - line1[r])
            distances = np.abs(np.linalg.norm(line2 - line2[l], axis=1) - dist)
            best_point = np.argmin(distances)
            
            transform_params = estimate_transform_params(line1[l], line1[r], line2[l], line2[best_point])
        except Exception as e:
#             raise e
            continue
        error = estimate_mean_squared_transformation_error(line1, line2, transform_params)
        
        if error < min_error:
            min_error = error
            best_transform = transform_params
    return best_transform

def estimate_max_squared_transformation_error(line1, line2, trasform_params):
    """
    line1, line2 - np.arrays, shape (n, 2)
    transforms_params - tuple (cos, sin, a, b) of transform parameters line1 -> line2
    returns mean squared error
    """
    transformed_line1 = transform_line(line1, trasform_params)
    return np.linalg.norm(line2 - transformed_line1, axis=1).max()

def find_best_transform_ransac(line1, line2):
    length = min(line1.shape[0], line2.shape[0])
    min_error = 10000
    best_transform = None
    
    for i in range(1000):
        l = random.choice(range(length))
        if (length - l) <= 25: 
            continue
        r = random.choice(range(l + 25, length))
        if r - l < 25: 
            continue
        try: 
            dist = np.linalg.norm(line1[l] - line1[r])
            distances = np.abs(np.linalg.norm(line2 - line2[l], axis=1) - dist)
            distances[:l] = 10000
            best_point = np.argmin(distances)
            
            transform_params = estimate_transform_params(line1[l], line1[r], line2[l], line2[best_point])
        except Exception as e:
            continue
        error = estimate_max_squared_transformation_error(line1, line2, transform_params)
        
        if error < min_error:
            min_error = error
            best_transform = transform_params
    return best_transform


def transform_line(line, trasform_params):
    """
    line - np.array, shape (n, 2)
    trasform_params - tuple (cos, sin, a, b) of transform parameters line -> line
    returns transformed line
    """
    cos, sin, a, b = trasform_params
    transformed_line = np.zeros(line.shape)
    transformed_line[:, 0] = line[:, 0] * cos - line[:, 1] * sin + a
    transformed_line[:, 1] = line[:, 0] * sin + line[:, 1] * cos + b
    return transformed_line

def estimate_mean_squared_transformation_error(line1, line2, trasform_params):
    """
    line1, line2 - np.arrays, shape (n, 2)
    transforms_params - tuple (cos, sin, a, b) of transform parameters line1 -> line2
    returns mean squared error
    """
    transformed_line1 = transform_line(line1, trasform_params)
    return np.linalg.norm(line2 - transformed_line1)