import numpy as np

from skimage.io import imread
from skimage.color import rgb2lab
from color_descriptor import ColorDescriptor
from utils import build_fragment_from_directory
from quantization import get_colors_from_masked_image
from tqdm import tqdm
from typing import Dict


def find_nearest(array: np.ndarray, value: np.ndarray) -> np.ndarray:
    """
    Finds the nearest value in an array to a given value.

    Parameters:
        array (np.ndarray): The array in which to search for the nearest value.
        value (np.ndarray): The value to which the nearest value is sought.

    Returns:
        np.ndarray: The nearest value in the array to the given value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def compute_color_dsc(mask_index, palette, img=None, data_dir = '../voronoi/example'):
    """
    palette: np.array of shape (M, 3)
    return: ColorDescriptor if index is valid else None
    """
    if img is None:
        img_name = 'fresco.jpg'
        img = imread(data_dir + '/' + img_name) / 255
        img = rgb2lab(img)
    mask_name = f'new_mask_{mask_index}.png'
    try: 
        mask = imread(data_dir + '/' + mask_name)
    except FileNotFoundError:
        return
    colors = get_colors_from_masked_image(img, mask) # shape (3, n)
    b = np.mean(colors, axis=1)
    dists = np.array([
        np.linalg.norm(colors - palette[i][None, :], axis=1).reshape((colors.shape[0],))
        for i in range(palette.shape[0])
    ]) ** 2 # shape (M, N)
    closest = dists.argmin(axis=0)
    h = np.histogram(closest, bins=[i - 0.5 for i in range(palette.shape[0] + 2)])[0]
    variances = np.array([
        np.mean((dists[i] * (closest == i)))
        for i in range(palette.shape[0])
    ])
    return ColorDescriptor(h, b, np.sqrt(variances))

def collect_descriptors_from_dataset(dataset_dir: str, palette: np.array, n=128):
    return {i: build_fragment_from_directory(dataset_dir + f"/{i}").color_descriptor for i in tqdm(range(1,n))}

def collect_descriptors_from_directory(palette, data_dir='../voronoi/example', n=128): # remove outdated
    img_name = 'fresco.jpg'
    img = imread(data_dir + '/' + img_name) / 255
    img = rgb2lab(img)
    all_dict = {
        index: compute_color_dsc(index, palette, data_dir=data_dir, img=img) 
        for index in tqdm(range(n))
    }
    return {key:value for (key, value) in all_dict.items() if value is not None}

def compute_dsc_distance(dsc1, dsc2):
    h1 = dsc1.h / dsc1.h.sum()
    h2 = dsc2.h / dsc2.h.sum()
    hist_intersection = ((h1 + h2) / 2 * (1 - np.abs(h1 - h2))).sum()
    
    return hist_intersection

def compute_all_distances(dsc_dict: Dict[int, ColorDescriptor]) -> np.ndarray:
    """
    Compute all distances between the color descriptors in the given dictionary.

    Args:
        dsc_dict (Dict[int, ColorDescriptor]): A dictionary mapping integer keys to color descriptors.
        
    Returns:
        np.ndarray: An array containing the computed distances between the color descriptors. The array has shape (max_key + 1, max_key + 1), where max_key is the largest key in the dictionary.
    """
    max_key = np.max(list(dsc_dict.keys()))
    all_dists = np.zeros((max_key + 1, max_key + 1))
    for i in tqdm(range(all_dists.shape[0])):
        for j in range(all_dists.shape[1]):
            if i == j: 
                all_dists[i, j] = -1
                continue
            all_dists[i, j] = compute_dsc_distance(dsc_dict[i], dsc_dict[j]) if i in dsc_dict.keys() and j in dsc_dict.keys() else -1
    return all_dists

def generate_good_pairs_from_distances(all_distances):
    q = np.quantile(all_distances, 0.5)
    good_pairs = np.where(all_distances > q)
    good_pairs = list(zip(*good_pairs))
    return good_pairs

def generate_good_pairs_from_directory(palette, data_dir): # TODO: remove outdated
    dsc_dict = collect_descriptors_from_directory(
        palette,
        data_dir
    )
    all_distances = compute_all_distances(dsc_dict)
    good_pairs = generate_good_pairs_from_distances(all_distances)
    return good_pairs

def generate_good_pairs_from_dataset_directory(palette, dataset_dir, n=10):
    dsc_dict = collect_descriptors_from_dataset(dataset_dir, palette, n=n)
    all_distances = compute_all_distances(dsc_dict)
    good_pairs = generate_good_pairs_from_distances(all_distances)
    return good_pairs