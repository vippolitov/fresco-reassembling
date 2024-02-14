import numpy as np

from skimage.io import imread
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from tqdm import tqdm

def get_colors_from_masked_image(img, mask):
    """
    img: np.array
    
    return: np.array (N, 3), where N - number of pixels on the image
    """
    return img[mask[:,:,0] == mask.max()]

def get_all_fragments_colors(data_dir, n=125):
    colors = []
    img_name = 'fresco.jpg'
    img = imread(data_dir + '/' + img_name)
    img = rgb2lab(img)
    for i in tqdm(range(1,n+1)):
#         img_name = f'new_fragment_{i}.png'
        mask_name = f'new_mask_{i}.png'
        try: 
            mask = imread(data_dir + '/' + mask_name)
        except FileNotFoundError:
            continue
        img_colors = get_colors_from_masked_image(img, mask)
        colors.append(img_colors)
    return np.vstack(colors)

def generate_palette_from_colors(colors: np.ndarray):
    """
    Generates a palette of colors from an array of colors.
    
    Parameters:
        colors (np.ndarray): An array of colors.
        
    Returns:
        np.ndarray: palette, the cluster centers of the KMeans model.
    """
    cls = KMeans(n_clusters=7, verbose=1)
    cls.fit(colors)
    return cls.cluster_centers_
