import cv2 
import numpy as np

from pathlib import Path
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb

from src.shape_utils import linearize_edge
from src.extend import extend_fragment, load_model
from src.color_dsc_operations import ColorDescriptor
from src.quantization import get_colors_from_masked_image
from src.utils import MaskPosition

class FragmentPreprocessor:
    def __init__(self, ext_step=30, model_name="lama"):
        self.ext_step = ext_step
        if model_name:
            self.model = load_model()
        else:
            self.model = None


    def find_crop_params(self, mask):    
        top = np.where(mask)[0].min() - self.ext_step
        left = np.where(mask)[1].min() - self.ext_step
        bottom = np.where(mask)[0].max() + self.ext_step
        right = np.where(mask)[1].max() + self.ext_step
        return MaskPosition(top, bottom, left, right)

    def leave_biggest_component (self, mask):
        """
        mask: np.array, binary mask
        return: np.array, binary mask of the biggest connected component of the mask
        """
        mask = mask.astype('uint8')
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
        sizes = stats[:, -1]

        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        img2 = np.zeros(output.shape)
        img2[output == max_label] = 255
        return img2

    def read_fragment(self, data_dir: str, mask_index: int):
        """
        Opens fragments, crops it to bounding box, extracts its binary mask and extends it corresponding to inpainting model
        """
        fresco = imread(f'{data_dir}/fresco.jpg')
        mask_name = data_dir + '/' + f'new_mask_{mask_index}.png'    
        mask = imread(mask_name) > 0.9
        mask = cv2.erode(mask[:, :, 0] * 1.0, np.ones((4, 4), np.uint8))[:,:,None].astype(bool)
        mask = self.leave_biggest_component(mask)[:, :, None].astype(bool)
        frag = fresco * mask
        extended_mask = cv2.dilate(mask * 1.0, np.ones((20, 20)))[:, :, None].astype(bool)
        
        crop_params = self.find_crop_params(mask)
        frag = frag[crop_params.top:crop_params.bottom, crop_params.left:crop_params.right]
        mask = mask[crop_params.top:crop_params.bottom, crop_params.left:crop_params.right]
        extended_mask = extended_mask[crop_params.top:crop_params.bottom, crop_params.left:crop_params.right]
        extended_frag = extend_fragment(frag, mask, extended_mask, self.model)
    #     extended_frag = frag
        return frag, extended_frag, mask, extended_mask

    def compute_color_dsc(self, img, mask, palette):
        """
        img: np.array image to be masked and to provide color histogram
        mask: np.array
        palette: np.array of shape (M, 3) - color centers from the LAB space
        return: ColorDescriptor if index is valid else None
        """
        lab_img = rgb2lab(img)
        colors = get_colors_from_masked_image(lab_img, mask) # shape (3, n)
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


    def get_colorized_edge(self, img, mask, palette):
        """
        img: np.array, RGB image
        mask: np.array, binary mask
        palette: np.array of shape (M, 3) - color centers from the LAB space
        """
        edge_1d = linearize_edge(mask)
        colorized_edge = np.zeros(img.shape)
        colorized_edge_1d = np.zeros((edge_1d.shape[0], 3))
        
        lab_frag = rgb2lab(img)
        for i, point in enumerate(edge_1d):
            color = lab_frag[point[0], point[1]]
            argmin = np.argmin(np.linalg.norm(palette - color, axis=1))
            q_color = palette[argmin]
            colorized_edge[point[0], point[1]] = color
            colorized_edge_1d[i] = color  #!
        
        return edge_1d, colorized_edge_1d, lab2rgb(colorized_edge)

    def convert_fragment_to_data_structure(
            self,
        data_dir: str, 
        mask_index: int, 
        output_dir: str,
        palette, 
    ):
        frag, extended_frag, mask, extended_mask = self.read_fragment(data_dir, mask_index)
        color_dsc = self.compute_color_dsc(frag, mask, palette)
        edge_coords, colorized_edge1, _ = self.get_colorized_edge(extended_frag, mask, palette)
        
        p = Path(output_dir).joinpath(str(mask_index))
        p.mkdir(parents=True, exist_ok=True)
        
        imsave(p / 'frag.png', frag)
        imsave(p / 'extended_frag.png', extended_frag)
        imsave(p / 'mask.png', mask)
        imsave(p / 'extended_mask.png', extended_mask)
        np.save(p / 'color_dsc_h.npy', color_dsc.h)
        np.save(p / 'color_dsc_b.npy', color_dsc.b)
        np.save(p / 'color_dsc_var.npy', color_dsc.var)
        np.save(p / 'edge_coords.npy', edge_coords)
        np.save(p / 'colorized_edge.npy', colorized_edge1)
    