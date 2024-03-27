import cv2
import numpy as np

from skimage.io import imread
from skimage.transform import rotate
from typing import Tuple
from dataclasses import dataclass

from src.color_descriptor import ColorDescriptor

@dataclass
class MaskPosition:
    top: int
    bottom: int
    left: int
    right: int

class Fragment:    
    def __init__(
        self,
        fragment,
        extended_frag,
        mask,
        extended_mask,
        color_descriptor = None,
        edge_coords = None,
        edge_colors = None
    ):
        self.fragment = fragment
        self.extended_frag = extended_frag
        self.mask = mask
        self.extended_mask = extended_mask
        self.color_descriptor = color_descriptor
        self.edge_coords = edge_coords
        self.edge_colors = edge_colors

def preprocess(
        fragment, mask, ext_step=10
) -> Tuple[np.array, np.array, 'MaskPosition']:
    top = np.where(mask)[0].min() - ext_step
    left = np.where(mask)[1].min() - ext_step
    bottom = np.where(mask)[0].max() + ext_step
    right = np.where(mask)[1].max() + ext_step
    size = max(bottom - top, right - left)

    crop = (mask * fragment)[
        (top + bottom) // 2 - (size // 2):(size // 2) + (top + bottom) // 2, 
        (left + right) // 2 - (size // 2):(size // 2) + (left + right) // 2
        ]
    mask_crop = np.invert(mask[
        (top + bottom) // 2 - (size // 2):(size // 2) + (top + bottom) // 2, 
        (left + right) // 2 - (size // 2):(size // 2) + (left + right) // 2
        ])
    return crop, mask_crop, MaskPosition(top, bottom, left, right)

def build_fragment_from_directory(fragment_dir: str):
    """
    fragment_dir: directory with frag, ext_frag, mask, ext_frag and some statistics
    """
    return Fragment(
        imread(fragment_dir + '/frag.png'),
        imread(fragment_dir + '/extended_frag.png'),
        imread(fragment_dir + '/mask.png').astype(bool),
        imread(fragment_dir + '/extended_mask.png').astype(bool),
        ColorDescriptor(
            np.load(fragment_dir + '/color_dsc_h.npy'),
            np.load(fragment_dir + '/color_dsc_b.npy'),
            np.load(fragment_dir + '/color_dsc_var.npy')
        ),
        np.load(fragment_dir + '/edge_coords.npy'),
        np.load(fragment_dir + '/colorized_edge.npy')
    )

def build_fragment(mask_index, model, indir='../voronoi/example', ext_step=15, pad=10):
    fragment = imread(indir + '/' + 'fresco.jpg')
    mask = imread(indir + '/' + f'new_mask_{mask_index}.png', as_gray=True) > 0.8
    mask = mask[:,:,None]
    cropped_frag, cropped_inv_mask, pos = preprocess(fragment, mask.astype(bool))
    cropped_frag = np.pad(cropped_frag, ((pad, pad), (pad, pad), (0, 0)))
    extended = cropped_frag
    
    mask = np.invert(cropped_inv_mask)
    extended_mask = cv2.dilate(mask * 1.0, np.ones((30, 30)))
    extended_mask = np.pad(extended_mask, ((pad, pad), (pad, pad)))
    extended_masked = extended_mask[:, :, None] * extended

    mask = np.pad(mask, ((pad, pad), (pad, pad), (0, 0)))
    mask = cv2.erode(mask * 1.0, np.ones((6, 6), np.uint8))[:,:,None]

    return Fragment(cropped_frag / 255, extended_masked / 255,mask, extended_mask[:,:,None])

def pad_fragment_to_size(frag, size):
    h, w = frag.mask.shape[0], frag.mask.shape[1]
    pad_h, pad_w = (3 * size - h) // 2, (3 * size - w) // 2
    new_frag = Fragment(
        np.pad(frag.fragment, ((pad_h, pad_h + 1 * (h % 2 != size % 2)), (pad_w, pad_w + 1 * (w % 2 != size % 2)), (0, 0))),
        np.pad(frag.extended_frag, ((pad_h, pad_h + 1 * (h % 2 != size % 2)), (pad_w, pad_w + 1 * (w % 2 != size % 2)), (0, 0))),
        np.pad(frag.mask, ((pad_h, pad_h + 1 * (h % 2 != size % 2)), (pad_w, pad_w + 1 * (w % 2 != size % 2)))),
        np.pad(frag.extended_mask, ((pad_h, pad_h + 1 * (h % 2 != size % 2)), (pad_w, pad_w + 1 * (w % 2 != size % 2)))),
        frag.color_descriptor,
        frag.edge_coords,
        frag.edge_colors
    )
    new_frag.edge_coords[:,0] += pad_h
    new_frag.edge_coords[:,1] += pad_w
    return new_frag

def pad_fragment(frag, size):
    h, w = frag.mask.shape[0], frag.mask.shape[1]
    pad_h, pad_w = (3 * size - h) // 2, (3 * size - w) // 2
    return Fragment(
        np.pad(frag.fragment, ((pad_h, pad_h + 1 * (h % 2 != size % 2)), (pad_w, pad_w + 1 * (w % 2 != size % 2)), (0, 0))),
        np.pad(frag.extended_frag, ((pad_h, pad_h + 1 * (h % 2 != size % 2)), (pad_w, pad_w + 1 * (w % 2 != size % 2)), (0, 0))),
        np.pad(frag.mask, ((pad_h, pad_h + 1 * (h % 2 != size % 2)), (pad_w, pad_w + 1 * (w % 2 != size % 2)), (0, 0))),
        np.pad(frag.extended_mask, ((pad_h, pad_h + 1 * (h % 2 != size % 2)), (pad_w, pad_w + 1 * (w % 2 != size % 2)), (0, 0))),
    )


def rotate_fragment(frag, angle, c=None):
    """
    fast rotate
    TODO: add edge_coords rotation
    """
    # TODO: rotate edge_coords
    h, w = frag.fragment.shape[:2]
    if c is None:
        c = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center=c, angle=angle, scale=1.0)
    fr = Fragment(
        cv2.warpAffine(frag.fragment, M=m, dsize=(w, h)),
        cv2.warpAffine(frag.extended_frag, M=m, dsize=(w, h)),
        cv2.warpAffine(frag.mask * 255.0, M=m, dsize=(w, h)) == 255,
        cv2.warpAffine(frag.extended_mask * 255.0, M=m, dsize=(w, h)) == 255
    )
    if fr.fragment.max() > 1:
        fr.fragment = fr.fragment / 255.0
    if fr.extended_frag.max() > 1:
        fr.extended_frag = fr.extended_frag / 255.0
    return fr

# def rotate_fragment(frag, angle):
#     # TODO: rotate edge_coords
#     return Fragment(
#         rotate(frag.fragment, angle),
#         rotate(frag.extended_frag, angle),
#         rotate(frag.mask, angle),
#         rotate(frag.extended_mask, angle),
#     )

def shift_fragment(frag, shift_x, shift_y):
    # TODO: shift edge_coords
    a, b, c, d = (
        np.roll(frag.fragment, (shift_y, shift_x), axis=(0, 1)), 
        np.roll(frag.extended_frag, (shift_y, shift_x), axis=(0, 1)), 
        np.roll(frag.mask, (shift_y, shift_x), axis=(0, 1)), 
        np.roll(frag.extended_mask, (shift_y, shift_x), axis=(0, 1))
    )
    f = Fragment(a, b, c, d)
    if shift_y > 0:
        f.fragment[:shift_y] = 0
        f.extended_frag[:shift_y] = 0
        f.mask[:shift_y] = 0
        f.extended_mask[:shift_y] = 0
    else:
        f.fragment[shift_y:] = 0
        f.extended_frag[shift_y:] = 0
        f.mask[shift_y:] = 0
        f.extended_mask[shift_y:] = 0
        
    if shift_x > 0:
        f.fragment[:, :shift_x] = 0
        f.extended_frag[:, :shift_x] = 0
        f.mask[:, :shift_x] = 0
        f.extended_mask[:, :shift_x] = 0
    else: 
        f.fragment[:, shift_x:] = 0
        f.extended_frag[:, shift_x:] = 0
        f.mask[:, shift_x:] = 0
        f.extended_mask[:, shift_x:] = 0
    return f

def transform_fragment(fragment, transform_params):
    """
    transform_params: (cos, sin, a, b)
    """
    cos, _, a, b = transform_params
    fragment = rotate_fragment(fragment, -np.rad2deg(np.arccos(cos)))
    fragment = shift_fragment(fragment, int(b), int(a))
    return fragment

def blend_fragments(frag1, frag2):
    return Fragment(
        (frag1.fragment * 1.0 + frag2.fragment * 1.0) / 2,
        (frag1.extended_frag * 1.0 + frag2.extended_frag * 1.0) / 2,
        (frag1.mask * 1.0 + frag2.mask * 1.0) / 2,
        (frag1.extended_mask * 1.0 + frag2.extended_mask * 1.0) / 2,
        
    )