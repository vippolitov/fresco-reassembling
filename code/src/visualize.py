import matplotlib.pyplot as plt
from src.utils import Fragment

def visualize_triplet(frag1, frag2, frag3):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    merged_fragment = frag1.fragment + frag2.fragment + frag3.fragment
    mask = frag1.mask + frag2.mask + frag3.mask
    axes[0].imshow(merged_fragment)
    axes[1].imshow(mask)

def visualize_fragment(frag):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7,7))
    axes[0][0].imshow(frag.fragment)
    axes[0][1].imshow(frag.mask * 255)
    axes[1][0].imshow(frag.extended_frag)
    axes[1][1].imshow(frag.extended_mask * 255)

def blend_fragments(frag1, frag2):
    if frag1.fragment.max() > 1:
        frag1.fragment = frag1.fragment / 255
    if frag2.fragment.max() > 1:
        frag2.fragment = frag2.fragment / 255
    if frag1.extended_frag.max() > 1:
        frag1.extended_frag = frag1.extended_frag / 255
    if frag2.extended_frag.max() > 1:
        frag2.extended_frag = frag2.extended_frag / 255
    if frag1.mask.max() > 1:
        frag1.mask = frag1.mask / 255
    if frag2.mask.max() > 1:
        frag2.mask = frag2.mask / 255
    if frag1.extended_mask.max() > 1:
        frag1.extended_mask = frag1.extended_mask / 255
    if frag2.extended_mask.max() > 1:
        frag2.extended_mask = frag2.extended_mask / 255
    return Fragment(
        (frag1.fragment * 1.0 + frag2.fragment * 1.0) / 2,
        (frag1.extended_frag * 1.0 + frag2.extended_frag * 1.0) / 2,
        (frag1.mask * 1.0 + frag2.mask * 1.0) / 2,
        (frag1.extended_mask * 1.0 + frag2.extended_mask * 1.0) / 2,
        
    )