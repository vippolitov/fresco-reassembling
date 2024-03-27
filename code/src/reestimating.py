import cv2
import numpy as np

from tqdm import tqdm
from scipy.ndimage import center_of_mass
from skimage import draw

from src.utils import Fragment, shift_fragment
from src.refine_transform import Translation


def generate_transform_neighborhood(tr: Translation):
    return [Translation(tr.x + dx, tr.y + dy, tr.angle + dalhpa, -1, -1) for dx in range(-6, 7, 3) for dy in range(-6, 7, 3) for dalhpa in range(-6, 7, 3)]

def transform_fragment(frag: Fragment, transform: Translation, transpose: bool = False):
    # TODO: sinchroniz all geometric operations
    if transpose:
        return rotate_fragment(shift_fragment(frag, -transform.x, -transform.y), -transform.angle)
    else:
        return shift_fragment(rotate_fragment(frag, transform.angle), transform.x, transform.y)
    
# def rotate_fragment(frag, angle, c=None):
#     """
#     fast rotate
#     """
#     # TODO: rotate edge_coords
#     h, w = frag.fragment.shape[:2]
#     if c is None:
#         c = (w // 2, h // 2)
#     m = cv2.getRotationMatrix2D(center=c, angle=angle, scale=1.0)
#     return Fragment(
#         cv2.warpAffine(frag.fragment, M=m, dsize=(w, h)) / 255,
#         cv2.warpAffine(frag.extended_frag, M=m, dsize=(w, h)) / 255,
#         cv2.warpAffine(frag.mask, M=m, dsize=(w, h)) == 255,
#         cv2.warpAffine(frag.extended_mask, M=m, dsize=(w, h)) == 255
#     )
    
def rotate_fragment(frag, angle, c=None):
    """
    fast rotate
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

def check_common_ext_intersection(
    anchor: Fragment, transformed1: Fragment, transformed2: Fragment
):
    """
    checks if there are three pairwise intersections: one fixed and two transformed
    """
    intersects1_2 = np.any(anchor.extended_mask & transformed1.extended_mask)
    intersects1_3 = np.any(anchor.extended_mask & transformed2.extended_mask)
    intersects2_3 = np.any(transformed1.extended_mask & transformed2.extended_mask)
    return intersects1_2 and intersects1_3 and intersects2_3
    
def check_common_intersection(
    anchor: Fragment, transformed1: Fragment, transformed2: Fragment
):
    """
    checks if all three fragments have pairwise intersections
    """
    intersects1_2 = np.any(anchor.mask & transformed1.mask)
    if intersects1_2:
        return True
    intersects1_3 = np.any(anchor.mask & transformed2.mask)
    if intersects1_3:
        return True
    intersects2_3 = np.any(transformed1.mask & transformed2.mask)
    if intersects2_3:
        return True
    return False

def compute_geom_morph_score(frag1, frag2, structure_elem=np.ones((25, 25))):
    """
    params:
    frag1, frag2: two fragments to check
    structure_elem: morphological structuring element for morphological closing
    suitable geometric score for reestimating: 1 - gap / extended intersection
    """
    two_mask = np.logical_or(frag1.mask, frag2.mask)
    wide_intersection = np.logical_and(frag1.extended_mask, frag2.extended_mask)
    merged = cv2.morphologyEx(two_mask * 1.0, cv2.MORPH_CLOSE, structure_elem, iterations=1)
    wide_union = np.logical_or(frag1.extended_mask, frag2.extended_mask)
    
    if len(wide_union.shape) == 2:
        wide_union = wide_union[:,:,None]
    if len(wide_intersection.shape) == 2:
        wide_intersection = wide_intersection[:,:,None]
    if len(two_mask.shape) == 2:
        two_mask = two_mask[:,:,None]
        
    # gap = np.logical_and(
    #     wide_intersection,
    #     np.logical_and(
    #         merged[:,:,None],
    #         np.logical_not(two_mask)
    #     )
    # )
    gap = wide_intersection & merged[:,:,None] & np.logical_not(two_mask)
    gap_sum, intersection_sum = gap.sum(), wide_intersection.sum()
    return 1 - gap_sum / intersection_sum

def check_if_two_fragments_are_too_far(frag1, frag2):
    """
    return True if two frags are too far from each other
    """
    center1 = center_of_mass(frag1.mask)
    center2 = center_of_mass(frag2.mask)
    start = (int(center1[0]), int(center1[1]))
    end = (int(center2[0]), int(center2[1]))
    line = draw.line(*start, *end)
    points = zip(line[0], line[1])
    belong_to_any_frag = [frag1.mask[p[0], p[1]] or frag2.mask[p[0], p[1]] for p in points]
    bad_points_number = len(belong_to_any_frag) - sum(belong_to_any_frag)
    return bad_points_number > 30

def check_if_fragments_intersect_too_much(frag1, frag2):
    """
    return True if two frags have too big intersection
    """
    i = (frag1.mask & frag2.mask).sum()
    return i / frag1.mask.sum() > 0.1 or i / frag2.mask.sum() > 0.1

def estimate_triplet(
    anchor: Fragment,
    second: Fragment,
    third: Fragment, 
    transform1: Translation, 
    transform2: Translation,
    transpose1: bool = False,
    transpose2: bool = False,
    verbose: int = 1
):
    """
    anchor: acnhor fragment 
    second: non-shifted neihgbor
    third: non-shifted neighbor, which is used to estimate pair (anchor, second), third neighbor
    transform1: Translation, transform to be used to align second 
    transform2: Translation, transform to be used to align third
    
    return: bool, is the triplet good
    """
#     print("Checking if too far or intersects too much")
    tr1 = transform_fragment(second, transform1, transpose1)
    tr2 = transform_fragment(third, transform2, transpose2)
    if check_if_two_fragments_are_too_far(tr1, tr2):
        return False
    if check_if_fragments_intersect_too_much(tr1, tr2):
        return False
    
    transf1_neighbourhood = generate_transform_neighborhood(transform1)
    transf2_neighbourhood = generate_transform_neighborhood(transform2)
    
    # iterate over neighborhood
    print("creating neighbor fragments...")
    transformed1_lst = [transform_fragment(second, tr1, transpose=transpose1) for tr1 in (tqdm(transf1_neighbourhood) if verbose else transf1_neighbourhood)]
    transformed2_lst = [transform_fragment(third, tr2, transpose=transpose2) for tr2 in transf2_neighbourhood]
    print("iterating over neighbor fragments...")
    for transformed1 in tqdm(transformed1_lst) if verbose else transformed1_lst:
        for transformed2 in transformed2_lst:
            common_ext_intersection_exists = check_common_ext_intersection(anchor, transformed1, transformed2)
            if not common_ext_intersection_exists:
                continue
            common_intersection_exists = check_common_intersection(anchor, transformed1, transformed2)
            if common_intersection_exists:
                continue
            geom_score1_2 = compute_geom_morph_score(anchor, transformed1)
            geom_score1_3 = compute_geom_morph_score(anchor, transformed2)
            geom_score2_3 = compute_geom_morph_score(transformed1, transformed2)
            # print(geom_score1_2, geom_score1_3, geom_score2_3)
            if geom_score1_2 > 0.6 and geom_score2_3 > 0.6 and geom_score1_3 > 0.6:
                return (transformed1, transformed2)
    return False
    
def check_triplet_to_pair_exists(anchor_i, second_i, refined_alignment, fragments, frag_numbers, map_id_to_idx, verbose=True):
    anchor = fragments[anchor_i]
    anchor.mask = anchor.mask.astype(bool)
    anchor.extended_mask = anchor.extended_mask.astype(bool)
    second = fragments[second_i]
    pair = (anchor_i, second_i) if anchor_i < second_i else (second_i, anchor_i)
    for transform2 in refined_alignment[pair][:4] if pair in refined_alignment else []:
#         print(f"Checking transform: {transform2}")
        for third_id in tqdm(frag_numbers) if verbose else frag_numbers:
            third_i = map_id_to_idx[third_id]
            print(f"CHecking triplet {anchor_i, second_i, third_i}")
            if third_i == anchor_i or third_i == second_i:
                continue
            third = fragments[third_i]
            pair1_3 = (anchor_i, third_i) if anchor_i < third_i else (third_i, anchor_i)
            for transform3 in refined_alignment[pair1_3][:4] if pair1_3 in refined_alignment else []:
                res = estimate_triplet(anchor, second, third, transform2, transform3, transpose1=anchor_i<second_i, transpose2=anchor_i<third_i, verbose=verbose)
                if res:
                    return (anchor, second, third, res[0], res[1], anchor_i<second_i, anchor_i<third_i)
    return False

def filter_pairs_without_triplet(refined_alignment, frags, frag_numbers, map_id_to_idx):
    good_pairs = {}
    for l in range(len(frag_numbers[:1])):
        for r in range(len(frag_numbers[:3])):
            if l == r:
                continue
            print(f"Checking pair {l}, {r}")
            res = check_triplet_to_pair_exists(l, r, refined_alignment, frags, frag_numbers, map_id_to_idx, verbose=True)
            if res:
                good_pairs[(l, r)] = res
#                 print(f"Found good triplet: {l}, {r}, {res[2]}")
    return good_pairs