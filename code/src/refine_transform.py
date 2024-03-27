import numpy as np
import cv2
import torch

from dataclasses import dataclass
from tqdm import tqdm 

from src.utils import pad_fragment, pad_fragment_to_size, rotate_fragment, shift_fragment

@dataclass
class Translation:
    x: int
    y: int
    angle: float
    confidence: float
    geom_score: float
        
    def __lt__(self, another):
        return self.confidence > another.confidence
    
def check_possibility_of_translation(anchor_frag, transformed_frag):
    intersects = np.logical_and(anchor_frag.mask, transformed_frag.mask).any()
    ext1_intersects2 = np.logical_and(anchor_frag.extended_mask, transformed_frag.mask).any()
    ext2_intersects1 = np.logical_and(transformed_frag.extended_mask, anchor_frag.mask).any()

    intersection = np.logical_and(anchor_frag.extended_mask, transformed_frag.extended_mask).sum()
    too_small_ext_intersection = max(intersection / anchor_frag.extended_mask.sum(), intersection / transformed_frag.extended_mask.sum()) < 0.025

    return not intersects and (ext1_intersects2 or ext2_intersects1) and not too_small_ext_intersection

def compute_geom_morph_score(frag1, frag2, structure_elem=np.ones((25, 25))):
    two_mask = np.logical_or(frag1.mask, frag2.mask)
    wide_intersection = np.logical_and(frag1.extended_mask, frag2.extended_mask)
    merged = cv2.morphologyEx(two_mask * 1.0, cv2.MORPH_CLOSE, structure_elem, iterations=1)
    wide_union = np.logical_or(frag1.extended_mask, frag2.extended_mask)
    gap = np.logical_and(
        wide_intersection,
        np.logical_and(
            merged[:,:,None],
            np.logical_not(two_mask)
        )
    )
#     gap = np.logical_and(
#         gap,
#         wide_union
#     )
    gap_sum, intersection_sum = gap.sum(), wide_intersection.sum()
    if intersection_sum / max(frag1.mask.sum(), frag2.mask.sum()) < 0.05 or gap_sum / intersection_sum < 0.08:
        return 0
    return 1 - gap_sum / intersection_sum


def compute_fast_geom_morph_score(subcurve1, subcurve2, transform_params, max_distance=30):
    """
    subcurve1: (n, 2) array, common subcurve from frag1 
    subcurve2: (n, 2) array, common subcurve from frag2
    [:, 0] - x coordinates
    [:, 1] - y coordinates
    transform_params: tuple of (theta, shift_x, shift_y)
    """
    theta, shift_x, shift_y = transform_params
    transformed_subcurve = np.zeros(subcurve1.shape)
    theta_rad = np.deg2rad(theta)
    transformed_subcurve[:, 0] = subcurve1[:, 0] * np.cos(theta_rad) - subcurve1[:, 1] * np.sin(theta_rad) + shift_x
    transformed_subcurve[:, 1] = subcurve1[:, 0] * np.sin(theta_rad) + subcurve1[:, 1] * np.cos(theta_rad) + shift_y
    score = (max_distance - np.max(np.linalg.norm(transformed_subcurve - subcurve2, axis=1))) / max_distance
    
    return score ** (1 / 3) if score > 0 else 0

def compute_cross_correlation(frag1, frag2):
    img1, img2 = frag1.extended_frag, frag2.extended_frag
#     img1 = (img1 - img1.min()) / (img1.max() - img1.min()) - 0.5
#     img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    if img1.max() > 1:
        img1 = img1 / 255
    if img2.max() > 1:
        img2 = img2 / 255
    img1 = img1 - 0.5
    img2 = img2 - 0.5
#     img1, img2 = img1 / img1.max() - 0.5, img2 / img2.max() - 0.5 
    where = np.logical_and(frag1.extended_mask, frag2.extended_mask)
    if len(where.shape) == 2:
        where = where[:, :, None]
    cov = (img1 * img2 * where).sum()
    corr = cov / np.sqrt((img1 ** 2 * where).sum() * (img2 ** 2 * where).sum())
    return corr

def nms(res):
    """
    non maximum suppression
    """
    current_max_i = 0
    while current_max_i < len(res):
        
        current_max = res[current_max_i]
        j = current_max_i + 1
        while j < len(res):
            if 0 < np.sqrt((res[j].x - current_max.x) ** 2 + (res[j].y - current_max.y) ** 2 + (res[j].angle - current_max.angle) ** 2) < 20:
                res.remove(res[j])
            else:
                j += 1
        current_max_i += 1
    return res

def match_fragments(frag1, frag2, initial_params, subcurve1, subcurve2):
    """
    initial_params: (angle, x, y)
    frag1, frag2: Fragments
    subcurve1, subcurve2: common subcurves from frag1 and frag2
    """
    theta, x_initial, y_initial = initial_params
    print(theta, x_initial, y_initial)
    size = max(max(frag1.mask.shape[0], frag1.mask.shape[1]), max(frag2.mask.shape[0], frag2.mask.shape[1]))
    padded_frag1 = pad_fragment(frag1, size)
    padded_frag2 = pad_fragment(frag2, size)
    
    global_res = []
    shifts = [(x, y) for x in range(x_initial - 10, x_initial + 10, 3) for y in range(y_initial - 10, y_initial + 10, 3)]
    for phi in np.arange(theta - 8, theta + 8, 2):
        rot_frag2 = rotate_fragment(padded_frag2, phi)
        for (x, y) in tqdm(shifts):
            transformed2 = shift_fragment(rot_frag2, x, y)
            if check_possibility_of_translation(padded_frag1, transformed2):

                geom_score = compute_fast_geom_morph_score(subcurve1, subcurve2, (phi, x, y))
                prob = geom_score
                # prob = estimate_probability_of_neighbourhood(frag1, transformed2)
                if prob > 0.7:
                    # compute content score
#                 prob = (1 + beta) / (1 / content_score + beta * 1 / geom_score)
                    trans = Translation(x, y, phi, prob)
#                     heapq.heappush(global_res, trans)
                    global_res.append(trans)
#         global_res.extend(res)
    filtered_res = nms(sorted(global_res, reverse=True, key=lambda val: val.confidence).copy())
    return filtered_res

def compute_new_content_score(frag1, frag2, features1, features2, resized_mask1, resized_mask2, shift, eps=0.001):
    """
    frag1 - fragment
    frag2 - transformed fragment
    features1 - frag1 features from pre-trained model, shape (n_features, height, width)
    features2 - frag2 features from pre-trained model, shape (n_features, height, width)
    shift - translation (shift_x, shift_y) frag1 -> frag2
    """
#     img_cross_corr = compute_content_score(frag1, frag2)
    pad = min(frag1.fragment.shape[0], frag2.fragment.shape[0])
    padded_features1 = np.pad(features1, ((0, 0), (pad, pad), (pad, pad)))
    padded_features2 = np.pad(features2, ((0, 0), (pad, pad), (pad, pad)))
    
    shifted_features1 = padded_features1[:,pad - shift[1]: -pad - shift[1], pad - shift[0]: -pad - shift[0]]

    if len(resized_mask2.shape) == 2:
        resized_mask2 = resized_mask2[:, :, None]
    if len(resized_mask1.shape) == 2:
        resized_mask1 = resized_mask1[:, :, None]

    padded_mask1 = np.pad(resized_mask1, ((pad, pad), (pad, pad), (0, 0)))
    shifted_mask1 = padded_mask1[pad - shift[1]: -pad - shift[1], pad - shift[0]: -pad - shift[0]]
    where = np.logical_and(resized_mask2, shifted_mask1).transpose(2, 0, 1)
    
    features2_masked = features2 * where
    features1_masked = shifted_features1 * where
    features_cov = (features1_masked * features2_masked).sum()
    features_cross_corr = features_cov / np.sqrt((features1_masked ** 2).sum() * (features2_masked ** 2).sum() + eps)
    return features_cross_corr

def geom_score_to_distance(geom_score, max_distance=30):
    d = max_distance - (geom_score ** 3) * max_distance
    return d if d != max_distance else max_distance * 2

def filter_too_far_translations(translations):
    if len(translations) == 0:
        return translations
    min_distance = min([geom_score_to_distance(t.geom_score) for t in translations])
    filtered = [t for t in translations if geom_score_to_distance(t.geom_score) < 2 * min_distance]
    print("Too far translations filtered out: {}".format(len(translations) - len(filtered)))
    return filtered
    

def match_two_aligned_fragments(frag1, frag2, list_of_initial_params, subcurves1, subcurves2, feature_extractor=None, beta=2, pad_size=200, verbose=1, eps=0.001):
    """
    list_of_initial_params: list of (angle, x, y)
    frag1, frag2: Fragments
    subcurves1, subcurves2: common subcurves from frag1 and frag2, each corresponds to initial_params
    finds optimal translation between frag1 -> frag2
    return: list of optimal translations (after non-max suppression)
    """
    padded_frag1 = pad_fragment_to_size(frag1, pad_size)
    padded_frag2 = pad_fragment_to_size(frag2, pad_size)
    
    # tensor2 = torch.tensor(padded_frag2.extended_frag, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    # if tensor2.max() > 1:
    #     tensor2 = tensor2 / 255
    # features2 = feature_extractor(tensor2)
    # features2 = features2.squeeze(0).detach().numpy()
    
    global_res = []
    
    for params_index, initial_params in enumerate(list_of_initial_params):
        best_translation = None
        theta, x_initial, y_initial = initial_params
        print(f"theta = {theta}, x_initial = {x_initial}, y_initial = {y_initial}")
        subcurve1 = subcurves1[params_index]
        subcurve2 = subcurves2[params_index]
        
        shifts = [(x, y) for x in range(x_initial - 30, x_initial + 31, 6) for y in range(y_initial - 30, y_initial + 31, 6)]
        for phi in np.arange(theta - 15, theta + 16, 5):
            rot_frag1 = rotate_fragment(padded_frag1, phi)
            # tensor1 = torch.tensor(rot_frag1.extended_frag, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            # if tensor1.max() > 1:
            #     tensor1 = tensor1 / 255
            # features1 = feature_extractor(tensor1)
            # features1 = features1.squeeze(0).detach().numpy()

            good_shifts = []
            for (x, y) in tqdm(shifts) if verbose == 1 else shifts:
                transformed1 = shift_fragment(rot_frag1, x, y)
                if check_possibility_of_translation(padded_frag2, transformed1):
                    geom_score = compute_fast_geom_morph_score(subcurve1, subcurve2, (phi, y, x), max_distance=50)
                    prob = geom_score
                    if prob > 0.5:
                        good_shifts.extend([(x_new, y_new) for x_new in range(x - 2, x + 3, 2) for y_new in range(y - 2, y + 3, 2)])
            for (x, y) in tqdm(good_shifts) if verbose == 1 else good_shifts:
                transformed1 = shift_fragment(rot_frag1, x, y)
                if check_possibility_of_translation(padded_frag2, transformed1):
                    geom_score = compute_fast_geom_morph_score(subcurve1, subcurve2, (phi, y, x), max_distance=50)
                    prob = geom_score
                    if prob > 0.5:
#                         geom_score = (geom_score1 + geom_score2) / 2
#                         content_score = compute_new_content_score(
#                             padded_frag2, transformed1,
#                             features2, features1,
#                             padded_frag2.extended_mask, transformed1.extended_mask,
#                             (x, y)                        
#                         )
                        content_score = compute_cross_correlation(padded_frag2, transformed1)
                        if content_score < eps:
                            content_score = eps
                        prob = (1 + beta) / (1 / (content_score) + beta * 1 / (geom_score + eps))
                        trans = Translation(x, y, phi, prob, geom_score)
                        if prob > 0.5:
                            global_res.append(trans)
                        if best_translation is None or best_translation.confidence < trans.confidence:
                            pass
#                             best_translation = trans
        if best_translation is None:
            continue
        global_res.append(sorted(best_translation, reverse=True, key=lambda val: val.confidence)[:10])
            
    filtered_res = filter_too_far_translations(global_res)
    filtered_res = nms(sorted(filtered_res, reverse=True, key=lambda val: val.confidence).copy())
    return filtered_res