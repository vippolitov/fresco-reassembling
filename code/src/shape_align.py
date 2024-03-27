import numpy as np

from numpy.linalg import norm
from tqdm import tqdm
from typing import List, Tuple, Dict
from joblib import Parallel, delayed

from src.find_transform import aligned_coords2line, find_best_transform_ransac, transform_line
from src.shape_utils import get_colorized_edge, linearize_edge, ShapeDescriptor
from src.curvature import edge_coords2curvatures
from src.utils import Fragment

class Alignment:
    def __init__(self, indices, conf):
        self.indices = indices
        self.conf = conf
        
        
        
    

def zeros(shape):
    retval = []
    for x in range(shape[0]):
        retval.append([])
        for y in range(shape[1]):
            retval[-1].append(0)
    return retval

match_award      = 20
mismatch_penalty = -20
gap_penalty      = -10 # both for opening and extanding

def match_score(alpha, beta):
    if alpha == beta:
        return match_award
    elif alpha == '-' or beta == '-':
        return gap_penalty
    else:
        return mismatch_penalty
    
def match_color_score(alpha, beta):
    if (alpha == beta).all():
        return match_award
    else:
        return mismatch_penalty

def old_compute_diag_score(curvs1, curvs2, seq1, seq2): # outdated -- saved just in case
    m, n = len(seq1), len(seq2)  # length of two sequences
    curv_diff = np.abs(curvs1[:, None] + curvs2[None, :])[:,:,0]
    color_diff = norm(seq1[:, None, :] - seq2[None, :, :], axis=2)
    color_sim = color_diff < 30
    diag_score = np.zeros((m , n))
    
    diag_score[np.logical_not(color_sim)] = - 30
    
    diag_score[np.logical_and(curv_diff < 0.02, color_sim)] = 45
    diag_score[np.logical_and(0.02 < curv_diff, curv_diff < 0.03, color_sim)] = 30
    diag_score[np.logical_and(0.03 < curv_diff, curv_diff < 0.04, color_sim)] = - 20
    diag_score[np.logical_and(0.04 < curv_diff, curv_diff < 0.05, color_sim)] = - 40
    diag_score[np.logical_and(0.05 < curv_diff, color_sim)] = - 80
    
    diag_score[np.logical_and(0.03 < curv_diff, curv_diff < 0.04, np.logical_not(color_sim))] = - 30
    diag_score[np.logical_and(0.04 < curv_diff, curv_diff < 0.05, np.logical_not(color_sim))] = - 60
    diag_score[np.logical_and(0.05 < curv_diff, np.logical_not(color_sim))] = - 120

    diag_score[np.logical_and((curvs1 > 0.05)[:, None, 0], (curvs2 < -0.05)[None, :, 0])] = 40
    diag_score[np.logical_and((curvs1 < -0.05)[:, None, 0], (curvs2 > 0.05)[None, :, 0])] = 40
    diag_score[np.logical_and((curvs1 > 0.07)[:, None, 0], (curvs2 < -0.07)[None, :, 0])] = 60
    diag_score[np.logical_and((curvs1 < -0.07)[:, None, 0], (curvs2 > 0.07)[None, :, 0])] = 60
    
    
    return diag_score

def compute_diag_score(curvs1, curvs2, seq1, seq2):
    """
    TODO: write docs about scores
    """
    print("new diag score is estimated")
#     m, n = len(seq1), len(seq2)  # length of two sequences
#     curv_diff = np.abs(curvs1[:, None] + curvs2[None, :])[:,:,0]
#     curvs1_small = np.abs(curvs1) < 0.03
#     curvs2_small = np.abs(curvs2) < 0.03
#     one_of_curvs_small = np.logical_or(curvs1_small[:, None, 0], curvs2_small[None, :, 0])
#     both_curvs_big = np.logical_not(one_of_curvs_small)
    
#     color_diff = norm(seq1[:, None, :] - seq2[None, :, :], axis=2)
#     color_sim = color_diff < 30
#     diag_score = np.zeros((m , n))
    
#     base = 30
#     diag_score[np.logical_not(color_sim)] = -base
    
    
#     diag_score[(curv_diff < 0.02) & color_sim & one_of_curvs_small] = base 
#     diag_score[(curv_diff < 0.02) & color_sim & both_curvs_big] = base * 3
# #     diag_score[np.logical_and(curv_diff < 0.02, color_sim)] = base
#     diag_score[(0.02 < curv_diff) & (curv_diff < 0.03) & color_sim & both_curvs_big] = base * 0.5
#     diag_score[(0.03 < curv_diff) & (curv_diff < 0.04) & color_sim] = -base
#     diag_score[(0.03 < curv_diff) & (curv_diff < 0.04) & color_sim & one_of_curvs_small] = -base * 2
#     diag_score[(0.04 < curv_diff) & (curv_diff < 0.05) & color_sim] = -base * 2
#     diag_score[(0.04 < curv_diff) & (curv_diff < 0.05) & color_sim & one_of_curvs_small] = -base * 3
#     diag_score[(0.05 < curv_diff) & (color_sim)] = -base * 4
#     diag_score[(0.05 < curv_diff) & (color_sim & one_of_curvs_small)] = -base * 5
    
#     diag_score[(0.03 < curv_diff) & (curv_diff < 0.04) & np.logical_not(color_sim)] = - base
#     diag_score[(0.04 < curv_diff) & (curv_diff < 0.05) & np.logical_not(color_sim)] = - base * 2
#     diag_score[(0.05 < curv_diff) & np.logical_not(color_sim)] = - base * 4

#     diag_score[(0.05 > curvs1)[:, None, 0] & (curvs1 > 0.03)[:, None, 0] & (-0.05 < curvs2)[None, :, 0] & (curvs2 < -0.03)[None, :, 0]] = base
#     diag_score[(-0.05 < curvs1)[:, None, 0] & (curvs1 < -0.03)[:, None, 0] & (0.05 > curvs2)[None, :, 0] & (curvs2 > 0.03)[None, :, 0]] = base
#     diag_score[(curvs1 > 0.05)[:, None, 0] &(curvs2 < -0.05)[None, :, 0]] = base * 3
#     diag_score[(curvs1 < -0.05)[:, None, 0] & (curvs2 > 0.05)[None, :, 0]] = base * 3
#     diag_score[(curvs1 > 0.07)[:, None, 0] & (curvs2 < -0.07)[None, :, 0]] = base * 4
#     diag_score[(curvs1 < -0.07)[:, None, 0] & (curvs2 > 0.07)[None, :, 0]] = base * 4
#     diag_score[(curv_diff < 0.02) & color_sim & both_curvs_big] = base * 3
    
    m, n = len(seq1), len(seq2)  # length of two sequences
    curv_diff = np.abs(curvs1[:, None] + curvs2[None, :])[:,:,0]
    curvs1_small = np.abs(curvs1) < 0.025
    curvs2_small = np.abs(curvs2) < 0.025
    one_of_curvs_small = np.logical_or(curvs1_small[:, None, 0], curvs2_small[None, :, 0])
    both_curvs_big = np.logical_not(one_of_curvs_small)
    
    color_diff = norm(seq1[:, None, :] - seq2[None, :, :], axis=2)
    color_sim = color_diff < 30
    diag_score = np.zeros((m , n))
    
    base = 30
    diag_score[np.logical_not(color_sim)] = -base
    
    
    diag_score[(curv_diff < 0.02) & color_sim & one_of_curvs_small] = base 
    diag_score[(curv_diff < 0.02) & color_sim & both_curvs_big] = base * 3
#     diag_score[np.logical_and(curv_diff < 0.02, color_sim)] = base
    diag_score[(0.02 < curv_diff) & (curv_diff < 0.03) & color_sim & both_curvs_big] = base * 0.5
    diag_score[(0.03 < curv_diff) & (curv_diff < 0.04) & color_sim] = -base
    diag_score[(0.03 < curv_diff) & (curv_diff < 0.04) & color_sim & one_of_curvs_small] = -base * 2
    diag_score[(0.04 < curv_diff) & (curv_diff < 0.05) & color_sim] = -base * 2
    diag_score[(0.04 < curv_diff) & (curv_diff < 0.05) & color_sim & one_of_curvs_small] = -base * 3
    diag_score[(0.05 < curv_diff) & (color_sim)] = -base * 4
    diag_score[(0.05 < curv_diff) & (color_sim & one_of_curvs_small)] = -base * 5
    
    diag_score[(0.03 < curv_diff) & (curv_diff < 0.04) & np.logical_not(color_sim)] = - base
    diag_score[(0.04 < curv_diff) & (curv_diff < 0.05) & np.logical_not(color_sim)] = - base * 2
    diag_score[(0.05 < curv_diff) & np.logical_not(color_sim)] = - base * 4

    diag_score[(0.05 > curvs1)[:, None, 0] & (curvs1 > 0.03)[:, None, 0] & (-0.05 < curvs2)[None, :, 0] & (curvs2 < -0.03)[None, :, 0]] = base
    diag_score[(-0.05 < curvs1)[:, None, 0] & (curvs1 < -0.03)[:, None, 0] & (0.05 > curvs2)[None, :, 0] & (curvs2 > 0.03)[None, :, 0]] = base
    diag_score[(curvs1 > 0.05)[:, None, 0] &(curvs2 < -0.05)[None, :, 0]] = base * 3
    diag_score[(curvs1 < -0.05)[:, None, 0] & (curvs2 > 0.05)[None, :, 0]] = base * 3
    diag_score[(curvs1 > 0.07)[:, None, 0] & (curvs2 < -0.07)[None, :, 0]] = base * 4
    diag_score[(curvs1 < -0.07)[:, None, 0] & (curvs2 > 0.07)[None, :, 0]] = base * 4
    diag_score[(curv_diff < 0.02) & color_sim & both_curvs_big] = base * 3
    
    
    return diag_score

def water(seq1, seq2, is_corner1, is_corner2, curvs1, curvs2):
    m, n = len(seq1), len(seq2)  # length of two sequences
    
    # Generate DP table and traceback path pointer matrix
    score = np.zeros((m+1, n+1))      # the DP table
    pointer = np.zeros((m+1, n+1))    # to store the traceback path
    
    max_score = 0        # initial maximum score in DP table
    gap_penalty = -100
    
    diag_score = compute_diag_score(curvs1, curvs2, seq1, seq2)
    
    
    # Calculate DP table and mark pointers
    for i in tqdm(range(1, m + 1)):
        for j in range(1, n + 1):
            score_diagonal = score[i - 1, j - 1] + diag_score[i - 1, j - 1]
            
            score_up, score_left = score[i,j-1] + gap_penalty, score[i-1,j] + gap_penalty
            m = max(0, score_left, score_up, score_diagonal)
            score[i,j] = m
            if m == 0:
                pointer[i,j] = 0 # 0 means end of the path
            elif m == score_diagonal:
                pointer[i,j] = 3 # 3 means trace diagonal
            elif m == score_left:
                pointer[i,j] = 1 # 1 means trace up
            elif m == score_up:
                pointer[i,j] = 2 # 2 means trace left
            if m >= max_score:
                max_i, max_j, max_score = i, j, score[i, j]
    
    align1, align2 = [], []    # initial sequences
    
    i,j = max_i,max_j    # indices of path starting point
    indices = []
    
    #traceback, follow pointers
    corner_met = False
    while pointer[i][j] != 0:
        indices.append((i, j))
        if pointer[i][j] == 3:
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif pointer[i][j] == 2:
            align1.append(None)
            align2.append(seq2[j-1])
            j -= 1
        elif pointer[i][j] == 1:
            align1.append(seq1[i-1])
            align2.append(None)
            i -= 1
    return indices, pointer, score

def fragment2shape_descriptor(palette, frag: Fragment) -> ShapeDescriptor:
    """Compute shape descriptors for fragment."""
    return ShapeDescriptor(
        frag.edge_colors,
        frag.edge_coords,
        edge_coords2curvatures(frag.edge_coords)
    )

def fragments2shape_descriptors(palette, fragments: List) -> List[ShapeDescriptor]:
    """Compute shape descriptors for fragments.

    Args:
        fragments: List of fragments.
    """
    return Parallel(n_jobs=-1)([delayed(fragment2shape_descriptor)(palette, f) for f in fragments])

def align_two_fragments(palette, frag1, frag2, to_print=None, shape_descriptor1=None, shape_descriptor2=None):
    if to_print is not None:
        print(to_print)
    if shape_descriptor1 is None:
        shape_descriptor1 = fragment2shape_descriptor(palette, frag1)
    color_edge1 , curvs1 = shape_descriptor1.color_edge, shape_descriptor1.curvatures
        
    if shape_descriptor2 is None:
        shape_descriptor2 = fragment2shape_descriptor(palette, frag2)
    color_edge2 , curvs2 = shape_descriptor2.color_edge, shape_descriptor2.curvatures

    return water(color_edge1, color_edge2[::-1], None, None, curvs1, curvs2[::-1])

def pairwise_alignment(palette, fragments: List) -> Tuple[List[ShapeDescriptor], Dict[Tuple[int, int], np.ndarray]]:
    """Compute pairwise alignment between fragments.

    Args:
        fragments: List of fragments.
    """
    print("Computing shape descriptors...")
    shape_descriptors = fragments2shape_descriptors(palette, fragments)
    print("Computing pairwise alignments...")
    alignment_dict =        {
            (i, j): align_two_fragments(
                palette,
                frag1, frag2, 
                to_print=f"Aligning fragments {i} and {j}:", 
                shape_descriptor1=shape_descriptors[i], 
                shape_descriptor2=shape_descriptors[j]
            )[0]
            for i, frag1 in enumerate(fragments)
            for j, frag2 in enumerate(fragments)
            if j > i
        }
        
    
    return shape_descriptors, alignment_dict


def estimate_max_squared_transformation_error(line1, line2, trasform_params):
    """
    line1, line2 - np.arrays, shape (n, 2)
    transforms_params - tuple (cos, sin, a, b) of transform parameters line1 -> line2
    returns mean squared error
    """
    transformed_line1 = transform_line(line1, trasform_params)
    return np.linalg.norm(line2 - transformed_line1, axis=1).max()

def iou(indices1, indices2):
    set1 = set([tuple(p) for p in indices1])
    set2 = set([tuple(p) for p in indices2])
    iou = len(set1.intersection(set2)) / len(set1.union(set2))
    return iou

def alignment_nms(aligns, edge_coords1, edge_coords2):
    aligns.sort(key=lambda x: x.conf, reverse=False)
    align_coords1 = [aligned_coords2line(aligns[i].indices, edge_coords1, left=True) for i in range(len(aligns))]
    align_coords2 = [aligned_coords2line(aligns[i].indices, edge_coords2[::-1], left=False) for i in range(len(aligns))]
    i = 0
    while i < len(aligns):
        j = i + 1
        while j < len(aligns):
            iou_score = min(iou(align_coords1[i], align_coords1[j]), iou(align_coords2[i], align_coords2[j]))
            if iou_score > 0.5:
                aligns.pop(j)
                align_coords1.pop(j)
                align_coords2.pop(j)
            else:
                j += 1
        i += 1
    return aligns

def backtrace(
    pointer,
    score,
    seq1, seq2,
    block_i, block_j, 
    block_size_y, block_size_x
):
    roi = score[block_i * block_size_y : (block_i + 1) * block_size_y, block_j * block_size_x : (block_j + 1) * block_size_x]
    argmax = np.argmax(roi)
    max_i, max_j = np.unravel_index(argmax, roi.shape)
    max_i, max_j = max_i + block_i * block_size_y, max_j + block_j * block_size_x
#     print(roi.shape, roi.max(), argmax, max_i, max_j)
    
    indices = []
    i, j = max_i, max_j
    while pointer[i][j] > 0:
        indices.append((i, j))
        if pointer[i][j] == 3:
            i -= 1
            j -= 1
        elif pointer[i][j] == 2:
            j -= 1
        elif pointer[i][j] == 1:
            i -= 1
    return np.array(indices)


def compute_max_curvatire(indices, dsc1, dsc2):
    max_curv1, max_curv2 = 0, 0
    curvs1, curvs2 = np.abs(dsc1.curvatures), np.abs(dsc2.curvatures[::-1])
    for (i, j) in indices:
        max_curv1 = max(max_curv1, curvs1[i - 1])
        max_curv2 = max(max_curv2, curvs2[j - 1])
    return min(max_curv1, max_curv2)

def generate_multiple_alignments(pointer, score, dsc1, dsc2, blocks_num):
    block_size_y = int(pointer.shape[0] / blocks_num)
    block_size_x = int(pointer.shape[1] / blocks_num)
    color_edge1 = dsc1.color_edge
    color_edge2 = dsc2.color_edge
    edge_coords1 = dsc1.edge_coords
    edge_coords2 = dsc2.edge_coords

    best_indices = None
    best_mse = 10000
    aligns = []
    for i in range(blocks_num):
        for j in range(blocks_num):
            indices = backtrace(pointer, score, color_edge1, color_edge2[::-1], i, j, block_size_y, block_size_x)
            if len(indices) < 50 or (len(indices) / len(edge_coords1) < 0.05 and len(indices) / len(edge_coords2) < 0.05):
#                 print("Skipping", i, j, len(indices), len(indices) / len(edge_coords1), len(indices) / len(edge_coords2))
                continue
            if compute_max_curvatire(indices, dsc1, dsc2) < 0.03:
                continue
            line1 = aligned_coords2line(indices, edge_coords1, left=True)
            line2 = aligned_coords2line(indices, edge_coords2[::-1], left=False)
            best_transform = find_best_transform_ransac(line1, line2)
            if best_transform is None:
                continue
            mse = estimate_max_squared_transformation_error(line1, line2, best_transform)
            aligns.append(Alignment(indices, mse))
    print(len(aligns))
    aligns = alignment_nms(aligns, edge_coords1, edge_coords2)
    print(len(aligns))
    return aligns

def align_two_frags_with_multiple_aligns(
        palette,
          frag1: Fragment, frag2, 
          to_print=None,
          blocks_num=6
          ):
    
    shape_dsc1 = fragment2shape_descriptor(palette, frag1)
    shape_dsc2 = fragment2shape_descriptor(palette, frag2)
    indices, pointer, score = align_two_fragments(
        palette,
        frag1, frag2,
        to_print=to_print,
        shape_descriptor1=shape_dsc1,
        shape_descriptor2=shape_dsc2,
    )
    aligns = generate_multiple_alignments(pointer, score, shape_dsc1, shape_dsc2, blocks_num)
    return aligns

def new_pairwise_alignment(palette, fragments: List, blocks_num=5) -> Tuple[List[ShapeDescriptor], Dict[Tuple[int, int], np.ndarray]]:
    """Compute pairwise alignment between fragments.

    Args:
        fragments: List of fragments.
    """
    print("Computing shape descriptors...")
    shape_descriptors = fragments2shape_descriptors(palette, fragments)
    print("Computing pairwise alignments...")
    alignment_dict = {}
    for i, frag1 in enumerate(fragments):
        for j, frag2 in enumerate(fragments):
            if j > i:
                indices, pointer, score = align_two_fragments(
                    palette,
                    frag1, frag2, 
                    to_print=f"Aligning fragments {i} and {j}:", 
                    shape_descriptor1=shape_descriptors[i], 
                    shape_descriptor2=shape_descriptors[j]
                )
                aligns = generate_multiple_alignments(pointer, score, shape_descriptors[i], shape_descriptors[j], blocks_num)
                alignment_dict[(i, j)] = aligns
    
    return shape_descriptors, alignment_dict