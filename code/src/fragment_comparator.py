import json
import numpy as np
import time

from src.shape_align import *
from src.utils import *
from src.refine_transform import *


class FragmentComparator:
    def __init__(self, blocks_num=6, data_dir="../../dataset") -> None:
        self.blocks_num = blocks_num
        self.data_dir = data_dir
        pass
    
    def pack_aligns_to_initial_params(
        self,
        alignments: List[Alignment], 
        descriptor1, 
        descriptor2, 
        fragments_shape
    ):
        """
        alignments: list of alignments from curvature matching
        descriptor1, descriptor2: shape descriptors of fragments
        fragments_shape: shape of fragments (assuming they have the same size)


        return: list of (angle, x, y), list of subcurves1, list of subcurves2 
        subcurves1[i] corresponds to subcurves2[i]
        """
        initial_params = []
        subcurves1 = []
        subcurves2 = []
        for alignment in alignments[:4]:
            line1 = aligned_coords2line(alignment.indices, descriptor1.edge_coords, left=True)
            line2 = aligned_coords2line(alignment.indices, descriptor2.edge_coords[::-1], left=False)
            line1 -= fragments_shape[0] // 2
            line2 -= fragments_shape[0] // 2
            best_transform_params = find_best_transform_ransac(line1, line2) # TODO: may be fix seed or increase number of iterations so that this step was more stable
    #         initial_params.append(best_transform_params)
            subcurves1.append(line1)
            subcurves2.append(line2)
            if best_transform_params is None:
                print("No best transform")
                continue
            cos = best_transform_params[0]
            cos = min(cos, 1)
            cos = max(cos, -1)
            theta, shift_x, shift_y = -np.rad2deg(np.arccos(cos)), best_transform_params[3], best_transform_params[2] # TODO: fix angle computation [0, pi] -> [-pi, pi]
            initial_params.append((theta, int(shift_x), int(shift_y)))
        return initial_params, subcurves1, subcurves2
    
    def align_two_frags_with_multiple_aligns(
            self,
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
        return shape_dsc1, shape_dsc2, aligns
    
    def align_two_fragments_by_idx(
        self, 
        idx1, idx2,
    ):
        print(f"Aligning {idx1} and {idx2}")
        start_time = time.time()
        frag1, frag2 = build_fragment_from_directory(self.data_dir + '/' + str(idx1)), build_fragment_from_directory(self.data_dir + '/' + str(idx2))
        pad_size = int(max([max(frag1.fragment.shape), max(frag2.fragment.shape)]) / 3) + 40
        frag1, frag2 = pad_fragment_to_size(frag1, pad_size), pad_fragment_to_size(frag2, pad_size)
        
        dsc1, dsc2, aligns = self.align_two_frags_with_multiple_aligns(
            None, # fix palette usage
            frag1, frag2,
            blocks_num=self.blocks_num,
            to_print=None
        )
        initial_params, subcurves1, subcurves2 = self.pack_aligns_to_initial_params(aligns, dsc1, dsc2, frag1.fragment.shape)
        alignments = match_two_aligned_fragments(
            frag1, frag2,
            initial_params,
            subcurves1,
            subcurves2,
#             feature_extractor,
            pad_size=pad_size,
            verbose=1
        )
        print(f"Alignment time: {time.time() - start_time}")
        return sorted(alignments, key=lambda x: x.confidence, reverse=True)
    
    def align_frags_and_save(
        self,
        frag_nums,
        existing_json='alignments_merged.json'        
    ):
        aligns = {}
        json_dict = None
        if existing_json is not None:
            json_dict = json.load(open(existing_json, 'r'))
        for i in range(len(frag_nums)):
            for j in range(i + 1, len(frag_nums)):
#                 aligns += self.align_two_fragments(frag_nums[i], frag_nums[j])
#                 print(f"Aligning {frag_nums[i]} and {frag_nums[j]}")

                if json_dict is not None:
                    if f"{frag_nums[i]}_{frag_nums[j]}" in json_dict:
                        continue
                translations = self.align_two_fragments_by_idx(frag_nums[i], frag_nums[j])
                print(f"Length: {len(translations)}")
    
                pair_aligns = []
                for tr in translations:
                    pair_aligns.append({
                        'x': tr.x,
                        'y': tr.y,
                        'angle': tr.angle,
                        'confidence': tr.confidence,
                        'geom_score': tr.geom_score
                    })

                aligns[f"{frag_nums[i]}_{frag_nums[j]}"] = pair_aligns
        with open('alignments1.json', 'w') as f:
            json.dump(aligns, f)