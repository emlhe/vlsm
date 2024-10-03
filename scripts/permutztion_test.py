from pathlib import Path 
import sys
import nibabel as nib
import numpy as np 
import os 
from scipy import stats, permutation_test

dir_maps = Path(sys.argv[1])
maps_paths = sorted(dir_maps.glob("**/*.nii.gz"))
print(maps_paths)

def statistic(x, y, axis):
    return stats.ttest_ind(x, y)

for map_path in maps_paths:
    map_volume = nib.load(map_path)
    map = map_volume.get_fdata()
    task = str(map_path).split("_")[-1].split(".nii.gz")[0]
    print(f"### {task} : ")
    
    number_comparisons = np.sum(np.where(map.astype(np.uint8) != 0, 1, 0))
    
    p_values_fwer_correction = np.where(map.astype(np.uint8)*number_comparisons == 0, 1, 0)
    p_values_fwer_correction = (p_values_fwer_correction < 0.05).astype(np.uint8)
    print(f"\tBonferroni fwe correction : {np.sum(p_values_fwer_correction)} voxels survived")
    ni_img = nib.Nifti1Image(p_values_fwer_correction, map_volume.affine)
    out = './out/p_value_maps_bonferroni/'
    if not os.path.exists(out):
        os.makedirs(out)
    nib.save(ni_img, os.path.join(out, "p-value-bonferroni-corrected-map_"+task+".nii.gz"))
    