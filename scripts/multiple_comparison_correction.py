from pathlib import Path 
import sys
import nibabel as nib
import numpy as np 
import os 

dir_maps = Path(sys.argv[1])
maps_paths = sorted(dir_maps.glob("**/*.nii.gz"))
print(maps_paths)

def false_discovery_control(ps, *, axis=0, method='bh'):
    
    # Input Validation and Special Cases
    ps = np.asarray(ps)

    ps_in_range = (np.issubdtype(ps.dtype, np.number)
                   and np.all(ps == np.clip(ps, 0, 1)))
    if not ps_in_range:
        raise ValueError("`ps` must include only numbers between 0 and 1.")

    methods = {'bh', 'by'}
    if method.lower() not in methods:
        raise ValueError(f"Unrecognized `method` '{method}'."
                         f"Method must be one of {methods}.")
    method = method.lower()

    if axis is None:
        axis = 0
        ps = ps.ravel()

    axis = np.asarray(axis)[()]
    if not np.issubdtype(axis.dtype, np.integer) or axis.size != 1:
        raise ValueError("`axis` must be an integer or `None`")

    if ps.size <= 1 or ps.shape[axis] <= 1:
        return ps[()]

    ps = np.moveaxis(ps, axis, -1)
    m = ps.shape[-1]

    order = np.argsort(ps, axis=-1)
    ps = np.take_along_axis(ps, order, axis=-1)  # this copies ps

    i = np.arange(1, m+1)
    ps *= m / i

    # Theorem 1.3 of [2]
    if method == 'by':
        ps *= np.sum(1 / i)

    np.minimum.accumulate(ps[..., ::-1], out=ps[..., ::-1], axis=-1)

    np.put_along_axis(ps, order, values=ps.copy(), axis=-1)
    ps = np.moveaxis(ps, -1, axis)

    return np.clip(ps, 0, 1)    

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
    
    p_value_map_1d = map.flatten()
    corrected_p_value_map = false_discovery_control(p_value_map_1d, method='bh').reshape(map.shape)
    p_values_fdr_correction = np.where(corrected_p_value_map == 0, 1, corrected_p_value_map)
    p_values_fdr_correction = (p_values_fdr_correction < 0.05).astype(np.uint8)
    print(f"\tBH fdr correction : {np.sum(p_values_fdr_correction)} voxels survived")
    ni_img = nib.Nifti1Image(p_values_fdr_correction, map_volume.affine)
    out = './out/p_value_maps_bh/'
    if not os.path.exists(out):
        os.makedirs(out)
    nib.save(ni_img, os.path.join(out, 'p-value-bh-corrected-map_'+task+'.nii.gz'))

    p_value_map_1d = map.flatten()
    corrected_p_value_map = false_discovery_control(p_value_map_1d, method='by').reshape(map.shape)
    p_values_fdr_correction = np.where(corrected_p_value_map == 0, 1, corrected_p_value_map)
    p_values_fdr_correction = (p_values_fdr_correction < 0.05).astype(np.uint8)
    print(f"\tBY fdr correction : {np.sum(p_values_fdr_correction)} voxels survived")
    ni_img = nib.Nifti1Image(p_values_fdr_correction, map_volume.affine)
    out = './out/p_value_maps_by/'
    if not os.path.exists(out):
        os.makedirs(out)
    nib.save(ni_img, os.path.join(out, 'p-value-by-corrected-map_'+task+'.nii.gz'))


    
    

    