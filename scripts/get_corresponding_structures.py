from pathlib import Path
import sys
import nibabel as nib
import numpy as np 

thresholded_map_path = Path(sys.argv[1])
atlas_path = Path(sys.argv[2])

thresholded_map_volume = nib.load(thresholded_map_path)
thresholded_map = thresholded_map_volume.get_fdata()

atlas_volume = nib.load(atlas_path)
atlas = atlas_volume.get_fdata()

corresponding_structures = np.unique(np.where(thresholded_map !=0, atlas, 0))[1:]
print(corresponding_structures)

sum_array = np.zeros(atlas.shape)
for e in corresponding_structures:
    structure = np.where(atlas==e, atlas, 0)
    sum_array = np.add(sum_array, structure)

print(sum_array.shape)
print(np.unique(sum_array))

ni_img = nib.Nifti1Image(sum_array, thresholded_map_volume.affine)
nib.save(ni_img, 'final_structures_map_jhu.nii.gz')
