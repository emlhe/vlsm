from pathlib import Path
import nibabel as nib
import numpy as np 
import math
from scipy import stats
import sys
import pandas as pd

dir_masks = Path(sys.argv[1])
list_masks = sorted(dir_masks.glob("**/*mask_flirt.nii.gz"))
subjects = {"001":"02", "005":"01", "006":"02", "008":"01", "009":"02", "010":"01", "012":"01", "013":"01", "014":"01", "015":"02", "030":"01", "031":"02", "032":"01", "036":"01", "037":"01", "039":"01", "040":"01", "041":"01", "043":"01","044":"01", "063":"01", "066":"01", "068":"01", "069":"01", "070":"01", "071":"01", }

scores = pd.read_csv("./data/scores.csv")
tasks = ["GMFCS score","AHA score total","AHA score logit","GMFM partie E","GMFM score global","MA2 dominante amplitude","MA2 dominante precision","MA2 dominante dexterite","MA2 dominante fluidite","MA2 autre amplitude","MA2 autre precision","MA2 autre dexterite","MA2 autre fluidite","Fonctions executives inhibition"]



print(f"{len(subjects)} subjects")

i=0
shape_imgs = nib.load(list_masks[0]).get_fdata().shape
sum_masks = np.zeros(shape_imgs)
t_map = np.zeros(shape_imgs).unzqueeze(0)
p_value_map = np.zeros(shape_imgs).unzqueeze(0)
imgs = []
for mask in list_masks:
    subject = str(mask).split("sub-")[-1][:3]
    session = str(mask).split("ses-")[-1][:2]    
    if subject in subjects.keys():
        if subjects[subject] == session:
            print(f"{i} : subject {subject}, session {session}")
            i+=1
            imgs.append(np.where(nib.load(mask).get_fdata()>0.5, 1.0, 0.0))
            sum_masks = sum_masks + imgs[-1]
            
ni_img = nib.Nifti1Image(sum_masks, nib.load(list_masks[0]).affine)
nib.save(ni_img, 'sum_masks.nii.gz')
print("saved")

comparison_number = np.sum(np.where(sum_masks>=5, 1.0, 0.0))
print(comparison_number)
fwer = 1-math.pow((1-0.05),comparison_number)
print(fwer)
for mask in list_masks:
    subject = str(mask).split("sub-")[-1][:3]
    session = str(mask).split("ses-")[-1][:2]    
    for x in range(shape_imgs[0]):
        print(x)
        for y in range(shape_imgs[1]):
            for z in range(shape_imgs[2]):
                if sum_masks[x,y,z]>=5:
                    for num_task in range(len(tasks)):
                        list_scores_lesion = []
                        list_scores_no_lesion = []
                        i=0
                        for img in imgs:
                            if img[x,y,z]==1:
                                list_scores_lesion.append(scores[tasks[num_task]].to_list()[i])
                            else:
                                list_scores_no_lesion.append(scores[tasks[num_task]].to_list()[i])
                            i+=1
                        t, p = stats.ttest_ind(list_scores_lesion, list_scores_no_lesion, alternative='less')
                        t_map[num_task, x,y,z] = t
                        p_value_map[num_task, x,y,z] = p

for num_task in range(len(tasks)):
    # p_value_map_aha = nib.load("p-value-map_aha.nii.gz")
                        
    ni_img = nib.Nifti1Image(t_map[num_task,:,:,:], nib.load(list_masks[0]).affine)
    nib.save(ni_img, 't-map_'+tasks[num_task]+'.nii.gz')
    ni_img = nib.Nifti1Image(p_value_map, nib.load(list_masks[0]).affine)
    nib.save(ni_img, 'p-value-map_'+tasks[num_task]+'.nii.gz')
    comparison_number = 197819

    p_values_fwer_correction = np.where(p_value_map[num_task, :,:,:].get_fdata().astype(np.uint8)*comparison_number == 0, 1, 0)
    p_values_fwer_correction = (p_values_fwer_correction < 0.05).astype(np.uint8)
    print(f"{np.sum(p_values_fwer_correction)} voxels survived")
    ni_img = nib.Nifti1Image(p_values_fwer_correction,nib.load(list_masks[0]).affine)
    nib.save(ni_img, 'p-value-fwer-corrected-map_'+tasks[num_task]+'.nii.gz')


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

for num_task in range(len(tasks)):
    p_value_map_1d = p_value_map[num_task, :,y:,:].flatten()
    corrected_p_value_map = false_discovery_control(p_value_map_1d).reshape(shape_imgs)

    p_values_fdr_correction = np.where(corrected_p_value_map == 0, 1, 0)
    p_values_fdr_correction = (p_values_fdr_correction < 0.05).astype(np.uint8)
    print(f"{np.sum(p_values_fdr_correction)} voxels survived")
    ni_img = nib.Nifti1Image(p_values_fdr_correction, nib.load(list_masks[0]).affine)
    nib.save(ni_img, 'p-value-fdr-corrected-map_'+tasks[num_task]+'.nii.gz')