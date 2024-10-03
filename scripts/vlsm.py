from pathlib import Path
import nibabel as nib
import numpy as np 
import math
from scipy import stats
import sys
import pandas as pd
import time 
import os
from tqdm import tqdm

dir_masks = Path(sys.argv[1])
list_masks = sorted(dir_masks.glob("**/wssub*mask.nii"))
subjects = {"001":"02", "005":"01", "006":"02", "008":"01", "009":"02", "010":"01", "012":"01", "013":"01", "014":"01", "015":"02", "030":"01", "031":"02", "032":"01", "036":"01", "037":"01", "039":"01", "040":"01", "041":"01", "043":"01","044":"01", "063":"01", "066":"01", "068":"01", "069":"01", "070":"01", "071":"01", }

scores = pd.read_csv(sys.argv[2], delimiter="\t")
# tasks = ["GMFCS score","AHA score total","AHA score logit","GMFM partie E","GMFM score global","MA2 dominante amplitude","MA2 dominante precision","MA2 dominante dexterite","MA2 dominante fluidite","MA2 autre amplitude","MA2 autre precision","MA2 autre dexterite","MA2 autre fluidite"]#,"Fonctions executives inhibition"]
tasks = list(scores.columns.values)[1:-1]

i=0
shape_imgs = nib.load(list_masks[0]).get_fdata().shape
sum_masks = np.zeros(shape_imgs)
t_map = np.empty((len(tasks), shape_imgs[0], shape_imgs[1], shape_imgs[2]))
p_value_map = np.empty((len(tasks), shape_imgs[0], shape_imgs[1], shape_imgs[2]))

t_map_after_permutation = np.empty((len(tasks), shape_imgs[0], shape_imgs[1], shape_imgs[2]))
p_value_map_after_permutation = np.empty((len(tasks), shape_imgs[0], shape_imgs[1], shape_imgs[2]))

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

number_iter = np.where(sum_masks>=2, 1.0, 0.0)
number_iter = np.sum(np.where(number_iter<=len(subjects)-2, 1.0, 0.0))
print(f"{number_iter} iterations")
# fwer = 1-math.pow((1-0.05),comparison_number)
# print(fwer) 

t_start = time.time()
for x in tqdm(range(shape_imgs[0])):
    for y in range(shape_imgs[1]):
        for z in range(shape_imgs[2]):
            if sum_masks[x,y,z]>=2 and sum_masks[x,y,z]<=len(subjects)-2:
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
                    t, p = stats.ttest_ind(list_scores_lesion, list_scores_no_lesion)                   
                    # r, pvalue_corrected = stats.ttest_ind(list_scores_lesion, list_scores_no_lesion, alternative='less', permutations=1000)
                    t_map[num_task, x,y,z] = t
                    p_value_map[num_task, x,y,z] = p
                    # t_map_after_permutation[num_task, x,y,z] = r
                    # p_value_map_after_permutation[num_task, x,y,z] = pvalue_corrected

print(f"{time.time() - t_start} seconds")

for num_task in range(len(tasks)):
    ni_img = nib.Nifti1Image(t_map[num_task, :,:,:], nib.load(list_masks[0]).affine)
    out = './out/t-maps/'
    if not os.path.exists(out):
        os.makedirs(out)
    nib.save(ni_img, os.path.join(out, 't-map_'+tasks[num_task]+'.nii.gz'))

    ni_img = nib.Nifti1Image(p_value_map[num_task, :,:,:], nib.load(list_masks[0]).affine)
    out = './out/p-value-maps/'
    if not os.path.exists(out):
        os.makedirs(out)
    nib.save(ni_img, os.path.join(out, 'p-value-map_'+tasks[num_task]+'.nii.gz'))

    # ni_img = nib.Nifti1Image(t_map_after_permutation[num_task, :,:,:], nib.load(list_masks[0]).affine)
    # out = './out/t-maps-after_permutation/'
    # if not os.path.exists(out):
    #     os.makedirs(out)
    # nib.save(ni_img, os.path.join(out, 't-maps-after-permutation_'+tasks[num_task]+'.nii.gz'))

    # ni_img = nib.Nifti1Image(p_value_map_after_permutation[num_task, :,:,:], nib.load(list_masks[0]).affine)
    # out = './out/p-value-maps-after_permutation/'
    # if not os.path.exists(out):
    #     os.makedirs(out)
    # nib.save(ni_img, os.path.join(out, 'p-value-maps-after-permutation_'+tasks[num_task]+'.nii.gz'))

#'''