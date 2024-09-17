from pathlib import Path
import nibabel as nib
import numpy as np 
import math
from scipy import stats
import sys
import pandas as pd
import time 

dir_masks = Path(sys.argv[1])
list_masks = sorted(dir_masks.glob("**/*mask.nii"))
subjects = {"001":"02", "005":"01", "006":"02", "008":"01", "009":"02", "010":"01", "012":"01", "013":"01", "014":"01", "015":"02", "030":"01", "031":"02", "032":"01", "036":"01", "037":"01", "039":"01", "040":"01", "041":"01", "043":"01","044":"01", "063":"01", "066":"01", "068":"01", "069":"01", "070":"01", "071":"01", }

scores = pd.read_csv("/media/emlhe/Disque Emma/DATA/derivatives/scores_tri.csv", delimiter="\t")
tasks = ["GMFCS score","AHA score total","AHA score logit","GMFM partie E","GMFM score global","MA2 dominante amplitude","MA2 dominante precision","MA2 dominante dexterite","MA2 dominante fluidite","MA2 autre amplitude","MA2 autre precision","MA2 autre dexterite","MA2 autre fluidite"]#,"Fonctions executives inhibition"]

i=0
shape_imgs = nib.load(list_masks[0]).get_fdata().shape
sum_masks = np.zeros(shape_imgs)
t_map = np.empty((len(tasks), shape_imgs[0], shape_imgs[1], shape_imgs[2]))
p_value_map = np.empty((len(tasks), shape_imgs[0], shape_imgs[1], shape_imgs[2]))

print(t_map.shape)

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

comparison_number = np.sum(np.where(sum_masks>=5, 1.0, 0.0))
print(comparison_number)
fwer = 1-math.pow((1-0.05),comparison_number)
print(fwer) 


t_start = time.time()
progression = 0
for x in range(shape_imgs[0]):
    for y in range(shape_imgs[1]):
        for z in range(shape_imgs[2]):
            if sum_masks[x,y,z]>=5:
                progression+=1
                if int((progression/ comparison_number)) == (progression / comparison_number):
                    print(f"{time.time() - t_start} : {progression/ comparison_number}%")
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
                    t_map[num_task, x,y,z] = t
                    p_value_map[num_task, x,y,z] = p
print(f"{time.time() - t_start} seconds")

#'''