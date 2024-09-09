from nipype.interfaces import fsl
from pathlib import Path 
import os

dir_masks = Path("/home/emlhe/Thèse/DATA/CAP/training_data/")
list_imgs = sorted(dir_masks.glob("**/*T1w.nii"))
list_masks = sorted(dir_masks.glob("**/*mask.nii"))
subjects = {"001":"02", "005":"01", "006":"02", "008":"01", "009":"02", "010":"01", "012":"01", "013":"01", "014":"01", "015":"02", "030":"01", "031":"02", "032":"01", "036":"01", "037":"01", "039":"01", "040":"01", "041":"01", "043":"01","044":"01", "063":"01", "066":"01", "068":"01", "069":"01", "070":"01", "071":"01", }

mni_path = "/media/emlhe/Disque\ Emma/DATA/VLSM/nihpd_sym_04.5-08.5_t1w.nii"

assert len(list_imgs) == len(list_masks)
i=0
for mask in list_masks:
    subject = str(mask).split("sub-")[-1][:3]
    session = str(mask).split("ses-")[-1][:2]    
    if subject in subjects.keys():
        if subjects[subject] != session:
            list_imgs.pop(i)
            list_masks.pop(i)
    else:
        list_imgs.pop(i)
        list_masks.pop(i)
    
    i+=1

for (img, mask) in zip(list_imgs, list_masks):
    subject = str(mask).split("sub-")[-1][:3]
    session = str(mask).split("ses-")[-1][:2]  
    
    sub_dir = Path(f"/media/emlhe/Disque Emma/DATA/VLSM/out/sub-{subject}")
    if not os.path.isdir(sub_dir):
        os.mkdir(sub_dir)
    
    out_affine_mat = os.path.join(sub_dir,f"affine_transfo_sub-{subject}_ses-{session}_to_mni.mat")
    out_non_linear = os.path.join(sub_dir,f"non_linear_transfo_sub-{subject}_ses-{session}_to_mni")
    out_t1_aff_to_mni = os.path.join(sub_dir,f"sub-{subject}_ses-{session}_affine_T1w.nii")
    out_t1_nl_to_mni = os.path.join(sub_dir,f"sub-{subject}_ses-{session}_normalized_T1w.nii")
    out_mask_nl_to_mni = os.path.join(sub_dir,f"sub-{subject}_ses-{session}_normalized_mask.nii")


    print(img)
    print(mni_path)
    print(out_affine_mat)
    print(out_t1_aff_to_mni)
    print(type(out_affine_mat))

    
    flt = fsl.FLIRT()
    flt.inputs.in_file = img
    flt.inputs.reference = mni_path
    flt.inputs.out_matrix_file = out_affine_mat
    flt.inputs.out_file=out_t1_aff_to_mni
    flt.cmdline 
    res = flt.run() 

    fnt = fsl.FNIRT()
    res=fnt.run(in_file=img, ref_file=mni_path, affine_file=out_affine_mat, fieldcoeff_file = out_non_linear) 

    aw = fsl.ApplyWarp()
    aw.run(in_file=img, ref_file=mni_path, field_file=out_non_linear, out_file=out_t1_nl_to_mni)
    
    aw = fsl.ApplyWarp()
    aw.run(in_file=mask, ref_file=mni_path, field_file=out_non_linear, out_file=out_mask_nl_to_mni)
   