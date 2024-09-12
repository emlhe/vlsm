import nipype; 
from nipype.interfaces import fsl
from pathlib import Path 
import os
import nipype.interfaces.spm as spm
import time
import datetime
import sys

def get_subjects_list(list_img, sessions_dict):
    i=0
    while i < len(list_img):
        path=list_img[i]
        sub = str(path).split("sub-")[-1][:3]
        ses = str(path).split("ses-")[-1][:2]

        print(f"{i} : sub {sub}, ses {ses} :")

        if not sub in sessions_dict.keys():
            print("\tSubject not in list")
            list_img.pop(i)
        else:
            if not sessions_dict.get(sub) == ses:
                print("\tNot the right session")
                list_img.pop(i)
            else:
                print("\tSubject in list, right session")
                i+=1
    return list_img


def run_normalization(list_imgs, list_masks, method='fsl'):
    for (img, mask) in zip(list_imgs, list_masks):
        t1 = time.time()
        subject = str(img).split("sub-")[-1][:3]
        session = str(img).split("ses-")[-1][:2]  
        
        sub_dir = Path(f"out/sub-{subject}")
        if not os.path.isdir(sub_dir):
            os.mkdir(sub_dir)
        
        out_affine_mat = Path(os.path.join(sub_dir,f"affine_transfo_sub-{subject}_ses-{session}_to_mni.mat"))
        out_img_roi = f"{str(img).split('.nii')[0]}_roi.nii.gz"
        out_non_linear = Path(os.path.join(sub_dir,f"non_linear_transfo_sub-{subject}_ses-{session}_to_mni.nii.gz"))
        out_t1_aff_to_mni = Path(os.path.join(sub_dir,f"sub-{subject}_ses-{session}_affine_T1w.nii.gz"))
        out_t1_nl_to_mni = Path(os.path.join(sub_dir,f"sub-{subject}_ses-{session}_normalized_T1w.nii.gz"))
        out_mask_nl_to_mni = Path(os.path.join(sub_dir,f"sub-{subject}_ses-{session}_normalized_mask.nii.gz"))

        print(img)
        print(mni_path)
        print(out_affine_mat)
        print(out_t1_aff_to_mni)
        print(type(out_affine_mat))

        if method == "fsl":    
            # rfov = fsl.RobustFOV()
            # rfov.inputs.in_file = img
            # rfov.inputs.out_roi = out_img_roi
            # print(rfov.cmdline)
            # rfov.run()

            flt = fsl.FLIRT()
            flt.inputs.in_file = img
            flt.inputs.reference = mni_path
            flt.inputs.out_matrix_file = out_affine_mat
            flt.inputs.out_file=out_t1_aff_to_mni
            print(flt.cmdline)
            flt.run() 

            tfnirt = time.time()
            print(f"FNIRT started at {datetime.datetime.fromtimestamp(tfnirt).strftime('%c')}")
            fnt = fsl.FNIRT()
            fnt.inputs.in_file = img
            fnt.inputs.ref_file = mni_path
            fnt.inputs.affine_file = out_affine_mat
            fnt.inputs.fieldcoeff_file = out_non_linear
            print(fnt.cmdline)
            fnt.run() 
            print(f"FNIRT took {time.time()-tfnirt} sec")

            aw = fsl.ApplyWarp()
            aw.inputs.in_file=img
            aw.inputs.ref_file=mni_path
            aw.inputs.field_file=out_non_linear
            aw.inputs.out_file=out_t1_nl_to_mni
            print(aw.cmdline)
            aw.run()
            
            aw = fsl.ApplyWarp()
            aw.inputs.in_file=mask
            aw.inputs.ref_file=mni_path
            aw.inputs.field_file=out_non_linear
            aw.inputs.out_file=out_mask_nl_to_mni
            print(aw.cmdline)
            aw.run()
    
        elif method == "spm":    
            spm.SPMCommand.set_mlab_paths(paths='/home/emma/Documents/MATLAB/spm12')
            spm.SPMCommand().version
            norm = spm.Normalize()
            norm.inputs.source = img
            norm.inputs.template = mni_path
            norm.inputs.apply_to_files = mask
            
            norm.run() 
        
        print(f"Subject {subject} took {time.time()-t1} sec")

if __name__ == "__main__":
    dir_imgs = Path(sys.argv[1])
    list_imgs = sorted(dir_imgs.glob("**/sub-*_synthstripped.nii"))
    list_masks = sorted(dir_imgs.glob("**/sub-*_mask.nii"))
    # subjects = {"001":"02", "005":"01", "006":"02", "008":"01", "009":"02", "010":"01", "012":"01", "013":"01", "014":"01", "015":"02", "030":"01", "031":"02", "032":"01", "036":"01", "037":"01", "039":"01", "040":"01", "041":"01", "043":"01","044":"01", "063":"01", "066":"01", "068":"01", "069":"01", "070":"01", "071":"01", }
    # subjects = {"010":"01", "012":"01", "013":"01", "014":"01", "015":"02", "030":"01", "031":"02", "032":"01", "036":"01", "037":"01", "039":"01", "040":"01", "041":"01", "043":"01","044":"01", "063":"01", "066":"01", "068":"01", "069":"01", "070":"01", "071":"01", }
    subjects = {"066":"01", "068":"01", "069":"01", "070":"01", "071":"01"}

    mni_path = "data/nihpd_sym_04.5-08.5_t1w_brain_roi.nii.gz"
    list_imgs_filtered = get_subjects_list(list_imgs, subjects)
    list_masks_filtered = get_subjects_list(list_masks, subjects)

    assert len(list_imgs_filtered) == len(list_masks_filtered)
    run_normalization(list_imgs_filtered, list_masks_filtered, method='fsl')