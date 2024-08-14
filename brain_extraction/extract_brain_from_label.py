import os
import nibabel as nib


directory = "/home/fehrdelt/data_ssd/data/"

for file in os.listdir(directory+"all_labels/"):
    
    if file.endswith(".nii.gz"):
    
        label = nib.load(directory+"all_labels/"+file)
        image = nib.load(directory+"all_images/"+file)
        
        extracted_brain = nib.Nifti1Image(label.get_fdata()*image.get_fdata(), image.affine, image.header)
        
        nib.save(extracted_brain, "/home/fehrdelt/data_ssd/data/SkullStripped_ground_truth/"+file[6:]+"_SkullStripped_clean.nii.gz")
        