from python_scripts.Volume_estimation import combined_regions_lesion_nifti
import os

directory = "/home/fehrdelt/data_ssd/data/mega_CT_TIQUA_temp/"

def run():
    for file in os.listdir(directory):
        if os.path.isdir(directory+file):
            try:
                print(f"trying patient {file}")
                combined_regions_lesion_nifti(atlas=directory+file+f"/{file}_matlab_Atlas_ANTS_hist_match_Registered.nii.gz",
                                          seg=directory+file+f"/{file}_seg.nii.gz", 
                                          atlas_labels_csv="/home/fehrdelt/data_ssd/MedicalImaging_GIN/CT-TIQUA_V4/data/Labels_With_0.csv",
                                          out_nifti_path=directory+file+"/combined_seg.nii.gz")
            except:
                print(f"Error with patient {file}")



if __name__=="__main__":
    run()
    
