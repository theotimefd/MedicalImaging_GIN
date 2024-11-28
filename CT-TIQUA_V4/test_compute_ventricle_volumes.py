from python_scripts.Volume_estimation import ventricle_volume_computation
import os

directory = "/home/fehrdelt/data_ssd/data/mega_CT_TIQUA_temp/"

def run():
    for file in os.listdir(directory):
        if os.path.isdir(directory+file):
            try:
                print(f"trying patient {file}")
                ventricle_volume_computation(seg_path = directory+file+"/TTS_brain_structures/ventricle.nii.gz", 
                                             atlas_path=directory+file+f"/{file}_matlab_Atlas_ANTS_hist_match_Registered.nii.gz", 
                                             out_csv_path=directory+file+"/ventricle_volumes.csv")
            
            except Exception as e:
                print(f"Error with patient {file}")
                print(e)



if __name__=="__main__":
    run()
    
