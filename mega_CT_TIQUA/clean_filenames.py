import os

MAIN_DIRECTORY = "/home/fehrdelt/data_ssd/data/FastDiag/"

for file in os.listdir(MAIN_DIRECTORY):
    if file.endswith(".nii.gz") or file.endswith(".json"):
    	if file[:10]=="to_convert":
	        #os.rename(MAIN_DIRECTORY+file, MAIN_DIRECTORY+file.split('_')[1])
        
        	os.rename(MAIN_DIRECTORY+file, MAIN_DIRECTORY+file[11:].split('_')[1])

