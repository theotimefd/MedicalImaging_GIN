import os
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm as tqdm


dataset = "/home/fehrdelt/data_ssd/data/final_dataset_brain_extraction/fixed_distribution/test_dataset/"
output_folder = "/home/fehrdelt/data_ssd/data/output_totalsegmentator/"


def run():
    for file in os.listdir(dataset):
        if file.endswith(".nii.gz"):
            totalsegmentator(dataset+file, output_folder+file[:17], roi_subset=["brain"])


if __name__=="__main__":
    run()