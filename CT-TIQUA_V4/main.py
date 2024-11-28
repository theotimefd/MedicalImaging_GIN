import sys
sys.path.append('demeter')


import os
import time
import nibabel as nib
import numpy as np
import scipy.ndimage
from bisect import bisect_left
import pandas as pd

from nipype.interfaces import fsl
import ants
import torch
from totalsegmentator.python_api import totalsegmentator
from monai.transforms import LoadImage, Compose, Resize

from python_scripts.apply_custom_nn_brain_extraction import apply_brain_extraction_net
from python_scripts.Volume_estimation import Single_Volume_Inference
from python_scripts.Script_Apply_DynUnet import ApplyDynUnet



# -------------------------------------------------------

# Process 0 GPU device 0 

# ---------------------- CONSTANTS ----------------------

#MAIN_DIRECTORY = "/home/fehrdelt/data_ssd/data/FastDiag_2_0/"
MAIN_DIRECTORY = "/home/fehrdelt/data_ssd/data/CORRECT-TBI/"
TEMP_DIRECTORY = "/home/fehrdelt/data_ssd/data/CORRECT-TBI_temp/"
DATA_DIRECTORY = "/home/fehrdelt/data_ssd/MedicalImaging_GIN/CT-TIQUA_V4/data/"

#MATLAB_APP_PATH = '/data_network/irmage_pa/_SHARE/DOCKER_CT-TIQUA/docker_light_test/compiled_matlab_scripts/App/application/run_SkullStrip.sh'
#MATLAB_RUNTIME_PATH = '/data_network/irmage_pa/_SHARE/DOCKER_CT-TIQUA/docker_light_test/compiled_matlab_scripts/RunTime/v910'

CLAMP_LOW_THRESHOLD = 1.0
CLAMP_HIGH_THRESHOLD = 80

GPU_DEVICE = 0

PIXDIM = [1,1,1]


# ---------------------- Utility Functions ----------------------

def take_closest(list, number):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(list, number)
    if pos == 0:
        return list[0]
    if pos == len(list):
        return list[-1]
    before = list[pos - 1]
    after = list[pos]
    if after - number < number - before:
        return after
    else:
        return before

def clamp_values(image, low_threshold, high_threshold):
    new_img = image.copy()
    
    new_img[new_img < low_threshold] = 0
    new_img[new_img > high_threshold] = high_threshold
    return new_img

def histogram_matching(target, source):
    n_bins = 100
    # Calcul des histogrammes et des fonctions de répartition cumulée
    mask_target = target > 1e-5
    hist_target, edge_bin_T = np.histogram(target[mask_target], bins=n_bins, density=True)
    cdf_target = hist_target.cumsum()
    # ic(hist_target,edge_bin_T, cdf_target)

    hist_source, edge_bin_S = np.histogram(source[source > 0], bins=n_bins, density=True)
    cdf_source = hist_source.cumsum()
    
    # Création de la nouvelle image
    new_T = np.zeros_like(target)
    for i, (gt_m, gt_M) in enumerate(zip(edge_bin_T[:-1], edge_bin_T[1:])):
        gs = np.argmin(np.abs(cdf_target[i] - cdf_source))
        mask_gt = np.logical_and((target > gt_m), (target < gt_M))
        np.putmask(new_T, mask_gt, (gs / n_bins))

    new_T[~mask_target] = 0
    return new_T

# ---------------------- Main Steps ----------------------

def brain_extraction(basename, device):

    # matlab 
    #img_resampled = nib.load(TEMP_DIRECTORY+basename+"_resampled.nii.gz")
    #nib.save(img_resampled, TEMP_DIRECTORY+basename+"_resampled.nii") # need the same input file but in nii format for the matlab algorithm
    #output_path = TEMP_DIRECTORY+basename+"_matlab_skullstripped.nii" # actually this will save as nii.gz with the matlab aglorithm idk why
    #output_ROI = TEMP_DIRECTORY+basename+"_matlab_ROI.nii"            # same here
    #os.system(MATLAB_APP_PATH+' '+MATLAB_RUNTIME_PATH+' '+TEMP_DIRECTORY+basename+"_resampled.nii"+' '+output_path+' '+output_ROI)

    # TotalSegmentator
    totalsegmentator(TEMP_DIRECTORY+basename+"_resampled.nii.gz", TEMP_DIRECTORY, roi_subset=["brain"], device=f"gpu:{GPU_DEVICE}")
    brain_mask = nib.load(TEMP_DIRECTORY+"brain.nii.gz")
    original_img = nib.load(TEMP_DIRECTORY+basename+"_resampled.nii.gz")
    TTS_extracted_brain = nib.Nifti1Image(brain_mask.get_fdata()*original_img.get_fdata(), original_img.affine, original_img.header)
    nib.save(TTS_extracted_brain, TEMP_DIRECTORY+basename+"_TTS_skullstripped.nii.gz")


def segmentation(basename, device):
    
    print('Start of the lesion segmentation: DynUnet on raw images (xxmm3)...')
    
    model_path = DATA_DIRECTORY+"24-02-23-13h18m_best_model.pt"
    
    ApplyDynUnet(MAIN_DIRECTORY+basename+".nii.gz", model_path, TEMP_DIRECTORY, device)

    tmp_seg = nib.load(TEMP_DIRECTORY+basename+'_seg.nii.gz')
    seg_resampled = nib.processing.resample_to_output(tmp_seg, PIXDIM, order = 0,  mode='nearest')
    nib.save(seg_resampled, TEMP_DIRECTORY+basename+'_Resampled_seg.nii.gz')
    

    #print('Start of the ventricles segmentation: Totalsegmentator...')
    #totalsegmentator(MAIN_DIRECTORY+basename+".nii.gz", TEMP_DIRECTORY+basename+"_brain_structures", task="brain_structures")




def FLIRT_ANTS_registration(basename, hist_match_bool):

    brain_extraction_method = "TTS"

    # REGISTRATION
    print('Start of the linear registration...')
    Atlas = DATA_DIRECTORY+'Resliced_Registered_Labels_mod.nii.gz'
    Atlas_vasc = DATA_DIRECTORY+'ArterialAtlas.nii.gz'

    hist_match = ""

    if hist_match_bool:
        template = DATA_DIRECTORY+'hist_match_TEMPLATE_miplab-ncct_sym_brain.nii.gz'
        hist_match = "_hist_match"
    else:
        template = DATA_DIRECTORY+'TEMPLATE_miplab-ncct_sym_brain.nii.gz'
    
    flt = fsl.FLIRT()

    flt.inputs.in_file = template
    flt.inputs.reference = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+hist_match+'_skullstripped.nii.gz'
    flt.inputs.out_file = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_Template_FLIRT'+hist_match+'_Registered.nii.gz'
    flt.inputs.out_matrix_file = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_FLIRT'+hist_match+'_RegisteredTemplate_transform-matrix.mat'
    flt.inputs.dof = 7
    flt.inputs.bins = 256
    flt.inputs.cost_func = 'normcorr'
    flt.inputs.interp = 'nearestneighbour'
    flt.inputs.searchr_x = [-180, 180]
    flt.inputs.searchr_y = [-180, 180]
    flt.inputs.searchr_z = [-180, 180]
    flt.run()
    
    applyxfm = fsl.ApplyXFM()
    applyxfm.inputs.in_matrix_file = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_FLIRT'+hist_match+'_RegisteredTemplate_transform-matrix.mat'
    applyxfm.inputs.in_file = Atlas
    applyxfm.inputs.out_file = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_Atlas_FLIRT'+hist_match+'_Registered.nii.gz'
    applyxfm.inputs.reference = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+hist_match+'_skullstripped.nii.gz'
    applyxfm.inputs.apply_xfm = True
    applyxfm.inputs.out_matrix_file = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_FLIRT'+hist_match+'_RegisteredAtlas_transform-matrix.mat'
    applyxfm.inputs.interp = 'nearestneighbour'
    applyxfm.run()
            
    applyxfm = fsl.ApplyXFM()
    applyxfm.inputs.in_matrix_file = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_FLIRT'+hist_match+'_RegisteredTemplate_transform-matrix.mat'
    applyxfm.inputs.in_file = Atlas_vasc
    applyxfm.inputs.out_file = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_AtlasVasc_FLIRT'+hist_match+'_Registered.nii.gz'
    applyxfm.inputs.reference = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+hist_match+'_skullstripped.nii.gz'
    applyxfm.inputs.apply_xfm = True
    applyxfm.inputs.out_matrix_file = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_FLIRT'+hist_match+'_RegisteredAtlasVasc_transform-matrix.mat'
    applyxfm.inputs.interp = 'nearestneighbour'
    applyxfm.run()
    
    print('End of the linear registration')
    
    print('Start of the elastic registration...')
    img_fixed = ants.image_read(TEMP_DIRECTORY+basename+'_'+brain_extraction_method+hist_match+'_skullstripped.nii.gz')
    img_moving = ants.image_read(TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_Template_FLIRT'+hist_match+'_Registered.nii.gz')
    outprefix=TEMP_DIRECTORY+basename
    reg = ants.registration(img_fixed, img_moving, outprefix=outprefix, random_seed=42)
    reg['warpedmovout'].to_file(TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_Template_ANTS'+hist_match+'_Registered.nii.gz')
    
    mytx = reg['fwdtransforms']
    im_to_embarque = ants.image_read(TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_Atlas_FLIRT'+hist_match+'_Registered.nii.gz')
    embarqued_im = ants.apply_transforms(img_fixed, im_to_embarque, transformlist=mytx, interpolator='nearestNeighbor')
    embarqued_im.to_file(TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_Atlas_ANTS'+hist_match+'_Registered.nii.gz')
    
    im_to_embarque = ants.image_read(TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_AtlasVasc_FLIRT'+hist_match+'_Registered.nii.gz')
    embarqued_im = ants.apply_transforms(img_fixed, im_to_embarque, transformlist=mytx, interpolator='nearestNeighbor')
    embarqued_im.to_file(TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_AtlasVasc_ANTS'+hist_match+'_Registered.nii.gz')
    
    print('End of the elastic registration')


def volume_computation(basename):

    brain_extraction_method = "TTS"
    registration_method = "ANTS_hist_match"

    print(f"Volume inference regular atlas on: {basename}, brain extraction: {brain_extraction_method}, registration: {registration_method}")



    segfile = TEMP_DIRECTORY+basename+'_Resampled_seg.nii.gz'
    atlas = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_Atlas_'+registration_method+"_Registered.nii.gz"
    outcsv = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_'+registration_method+'_Volumes.csv'
    Single_Volume_Inference(atlas, segfile, DATA_DIRECTORY+'Labels_With_0.csv', outcsv)
    
    print(f"Volume inference vascular atlas on: {basename}, brain extraction: {brain_extraction_method}, registration: {registration_method}")

    atlas = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_AtlasVasc_'+registration_method+"_Registered.nii.gz"
    outcsv = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_'+registration_method+'_VolumesVasc.csv'
    Single_Volume_Inference(atlas, segfile, DATA_DIRECTORY+'Labels_With_0_vasc.csv', outcsv)



# ---------------------- NII file process ----------------------

def process_file(initial_file, device):

    basename = initial_file[:-7]


    opt={"Sform_code":'aligned', "Qform_code":'unknown'}

    #sometimes the image is broken and can't be loaded
    try:
        img = nib.load(MAIN_DIRECTORY+initial_file)
        img_data = img.get_fdata()
    except:
        return
    
    img_cleaned = nib.Nifti1Image(img_data, img.affine)
    sform_code = opt['Sform_code']
    qform_code = opt['Qform_code']	
    img_cleaned.set_sform(img_cleaned.get_sform(), code=sform_code)
    img_cleaned.set_qform(img_cleaned.get_qform(), code=qform_code)
    clean_file =  TEMP_DIRECTORY+basename+'_clean.nii.gz'
    nib.save(img_cleaned, clean_file)

    initial_img = nib.load(TEMP_DIRECTORY+basename+"_clean.nii.gz")
    pixdim = [1,1,1]
    img_resampled = nib.processing.resample_to_output(initial_img, pixdim, order=0)
    nib.save(img_resampled, TEMP_DIRECTORY+basename+"_resampled.nii.gz")


    brain_extraction(basename, device)
    
    segmentation(basename, device)
    

    try:
        reference = nib.load(TEMP_DIRECTORY+basename+'_TTS_skullstripped.nii.gz')
    except:
        print("FILE LOADING ERROR ")

    reference_data = reference.get_fdata()

    # values below low_threshold are set to 0 and values over high_threshold are set to high_threshold. Important otherwise the histogram matching won't perform correctly
    reference_data = clamp_values(reference_data, CLAMP_LOW_THRESHOLD, CLAMP_HIGH_THRESHOLD)
    reference_data = reference_data/np.max(reference_data) # normalize values between 0 and 1
    reference_data = histogram_matching(reference_data, nib.load(DATA_DIRECTORY+'hist_match_TEMPLATE_miplab-ncct_sym_brain.nii.gz').get_fdata())
    new_reference_data = nib.Nifti1Image(reference_data, reference.affine, reference.header)
    nib.save(new_reference_data, TEMP_DIRECTORY+basename+'_TTS_hist_match'+'_skullstripped.nii.gz')


    FLIRT_ANTS_registration(basename, hist_match_bool=True)


    volume_computation(basename)

    # move all the temp files to a temp subdirectory with the patient number.
    try:
        os.mkdir(TEMP_DIRECTORY+basename)
    except:
        pass

    for file in os.listdir(TEMP_DIRECTORY):

        if not os.path.isdir(TEMP_DIRECTORY+file):
            os.rename(TEMP_DIRECTORY+file, TEMP_DIRECTORY+basename+'/'+file)


# ---------------------- Main Process ----------------------

def run():

    if torch.cuda.is_available() and torch.cuda.device_count()>0:
        device = f'cuda:{GPU_DEVICE}'
        #device = torch.cuda.current_device()
        print('Segmentation will run on GPU: ID='+str(device)+', NAME: '+torch.cuda.get_device_name(device))
    else:
        device = 'cpu'
        print('Segmentation will run on CPU')


    # Make clamped and normalized Template source file
    template = nib.load(DATA_DIRECTORY+'TEMPLATE_miplab-ncct_sym_brain.nii.gz')
    template_data = template.get_fdata()
    template_data = clamp_values(template_data, CLAMP_LOW_THRESHOLD, CLAMP_HIGH_THRESHOLD)
    template_data = template_data/np.max(template_data)
    new_template = nib.Nifti1Image(template_data, template.affine, template.header)
    nib.save(new_template, DATA_DIRECTORY+'hist_match_TEMPLATE_miplab-ncct_sym_brain.nii.gz')


    for initial_file in os.listdir(MAIN_DIRECTORY):

        if initial_file.endswith(".nii.gz") and not os.path.isdir(TEMP_DIRECTORY+initial_file[:-7]): # prevents from working on file that has already been processed
            
            print("---------------------")
            print(f"processing file {initial_file}")

            process_file(initial_file, device=device)


if __name__=="__main__":
    run()




