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

import demeter.metamorphosis as mt
from demeter.utils.toolbox import update_progress,format_time
import demeter.utils.reproducing_kernels as rk
from demeter.utils.constants import *
import demeter.utils.torchbox as tb

# -------------------------------------------------------

# Process 0 GPU device 0 

# ---------------------- CONSTANTS ----------------------

MAIN_DIRECTORY = "/home/fehrdelt/data_ssd/data/FastDiag_2_0/"
TEMP_DIRECTORY = "/home/fehrdelt/data_ssd/data/mega_CT_TIQUA_temp_2_0/"
DATA_DIRECTORY = "/home/fehrdelt/data_ssd/MedicalImaging_GIN/mega_CT_TIQUA/data_2_0/"

MATLAB_APP_PATH = '/data_network/irmage_pa/_SHARE/DOCKER_CT-TIQUA/docker_light_test/compiled_matlab_scripts/App/application/run_SkullStrip.sh'
MATLAB_RUNTIME_PATH = '/data_network/irmage_pa/_SHARE/DOCKER_CT-TIQUA/docker_light_test/compiled_matlab_scripts/RunTime/v910'

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
    
    # TTS (totalsegmentator)
    start = time.time()
    totalsegmentator(TEMP_DIRECTORY+basename+"_resampled.nii.gz", TEMP_DIRECTORY, roi_subset=["brain"], device=f"gpu:{GPU_DEVICE}")
    brain_mask = nib.load(TEMP_DIRECTORY+"brain.nii.gz")
    original_img = nib.load(TEMP_DIRECTORY+basename+"_resampled.nii.gz")
    TTS_extracted_brain = nib.Nifti1Image(brain_mask.get_fdata()*original_img.get_fdata(), original_img.affine, original_img.header)
    nib.save(TTS_extracted_brain, TEMP_DIRECTORY+basename+"_TTS_skullstripped.nii.gz")
    end = time.time()
    tts_time = end-start

    # matlab 
    start = time.time()
    img_resampled = nib.load(TEMP_DIRECTORY+basename+"_resampled.nii.gz")
    nib.save(img_resampled, TEMP_DIRECTORY+basename+"_resampled.nii") # need the same input file but in nii format for the matlab algorithm
    output_path = TEMP_DIRECTORY+basename+"_matlab_skullstripped.nii" # actually this will save as nii.gz with the matlab aglorithm idk why
    output_ROI = TEMP_DIRECTORY+basename+"_matlab_ROI.nii"            # same here
    os.system(MATLAB_APP_PATH+' '+MATLAB_RUNTIME_PATH+' '+TEMP_DIRECTORY+basename+"_resampled.nii"+' '+output_path+' '+output_ROI)
    end = time.time()
    matlab_time = end-start

    # custom_nn
    start = time.time()
    apply_brain_extraction_net(TEMP_DIRECTORY+basename+"_resampled.nii.gz", DATA_DIRECTORY+"custom_nn_brain_extraction.pt", TEMP_DIRECTORY, device)
    extracted_mask = nib.load(TEMP_DIRECTORY+basename+"_resampled_custom_nn_brain_mask.nii.gz")
    custom_nn_extracted_brain = nib.Nifti1Image(original_img.get_fdata()*extracted_mask.get_fdata(), original_img.affine, original_img.header)
    nib.save(custom_nn_extracted_brain, TEMP_DIRECTORY+basename+"_custom_nn_skullstripped.nii.gz")
    end = time.time()
    custom_nn_time = end-start

    return tts_time, matlab_time, custom_nn_time

def segmentation(basename, device):
    
    print('Start of the segmentation: DynUnet on raw images (xxmm3)...')
    
    start = time.time()
    
    model_path = DATA_DIRECTORY+"24-02-23-13h18m_best_model.pt"

    # Uncomment this code if you want to apply DynUnet on resampled images
    # ApplyDynUnet(resampled_file, model_path, outfolder_seg, device)
    # here we choose to apply DynUnet on raw images
    
    ApplyDynUnet(MAIN_DIRECTORY+basename+".nii.gz", model_path, TEMP_DIRECTORY, device)

    tmp_seg = nib.load(TEMP_DIRECTORY+basename+'_seg.nii.gz')
    seg_resampled = nib.processing.resample_to_output(tmp_seg, PIXDIM, order = 0,  mode='nearest')
    nib.save(seg_resampled, TEMP_DIRECTORY+basename+'_Resampled_seg.nii.gz')
    
    end = time.time()

    print('End of the segmentation resampled to 1mm3: DynUnet')

    return end-start


def registration(basename, brain_extraction_method, device):
    
    # FLIRT+ANTS without prior histogram matching
    start = time.time()
    FLIRT_ANTS_registration(basename, brain_extraction_method, hist_match_bool=False)
    end = time.time()
    ants_time = end-start
    
    # FLIRT+ANTS with prior histogram matching
    start = time.time()
    FLIRT_ANTS_registration(basename, brain_extraction_method, hist_match_bool=True)
    end = time.time()
    ants_hist_match_time = end-start

    start = time.time()
    LDDMM_registration(basename, brain_extraction_method, device)
    end = time.time()
    lddmm_time = end-start

    return ants_time, ants_hist_match_time, lddmm_time


def FLIRT_ANTS_registration(basename, brain_extraction_method, hist_match_bool):
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


def LDDMM_registration(basename, brain_extraction_method, device):

    source_name = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_Template_FLIRT'+'_hist_match'+'_Registered.nii.gz'
    target_name = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_hist_match'+'_skullstripped.nii.gz'

    load_image = Compose([LoadImage(image_only=True, ensure_channel_first=True)])
    load_resize = Compose([LoadImage(image_only=True, ensure_channel_first=True), Resize(nib.load(source_name).get_fdata().shape)])
    S_tmp = load_image(source_name)[0, :, :, :]
    T_tmp = load_resize(target_name)[0, :, :, :]
    #TODO : REVERIFIER : C'EST PAS PLUTOT CROP_OR_PAD ??
    

    
    #T_tmp[T_tmp < CLAMP_LOW_THRESHOLD] = 0
    #T_tmp[T_tmp > CLAMP_HIGH_THRESHOLD] = CLAMP_HIGH_THRESHOLD

    #S_tmp[S_tmp < CLAMP_LOW_THRESHOLD] = 0
    #S_tmp[S_tmp > CLAMP_HIGH_THRESHOLD] = CLAMP_HIGH_THRESHOLD

    print("Shapes : ")
    print(S_tmp.shape)
    print(T_tmp.shape)

    sig = 2
    smooth = rk.GaussianRKHS((sig,sig,sig))

    T_smooth = smooth(
        torch.Tensor(T_tmp)[None, None]
    ).numpy()[0, 0].astype(np.float64)
    T_smooth[T_tmp == 0] = 0

    T_tmp = T_smooth
    

    T_tmp = (T_tmp / T_tmp.max() )
    S_tmp = (S_tmp / S_tmp.max() )

    #T_tmp = histogram_matching(T_tmp, S_tmp)
    
    S = np.zeros((1, 1, S_tmp.shape[0], S_tmp.shape[1], S_tmp.shape[2]))
    S[0, :, :, :, :] = S_tmp
    T = np.zeros((1, 1, T_tmp.shape[0], T_tmp.shape[1], T_tmp.shape[2]))
    T[0, :, :, :, :] = T_tmp

    S = torch.from_numpy(S)
    S.to(device)
    S = S.cuda().to(torch.float).to(device) # <---------------------------------------------------------------------------------------------------- ICI


    T = torch.from_numpy(T)
    T.to(device)
    T = T.cuda().to(torch.float).to(device) # <---------------------------------------------------------------------------------------------------- ICI
    print(device)


    
    # if the image is too big for your GPU, you can downsample it quite barbarically :
    step = 2 # TODO: Verify if this fits for GPU memory of if need to downsample
    if step > 0:
        S = S[:,:,::step,::step,::step]
        T = T[:,:,::step,::step,::step]


    _, _,D,H,W = S.shape
    residuals = 0

    mu = 0
    mu,rho,lamb = 0, 0, .0001   # LDDMM

    print("Apply LDDMM")
    mr_lddmm = mt.lddmm(S,T,residuals,
        sigma=[2,6],          #  Kernel size
        #sigma=(4,4,4),          #  Kernel size
        cost_cst=0.01,         # Regularization parameter
        integration_steps=10,   # Number of integration steps
        n_iter=600,             # Number of optimization steps
        grad_coef=10,            # max Gradient coefficient
        data_term=None,         # Data term (default Ssd)
        sharp=False,            # Sharp integration toggle
        safe_mode = False,      # Safe mode toggle (does not crash when nan values are encountered)
        integration_method='semiLagrangian',  # You should not use Eulerian for real usage
    )
    #mr_lddmm.plot_cost()

    name = source_name
    

    deformator = mr_lddmm.mp.get_deformator()
    
    img_deform = tb.imgDeform(S.cpu(), deformator, dx_convention="pixel", clamp=True)
    
    orig = nib.load(source_name)
    empty_header = nib.Nifti1Header()
    registered_lddmm_image = nib.Nifti1Image(img_deform.cpu().numpy()[0,0,:,:,:], orig.affine, empty_header)
    nib.save(registered_lddmm_image, TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_Template_LDDMM_Registered.nii.gz')

    # apply the deformation to the two atlases
    atlas = load_resize(TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_Atlas_FLIRT_hist_match_Registered.nii.gz')[0, :, :, :]
    #atlas = nib.load(TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_hist_match'+'_Atlas_FLIRTRegistered.nii.gz')
    atlasVasc = load_resize(TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_AtlasVasc_FLIRT_hist_match_Registered.nii.gz')[0, :, :, :]
    #atlasVasc = nib.load(TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_hist_match'+'_AtlasVasc_FLIRTRegistered.nii.gz')

    # downsample the atlas, then apply the deformation (which was computed on downsampled images)
    
    atlas_temp = np.zeros((1, 1, S_tmp.shape[0], S_tmp.shape[1], S_tmp.shape[2]))
    atlas_temp[0, 0, :, :, :] = atlas
    atlas_temp = torch.from_numpy(atlas_temp)
    atlas_temp.to(device)
    atlas_temp = atlas_temp.cuda().to(torch.float)

    atlasVasc_temp = np.zeros((1, 1, S_tmp.shape[0], S_tmp.shape[1], S_tmp.shape[2]))
    atlasVasc_temp[0, 0, :, :, :] = atlasVasc
    atlasVasc_temp = torch.from_numpy(atlasVasc_temp)
    atlasVasc_temp.to(device)
    atlasVasc_temp = atlasVasc_temp.cuda().to(torch.float)

    atlas_deform = tb.imgDeform_nearest(atlas_temp.cpu(), deformator, dx_convention="pixel", clamp=True).numpy()[0, 0, :, :, :]
    atlasVasc_deform = tb.imgDeform_nearest(atlasVasc_temp.cpu(), deformator, dx_convention="pixel", clamp=True).numpy()[0, 0, :, :, :]
    
    atlas_nib = nib.load(TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_Atlas_FLIRT_hist_match_Registered.nii.gz')
    atlasVasc_nib = nib.load(TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_AtlasVasc_FLIRT_hist_match_Registered.nii.gz')


    # then upscale the atlases ( !! with nearest interpolation because it's a labelmap !! )
    atlas_deform = scipy.ndimage.zoom(atlas_deform, step, order=0)
    atlasVasc_deform = scipy.ndimage.zoom(atlasVasc_deform, step, order=0)

    atlas_lddmm_registered = nib.Nifti1Image(atlas_deform, atlas_nib.affine, atlas_nib.header)
    atlasVasc_lddmm_registered = nib.Nifti1Image(atlasVasc_deform, atlasVasc_nib.affine, atlasVasc_nib.header)

    nib.save(atlas_lddmm_registered, TEMP_DIRECTORY+basename+'_'+brain_extraction_method+"_Atlas_LDDMM_Registered.nii.gz")
    nib.save(atlasVasc_lddmm_registered, TEMP_DIRECTORY+basename+'_'+brain_extraction_method+"_AtlasVasc_LDDMM_Registered.nii.gz")


def volume_computation(basename, brain_extraction_method, registration_method):

    print(f"Volume inference regular atlas on: {basename}, brain extraction: {brain_extraction_method}, registration: {registration_method}")

    start = time.time()

    segfile = TEMP_DIRECTORY+basename+'_Resampled_seg.nii.gz'
    atlas = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_Atlas_'+registration_method+"_Registered.nii.gz"
    outcsv = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_'+registration_method+'_Volumes.csv'
    Single_Volume_Inference(atlas, segfile, DATA_DIRECTORY+'Labels_With_0.csv', outcsv)
    
    print(f"Volume inference vascular atlas on: {basename}, brain extraction: {brain_extraction_method}, registration: {registration_method}")

    atlas = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_AtlasVasc_'+registration_method+"_Registered.nii.gz"
    outcsv = TEMP_DIRECTORY+basename+'_'+brain_extraction_method+'_'+registration_method+'_VolumesVasc.csv'
    Single_Volume_Inference(atlas, segfile, DATA_DIRECTORY+'Labels_With_0_vasc.csv', outcsv)
    end = time.time()
    return end-start

# ---------------------- NII file process ----------------------

def process_file(initial_file, device, durations_df):

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


    brain_extraction_tts_time, brain_extraction_matlab_time, brain_extraction_custom_nn_time = brain_extraction(basename, device)
    
    segmentation_time = segmentation(basename, device)
    
    extraction_methods = ["matlab", "TTS", "custom_nn"]
    registration_methods = ["ANTS", "ANTS_hist_match", "LDDMM"]

    for extraction_method in extraction_methods:
        
        start = time.time()

        try:
            reference = nib.load(TEMP_DIRECTORY+basename+'_'+extraction_method+'_skullstripped.nii.gz')
        except:
            print("FILE LOADING ERROR "+extraction_method)
            broke_quantification_df = pd.DataFrame(columns=["Patient"])
            broke_quantification_df.loc[len(broke_quantification_df)] = basename
            broke_quantification_df.to_csv(TEMP_DIRECTORY+basename+"_broke.csv")
            break
        reference_data = reference.get_fdata()

        # values below low_threshold are set to 0 and values over high_threshold are set to high_threshold. Important otherwise the histogram matching won't perform correctly
        reference_data = clamp_values(reference_data, CLAMP_LOW_THRESHOLD, CLAMP_HIGH_THRESHOLD)
        reference_data = reference_data/np.max(reference_data) # normalize values between 0 and 1
        reference_data = histogram_matching(reference_data, nib.load(DATA_DIRECTORY+'hist_match_TEMPLATE_miplab-ncct_sym_brain.nii.gz').get_fdata())
        new_reference_data = nib.Nifti1Image(reference_data, reference.affine, reference.header)
        nib.save(new_reference_data, TEMP_DIRECTORY+basename+'_'+extraction_method+'_hist_match'+'_skullstripped.nii.gz')

        end = time.time()
        histogram_match_time = end-start

        registration_ANTS_time, registration_ANTS_hist_match_time, registration_LDDMM_time = registration(basename, extraction_method, device)

        for registration_method in registration_methods:
            volume_computation_time = volume_computation(basename, extraction_method, registration_method)

    try:
        durations_df.loc[len(durations_df)] = [basename, brain_extraction_tts_time, brain_extraction_matlab_time, brain_extraction_custom_nn_time, registration_ANTS_time, registration_ANTS_hist_match_time, registration_LDDMM_time, histogram_match_time, volume_computation_time, segmentation_time]
    except:
        pass

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

    durations_df = pd.DataFrame(columns=["name", "TTS_extraction", "matlab_extraction", "custom_nn_extraction", "ANTS_registration", "ANTS_hist_match_registration", "LDDMM_registration", "histogram_matching", "volume_computation", "segmentation"])

    for initial_file in os.listdir(MAIN_DIRECTORY):

        if initial_file.endswith(".nii.gz") and not os.path.isdir(TEMP_DIRECTORY+initial_file[:-7]): # prevents from working on file that has already been processed
            
            print("---------------------")
            print(f"processing file {initial_file}")

            process_file(initial_file, device=device, durations_df=durations_df)

            durations_df.to_csv(TEMP_DIRECTORY+"durations.csv")


if __name__=="__main__":
    run()




