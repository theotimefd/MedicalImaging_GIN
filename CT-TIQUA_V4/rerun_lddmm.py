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


MAIN_DIRECTORY = "/home/fehrdelt/data_ssd/data/FastDiag/"
TEMP_DIRECTORY = "/home/fehrdelt/data_ssd/data/mega_CT_TIQUA_temp/"
DATA_DIRECTORY = "/home/fehrdelt/data_ssd/MedicalImaging_GIN/mega_CT_TIQUA/data/"

MATLAB_APP_PATH = '/data_network/irmage_pa/_SHARE/DOCKER_CT-TIQUA/docker_light_test/compiled_matlab_scripts/App/application/run_SkullStrip.sh'
MATLAB_RUNTIME_PATH = '/data_network/irmage_pa/_SHARE/DOCKER_CT-TIQUA/docker_light_test/compiled_matlab_scripts/RunTime/v910'

CLAMP_LOW_THRESHOLD = 1.0
CLAMP_HIGH_THRESHOLD = 80

GPU_DEVICE = 0

PIXDIM = [1,1,1]



def LDDMM_registration(basename, brain_extraction_method, device, directory):

    source_name = directory+basename+'_'+brain_extraction_method+'_Template_FLIRT'+'_hist_match'+'_Registered.nii.gz'
    target_name = directory+basename+'_'+brain_extraction_method+'_hist_match'+'_skullstripped.nii.gz'

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
    
    deformation = mr_lddmm.mp.get_deformation()
    deformator = mr_lddmm.mp.get_deformator()
    
    img_deform = tb.imgDeform(S.cpu(), deformator, dx_convention="pixel", clamp=True)
    
    orig = nib.load(source_name)
    empty_header = nib.Nifti1Header()
    registered_lddmm_image = nib.Nifti1Image(img_deform.cpu().numpy()[0,0,:,:,:], orig.affine, empty_header)
    nib.save(registered_lddmm_image, directory+basename+'_'+brain_extraction_method+'_Template_LDDMM_Registered.nii.gz')

    # apply the deformation to the two atlases
    atlas = load_resize(directory+basename+'_'+brain_extraction_method+'_Atlas_FLIRT_hist_match_Registered.nii.gz')[0, :, :, :]
    #atlas = nib.load(directory+basename+'_'+brain_extraction_method+'_hist_match'+'_Atlas_FLIRTRegistered.nii.gz')
    atlasVasc = load_resize(directory+basename+'_'+brain_extraction_method+'_AtlasVasc_FLIRT_hist_match_Registered.nii.gz')[0, :, :, :]
    #atlasVasc = nib.load(directory+basename+'_'+brain_extraction_method+'_hist_match'+'_AtlasVasc_FLIRTRegistered.nii.gz')

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
    
    atlas_nib = nib.load(directory+basename+'_'+brain_extraction_method+'_Atlas_FLIRT_hist_match_Registered.nii.gz')
    atlasVasc_nib = nib.load(directory+basename+'_'+brain_extraction_method+'_AtlasVasc_FLIRT_hist_match_Registered.nii.gz')



    # then upscale the atlases ( !! with nearest interpolation because it's a labelmap !! )
    atlas_deform = scipy.ndimage.zoom(atlas_deform, step, order=0)
    atlasVasc_deform = scipy.ndimage.zoom(atlasVasc_deform, step, order=0)

    atlas_lddmm_registered = nib.Nifti1Image(atlas_deform, atlas_nib.affine, atlas_nib.header)
    atlasVasc_lddmm_registered = nib.Nifti1Image(atlasVasc_deform, atlasVasc_nib.affine, atlasVasc_nib.header)

    nib.save(atlas_lddmm_registered, directory+basename+'_'+brain_extraction_method+"_Atlas_LDDMM_Registered.nii.gz")
    nib.save(atlasVasc_lddmm_registered, directory+basename+'_'+brain_extraction_method+"_AtlasVasc_LDDMM_Registered.nii.gz")


def volume_computation(basename, brain_extraction_method, registration_method, directory):

    print(f"Volume inference regular atlas on: {basename}, brain extraction: {brain_extraction_method}, registration: {registration_method}")

    start = time.time()

    segfile = directory+basename+'_Resampled_seg.nii.gz'
    atlas = directory+basename+'_'+brain_extraction_method+'_Atlas_'+registration_method+"_Registered.nii.gz"
    outcsv = directory+basename+'_'+brain_extraction_method+'_'+registration_method+'_Volumes.csv'
    Single_Volume_Inference(atlas, segfile, DATA_DIRECTORY+'Labels_With_0.csv', outcsv)
    
    print(f"Volume inference vascular atlas on: {basename}, brain extraction: {brain_extraction_method}, registration: {registration_method}")

    atlas = directory+basename+'_'+brain_extraction_method+'_AtlasVasc_'+registration_method+"_Registered.nii.gz"
    outcsv = directory+basename+'_'+brain_extraction_method+'_'+registration_method+'_VolumesVasc.csv'
    Single_Volume_Inference(atlas, segfile, DATA_DIRECTORY+'Labels_With_0_vasc.csv', outcsv)
    end = time.time()
    return end-start

def run():

    if torch.cuda.is_available() and torch.cuda.device_count()>0:
        device = f'cuda:{GPU_DEVICE}'
        #device = torch.cuda.current_device()
        print('Segmentation will run on GPU: ID='+str(device)+', NAME: '+torch.cuda.get_device_name(device))
    else:
        device = 'cpu'
        print('Segmentation will run on CPU')

    durations_df = pd.DataFrame(columns=["lddmm_duration"])

    for dir in os.listdir("/home/fehrdelt/data_ssd/data/mega_CT_TIQUA_temp/"):

        if os.path.isdir("/home/fehrdelt/data_ssd/data/mega_CT_TIQUA_temp/"+dir):
            if dir[:2] == "P0" and len(dir)==5:
                if (int(dir[1:]) > 166 and int(dir[1:])<185) or dir in ["P0017", "P0041", "P0061"]:
                    
                    print(dir)

                    directory = "/home/fehrdelt/data_ssd/data/mega_CT_TIQUA_temp/"+dir+"/"
                
                    extraction_methods = ["TTS", "matlab", "custom_nn"]
                    registration_methods = ["ANTS", "ANTS_hist_match", "LDDMM"]

                    for extraction_method in extraction_methods:
                        duration = LDDMM_registration(dir, extraction_method, device, directory)
                        durations_df.loc[len(durations_df)] = [duration]

                        for registration_method in registration_methods:
                            volume_computation(dir, extraction_method, registration_method, directory)
            
            
    
    durations_df.to_csv("/home/fehrdelt/data_ssd/data/mega_CT_TIQUA_temp/lddmm_durations.csv")




if __name__=="__main__":
    run()