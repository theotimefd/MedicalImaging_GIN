
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, 
    LoadImaged,
    Orientationd,
    EnsureTyped,
    Spacingd,
    AsDiscreted,
    SaveImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    GaussianSmoothd,
    ScaleIntensityd,
    Activationsd,
    KeepLargestConnectedComponentd,
)

from monai.networks.nets import SegResNet

from monai.data import (
    Dataset, DataLoader,
    decollate_batch,
)


import torch




def apply_brain_extraction_net(infile, model_path, outfolder, device):
    print(device)

    labels = {
        "brain": 1,
    }
    target_spacing=(1.0, 1.0, 1.0)

    test_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureTyped(keys="image", device=device),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=target_spacing, allow_missing_keys=True),
            NormalizeIntensityd(keys="image", nonzero=True),
            GaussianSmoothd(keys="image", sigma=0.4),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
        ]
    )

    post_transforms = Compose(
        [
            EnsureTyped(keys="image", device=device),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            KeepLargestConnectedComponentd(keys="pred"),
            SaveImaged(keys="pred",
                       meta_keys="pred_meta_dict",
                       output_dir=outfolder,
                       output_postfix="custom_nn_brain_mask",
                       separate_folder = False,
                       resample=False),
        ]
    )


    model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=len(labels) + 1,  # labels plus background,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.2,
        ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))


    test_dict_ds = [{"image": image_name} for image_name in zip([infile])]
    test_ds = Dataset(data=test_dict_ds, transform=test_transforms)

    test_loader = DataLoader(test_ds, batch_size=1)# num_workers=4)


    model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            test_inputs = test_data["image"].to(device)
            roi_size = (96, 96, 96)
            sw_batch_size = 16
            test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model, mode="gaussian", overlap=0.8)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
