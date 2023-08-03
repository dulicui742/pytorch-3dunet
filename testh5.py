import h5py
import os
import SimpleITK as sitk
import numpy as np


def read_dicom(dicom_path):
    # print("Start trans ", dicom_path)
    series_reader = sitk.ImageSeriesReader()
    dicom_names = series_reader.GetGDCMSeriesFileNames(dicom_path)
    series_reader.SetFileNames(dicom_names)
    dicom_img = series_reader.Execute()
    return dicom_img


def resample_sitk_1(src_img, interpolator=sitk.sitkLinear, dest_size=[64, 64, 64]):
    src_size = src_img.GetSize()
    src_spacing = src_img.GetSpacing()
    # dest_size = np.array(src_size) * np.array(src_spacing) / np.array(dest_spacing)
    # dest_size = np.round(dest_size).astype(np.int64).tolist()
    # print(f"=======src size: {src_size}, dest size: {dest_size}")
    dest_spacing = np.array(src_size) * np.array(src_spacing) / np.array(dest_size)

    return sitk.Resample(src_img, dest_size, sitk.Transform(), interpolator,
                        src_img.GetOrigin(), dest_spacing, src_img.GetDirection(),
                        0, src_img.GetPixelID())


# uid_file = "/mnt/Data/data03/dst/450_176/train.txt"
# base_path = r"D:\project\TrueHealth\20230217_Alg1\data\examples\Sphere_train"


base_path = "/mnt/Data/data03/dst/"
train_uid_file = "mnt/Data/data03/dst/450_176/train.txt"
val_uid_file = "/mnt/Data/data03/dst/450_176/valid.txt"

save_base_path = "/mnt/Data/data03_h5/"


# uid_files = [val_uid_file]
uid_files = [train_uid_file]
for uid_file in uid_files:
    if "train" in uid_file:
        save_path = os.path.join(save_base_path, "train")
    elif "val" in uid_file:
        save_path = os.path.join(save_base_path, "valid")
    
    uids = []   
    with open(uid_file) as f:
        uids = [x.strip() for x in f.readlines()]
    print(f"uid num: {len(uids)}")

    # uids = os.listdir(os.path.join(base_path, "dicom"))
    for idx, uid in enumerate(uids):
        print(f"num {idx}: {uid}")
        img_path = os.path.join(base_path, "dicom", uid)
        img_itk = read_dicom(img_path)
        img = sitk.GetArrayFromImage(img_itk)


        mask_path = os.path.join(base_path, "mask", "Sphere", uid)
        mask_itk = read_dicom(mask_path)
        mask = sitk.GetArrayFromImage(mask_itk)

        # import pdb; pdb.set_trace()
        # ### save nii to ensure resample operation is ok or not
        # img_itk_resample = resample_sitk_1(img_itk)
        # mask_itk_resample = resample_sitk_1(mask_itk)
        # sitk.WriteImage(sitk.Cast(img_itk_resample, sitk.sitkInt16), os.path.join("output", "image", f"{uid}_image.nii"))
        # sitk.WriteImage(sitk.Cast(mask_itk_resample, sitk.sitkInt16), os.path.join("output", "image", f"{uid}_mask.nii"))

        h5_file = os.path.join(save_path, f'{uid}.h5')
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset("raw", data=img)
            f.create_dataset("label", data=mask)




