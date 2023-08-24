import h5py
import os
import SimpleITK as sitk
import numpy as np
import vtk
from vtk.util import numpy_support


def read_dicom(dicom_path):
    # print("Start trans ", dicom_path)
    series_reader = sitk.ImageSeriesReader()
    dicom_names = series_reader.GetGDCMSeriesFileNames(dicom_path)
    series_reader.SetFileNames(dicom_names)
    dicom_img = series_reader.Execute()
    return dicom_img


def read_dicom_vtk(dicom_path):
    dicomreader = vtk.vtkDICOMImageReader()
    dicomreader.SetDirectoryName(dicom_path)
    dicomreader.Update()
    output = dicomreader.GetOutput()
    dimensions = output.GetDimensions()
    # print("dimension:", dimensions)
    dicomArray = numpy_support.vtk_to_numpy(output.GetPointData().GetScalars())
    dicomArray = dicomArray.reshape(dimensions[::-1]).astype(np.float32)
    dicomArray = dicomArray[:,::-1,:]
    return dicomArray


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



def resample_sitk(
    src_img, 
    interpolator=sitk.sitkLinear, 
    dest_spacing=[0.6796875, 0.6796875, 1]   
):
    src_size = src_img.GetSize()
    src_spacing = src_img.GetSpacing()
    dest_size = np.array(src_size) * np.array(src_spacing) / np.array(dest_spacing)
    dest_size = np.round(dest_size).astype(np.int64).tolist()
    dest_size = [512, 512, dest_size[2]]
    print(f"=======src size: {src_size}, dest size: {dest_size}")

    return sitk.Resample(src_img, dest_size, sitk.Transform(), interpolator,
                        src_img.GetOrigin(), dest_spacing, src_img.GetDirection(),
                        0, src_img.GetPixelID())

# uid_file = "/mnt/Data/data03/dst/450_176/train.txt"
# base_path = r"D:\project\TrueHealth\20230217_Alg1\data\examples\Sphere_train"


# base_path = "/mnt/Data/data03/dst/"
# train_uid_file = "/mnt/Data/data03/dst/450_176/train.txt"
# val_uid_file = "/mnt/Data/data03/dst/450_176/valid.txt"
# save_base_path = "/data/dulicui/project/data/data03_h5/"
# phase = "test"
labels = ["Sphere"]

base_path = "/mnt/Data/data00/dst/"
train_uid_file = "/mnt/Data/data00/dst/500_100/train.txt"
val_uid_file = "/mnt/Data/data00/dst/500_100/valid.txt"
save_base_path = "/data/dulicui/project/data/data00_h5/"
phase = "train"
labels = ["Bronchial", "PulmonaryVessels", "TotalPulmonaryVessels"]


if phase != "test":
    uid_files = [val_uid_file]
    # uid_files = [train_uid_file]
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
        

            # import pdb; pdb.set_trace()
            # ### save nii to ensure resample operation is ok or not
            # img_itk_resample = resample_sitk_1(img_itk)
            # mask_itk_resample = resample_sitk_1(mask_itk)
            # sitk.WriteImage(sitk.Cast(img_itk_resample, sitk.sitkInt16), os.path.join("output", "image", f"{uid}_image.nii"))
            # sitk.WriteImage(sitk.Cast(mask_itk_resample, sitk.sitkInt16), os.path.join("output", "image", f"{uid}_mask.nii"))

            h5_file = os.path.join(save_path, f'{uid}.h5')
            with h5py.File(h5_file, 'w') as f:

                img_path = os.path.join(base_path, "dicom", uid)
                img_itk = read_dicom(img_path)
                img = sitk.GetArrayFromImage(img_itk)
                f.create_dataset("raw", data=img)

                for label in labels:
                    mask_path = os.path.join(base_path, "mask", label, uid)
                    mask_itk = read_dicom(mask_path)
                    mask = sitk.GetArrayFromImage(mask_itk)
                    f.create_dataset(label, data=mask)
        # import pdb; pdb.set_trace()

    
else:
    # ###=================for test dataset   sitk===========================
    # base_path = "/home/th/Data/data_sphere_test/"
    # # uids = ["30", "46"]
    # uids = [str(i) for i in range(0, 101)]

    # save_path = os.path.join(save_base_path, "test_debug")
    # for idx, uid in enumerate(uids):
    #     # if uid not in ["1", "14", "25", "35", "45", "56", "57", "58", "59", "63", "64", "66", "91", 
    #     #                "64", "55", "61", "62", "65", "22", "39", "60", "5", "44", "50", "7", "38", 
    #     #                "41", "46", "49", "48", "37", "70", "43", "36", "42"]:
    #     if uid not in ["53"]: #["40", "54"]:
    #         continue
    #     print(f"num {idx}: {uid}")
    #     img_path = os.path.join(base_path, uid, "dicom")
    #     img_itk = read_dicom(img_path)

    #     # img_itk = resample_sitk(img_itk)
    #     img = sitk.GetArrayFromImage(img_itk)
    #     img[img == -3024] = -1024


    #     import pdb; pdb.set_trace()
    #     h5_file = os.path.join(save_path, f'{uid}.h5')
    #     with h5py.File(h5_file, 'w') as f:
    #         f.create_dataset("raw", data=img)
    #         # spacing = img_itk.GetSpacing()
    #         # spacing = np.array(spacing)
    #         # f.create_dataset("spacing", spacing)
    # ###=================for test dataset   sitk===========================

    ###=================for test dataset   vtk===========================
    base_path = "/mnt/Data/data_sphere_test/"
    # uids = ["30", "46"]
    uids = [str(i) for i in range(1, 101)]

    save_path = os.path.join(save_base_path, "test_vtk")
    for idx, uid in enumerate(uids):
        # if uid not in ["1", "14", "25", "35", "45", "56", "57", "58", "59", "63", "64", "66", "91", 
        #                "64", "55", "61", "62", "65", "22", "39", "60", "5", "44", "50", "7", "38", 
        #                "41", "46", "49", "48", "37", "70", "43", "36", "42"]:
        # if uid not in ["53"]: #["40", "54"]:
        #     continue
        print(f"num {idx}: {uid}")
        img_path = os.path.join(base_path, uid, "dicom")
        # img_itk = read_dicom(img_path)
        # img = sitk.GetArrayFromImage(img_itk)

        img = read_dicom_vtk(img_path)
        img[img == -3024] = -1024


        # import pdb; pdb.set_trace()
        h5_file = os.path.join(save_path, f'{uid}.h5')
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset("raw", data=img)
    ###=================for test dataset   vtk===========================


