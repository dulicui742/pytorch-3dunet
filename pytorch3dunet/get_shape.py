import SimpleITK as sitk
import os
import pydicom
import pandas as pd
import json



def read_dicom(dicom_path):
    series_reader = sitk.ImageSeriesReader()
    dicom_names = series_reader.GetGDCMSeriesFileNames(dicom_path)
    series_reader.SetFileNames(dicom_names)
    dicom_img = series_reader.Execute()
    return dicom_img


if __name__ == "__main__":
    train_uid_file = "/home/th/Data/data03/dst/450_176/train.txt"
    val_uid_file = "/home/th/Data/data03/dst/450_176/valid.txt"
    base_path = "/home/th/Data/data03/dst"
    save_base_path = "/home/th/Data/data03_h5/json"


    name = base_path.split("/")[-2]
    meta_shape = {}
    uid_files = [train_uid_file, val_uid_file]
    for uid_file in uid_files:
        if "train" in uid_file:
            # save_path = os.path.join(save_base_path, "train")
            phase = "train"
        elif "val" in uid_file:
            # save_path = os.path.join(save_base_path, "valid")
            phase = "valid"

        uids = []   
        with open(uid_file) as f:
            uids = [x.strip() for x in f.readlines()]
        print(f"uid num: {len(uids)}")

        for idx, uid in enumerate(uids):
            print(f"num {idx}: {uid}")
            img_path = os.path.join(base_path, "dicom", uid)
            img_itk = read_dicom(img_path)
            img = sitk.GetArrayFromImage(img_itk)


            classes = os.listdir(os.path.join(base_path, "mask"))
            tmp = {}
            for cc in classes:
                mask_path = os.path.join(base_path, "mask", cc, uid)
                tmp[cc] = mask_path

            mask_path = os.path.join(base_path, "mask", "Sphere", uid)
            mask_itk = read_dicom(mask_path)
            mask = sitk.GetArrayFromImage(mask_itk)

            img_shape = img.shape
            mask_shape = mask.shape
            print(img_shape, mask_shape)

            meta_shape[uid] = {}
            meta_shape[uid]["image_path"] = img_path
            meta_shape[uid]["mask_path"] = tmp
            meta_shape[uid]["image_shape"] = img_shape
            meta_shape[uid]["mask_shape"] = mask_shape
            meta_shape[uid]["weight_shape"] = None

        if not os.path.exists(save_base_path):
            os.makedirs(save_base_path)
        json_file_name = os.path.join(save_base_path, f"{name}_{phase}.json")
        with open (json_file_name, "w") as f:
            json.dump(meta_shape, f)








