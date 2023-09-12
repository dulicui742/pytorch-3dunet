import importlib
import os

import time
import torch
import torch.nn as nn
import numpy as np
import pydicom
import vtk
from vtk.util import numpy_support


def read_dicom_vtk(dicom_path):
    dicomreader = vtk.vtkDICOMImageReader()
    dicomreader.SetDirectoryName(dicom_path)
    dicomreader.Update()
    output = dicomreader.GetOutput()
    dimensions = output.GetDimensions()
    dicomArray = numpy_support.vtk_to_numpy(output.GetPointData().GetScalars())
    dicomArray = dicomArray.reshape(dimensions[::-1]).astype(np.float32)
    dicomArray = dicomArray[:,::-1,:]
    # dicomArray[dicomArray == -3024] = -1024
    return {
        "vtk": output,
        "array": dicomArray
    }



# base_path = "/media/ubuntu/Elements SE/data_sphere_test/"
# uids = os.listdir(base_path)
# uids = [i for i in uids if "xls" not in i]
# uids.sort(key=lambda x: int(x))
# for uid in uids[:10]:
#     image_path = os.path.join(base_path, uid, "dicom")
#     print(image_path)
#     start = time.time()
#     aa = read_dicom_vtk(image_path)
#     print(f"\n{uid}: {time.time() - start}")


base_path = "/media/ubuntu/Elements SE/data00/dst/"
uid_file = "/media/ubuntu/Elements SE/data00/dst/500_100/train.txt"
with open(uid_file) as f:
    uids = [x.strip() for x in f.readlines()]


for uid in uids[:15]:
    image_path = os.path.join(base_path, "dicom",  uid)
    # print(image_path)
    start = time.time()
    aa = read_dicom_vtk(image_path)
    print(f"\n{uid}: {time.time() - start}")


