import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import vtk
import random
import pydicom
import h5py
import pandas as pd

from vtk.util import numpy_support



def get_affine_matrix(dicom_dir):
    '''
    Calculate the affine matrix which maps the voxel from the DICOM voxel space (column, row, depth) to patient
    coordinate system (x, y, z)
    reference: https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-slice-affine
    written by Shumao Pang, pangshumao@126.com
    :param dicom_dir: a dicom series dir
    :return: a numpy array with shape of (4, 4)
    '''
    
    dicom_file_list = os.listdir(dicom_dir)
    slice_num = len(dicom_file_list)
    dicom_list = [0] * slice_num
    for dicom_file in dicom_file_list:
        ds = pydicom.read_file(os.path.join(dicom_dir, dicom_file))
        instance_number = ds.InstanceNumber
        dicom_list[instance_number - 1] = ds

    rows = dicom_list[0].Rows
    columns = dicom_list[0].Columns
    pixel_spacing = dicom_list[0].PixelSpacing
    image_orientation_patient = dicom_list[0].ImageOrientationPatient
    orientation_matrix = np.reshape(image_orientation_patient, [3, 2], order='F')
    # orientation_matrix = orientation_matrix[:, ::-1]
    print("orientation_matrix:\n", orientation_matrix)
 
    first_image_position_patient = np.array(dicom_list[0].ImagePositionPatient)
    last_image_position_patient = np.array(dicom_list[-1].ImagePositionPatient)
    k = (last_image_position_patient - first_image_position_patient) / (slice_num - 1)
 
    # affine_matrix = np.zeros((4, 4), dtype=np.float32)
    # affine_matrix[:3, 0] = orientation_matrix[:, 0] * pixel_spacing[0]
    # affine_matrix[:3, 1] = orientation_matrix[:, 1] * pixel_spacing[1]
    # affine_matrix[:3, 2] = k
    # affine_matrix[:3, 3] = first_image_position_patient ##debug

    ## ref: https://github.com/dgobbi/vtk-dicom/blob/master/Source/vtkDICOMReader.cxx
    if k[2] < 0:
        k[2] = -1
    else: 
        k[2] = 1
    affine_matrix = np.zeros((4, 4), dtype=np.float32)
    affine_matrix[:3, 0] = orientation_matrix[:, 0] 
    affine_matrix[:3, 1] = orientation_matrix[:, 1] * (-1) 
    affine_matrix[:3, 2] = k 
    affine_matrix[:3, 3] = first_image_position_patient + pixel_spacing[1] * (rows - 1) * orientation_matrix[:, 1]
    
    affine_matrix[3, 3] = 1.0
    return affine_matrix



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def h52stl(
        dicom_base_path, 
        meta_data_path,
        pred_base_path,
        save_base_path="",
        epoch=18, 
        labels=["Sphere"], 
        sigmoid_threshold=0.7
):
    uids = os.listdir(dicom_base_path)
    uids = [i for i in uids if "xls" not in i]
    uids.sort(key=lambda x: int(x))
    print(uids)
    for uid in uids:
        
        # start_infernce = time.time()
        if "txt" in uid or "xlsx" in uid: ## except readme
            continue


        # if uid in ["11", "13", "19", "20", "2", "23", "24", "26", "28", "29", "30", "31", "33", "34", "35", "37", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "7", "8", "9", ]:
        #     continue

        # if uid not in ["29", "12", "18", "32", "53", "74", "86", "39", "55", "91"]:
        #     continue

        # if uid not in ["30", "79", "80", "82", "83", "84", "86", "90", "91", "88"]:
        #     continue

        if uid not in ["5", "7"]: #["16", "42"]
            continue

        # if uid in ["1", "14", "22", "25", "35", "36", "37", "38", "39", "41", "42", "43", "44", "45", "46", "48", "49", "50", "5", "7", "55", "56"]:
        #     continue
        
        # import pdb; pdb.set_trace()
        h5_file = f"{uid}_predictions.h5"
        if h5_file not in os.listdir(pred_base_path):
            continue

        

        # if uid not in ["16", "17", "32","30","46", "7", "22", "35", "39", "40", "53", "91", "93"]: ## sphere
        #     continue
        
        print("\n------------------------------")
        print("dealing with: {}" .format(uid))
        # time_dict[uid] = {}

        image_path = os.path.join(dicom_base_path, uid, "dicom")
        dicomreader = vtk.vtkDICOMImageReader()
        dicomreader.SetDirectoryName(image_path)
        dicomreader.Update()
        output = dicomreader.GetOutput()
        dimensions = output.GetDimensions()
        slice_spacing = output.GetSpacing()
        origin = output.GetOrigin()
        print("dimension:", dimensions, "spacing:", slice_spacing, "origin:", origin)


        # df = pd.read_csv(meta_data_path)
        # uid_df = df[df["uid"] == int(uid)]
        # slices = uid_df["SliceNums"].to_numpy()[0]
        # PixelSpacing = uid_df["PixelSpacing"].to_numpy()[0]
        # PixelSpacing = eval(PixelSpacing)
        # sliceThickness = uid_df["sliceThickness"].to_numpy()[0]

        # dimensions = [512, 512, slices]
        # slice_spacing = [PixelSpacing[0], PixelSpacing[1], sliceThickness]
        # ImagePosition = eval(uid_df["ImagePosition"].to_numpy()[0])
        # print("dimension:", dimensions, "spacing:", slice_spacing, "origin:", ImagePosition)

        h5_file = os.path.join(pred_base_path, f"{uid}_predictions.h5")
        with h5py.File(h5_file, 'r') as f:
            ds = f["predictions"][:]
            ds = np.squeeze(ds)
            ds = sigmoid(ds)
            print(f">={sigmoid_threshold}: {np.argwhere(ds > sigmoid_threshold).shape[0]}")
            ds[ds >= sigmoid_threshold] = 1
            ds[ds < sigmoid_threshold] = 0

        # ds = ds.astype(np.uint8)
        ds = ds.astype(np.uint8)[:,::-1,:]
        vtk_data = numpy_support.numpy_to_vtk(
            np.ravel(ds), dimensions[2], vtk.VTK_UNSIGNED_CHAR
        )

        # import pdb; pdb.set_trace()
        image = vtk.vtkImageData()
        image.SetDimensions(output.GetDimensions())
        image.SetSpacing(output.GetSpacing())
        image.SetOrigin(output.GetOrigin())

        # image.SetDimensions(dimensions)
        # image.SetSpacing(slice_spacing)
        # image.SetOrigin(ImagePosition)

        image.GetPointData().SetScalars(vtk_data)
        output = image
        print(image.GetDimensions())

        contour = vtk.vtkDiscreteMarchingCubes()
        contour.SetInputData(output)
        contour.ComputeNormalsOn()
        # contour.SetValue(0, 1)
        contour.GenerateValues(len(labels), 0, len(labels) + 1)
        contour.Update()
        output = contour.GetOutput()

        matrix = get_affine_matrix(image_path)
        print("-------------")
        print(matrix)

        def Array2vtkTransform(arr):
            T = vtk.vtkTransform()
            matrix = vtk.vtkMatrix4x4()
            for i in range(0, 4):
                for j in range(0, 4):
                    matrix.SetElement(i, j, arr[i, j])
            T.SetMatrix(matrix)
            # T.SetUserMatrix(matrix)
            return T

        # transform = vtk.vtkTransform()
        # transform.SetMatrix(matrix)
        transform = Array2vtkTransform(matrix)
        transformPolyDataFilter = vtk.vtkTransformPolyDataFilter()
        transformPolyDataFilter.SetInputData(output)
        transformPolyDataFilter.SetTransform(transform)
        transformPolyDataFilter.Update()
        output = transformPolyDataFilter.GetOutput()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(output)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 1, 1)
        writer = vtk.vtkSTLWriter()
        writer.SetInputData(output)

        label_name = labels[0]
        # epoch = int(best_model_path.split("_")[-1].split(".")[0])
        stl_save_base_path = os.path.join(
            save_base_path,  
            # label_name,
            # "epoch{}_sigmoid{}" .format(epoch, str(sigmoid_threshold)),
            "{}_sigmoid{}" .format(epoch, str(sigmoid_threshold)),
        )
        if not os.path.exists(stl_save_base_path):
            os.makedirs(stl_save_base_path)
        stl_name = os.path.join(
            stl_save_base_path, "{}_{}.stl" .format(
                uid, time.strftime("%m%d_%H%M%S")
            )
        )
        writer.SetFileName(stl_name.encode("GBK"))
        writer.SetFileTypeToBinary()
        writer.Update()


        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(renderWindow)
        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.2, 0.3)
        # interactor.Initialize()
        # renderWindow.Render()
        # interactor.Start()
        end_time = time.time()


if __name__ == "__main__":
    meta_data_file = "/data/dulicui/project/data/data03_h5/meta_data.csv"
    dicom_base_path = "/mnt/Data/data_sphere_test/"
    # pred_base_path = "./output/res/h5/epoch_21_niters_93000"
    # pred_base_path = "./output/res/h5/epoch_21_niters_93000_3024-1024"
    # pred_base_path = "./output/res/h5/epoch_21_niters_93000_vtk"

    ## bronchial
    pred_base_path = "./output/res/h5/Bronchial/0811_180000/epoch_16_niters_63000_vtk/"
    save_base_path = "./output/res/stl/Bronchial/0811_180000/"
    epoch = "epoch_16_niters_63000_vtk"
    sigmoid_threshold = 0.53 #0.51 #0.6 #0.7
    labels = ["Bronchial"] #["Sphere"]

    h52stl(
        dicom_base_path=dicom_base_path, 
        meta_data_path=meta_data_file,
        pred_base_path=pred_base_path,
        save_base_path=save_base_path,
        epoch=epoch, 
        labels=labels, 
        sigmoid_threshold=sigmoid_threshold
    )


