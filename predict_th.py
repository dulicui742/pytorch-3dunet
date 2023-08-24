import importlib
import os

import time
import torch
import torch.nn as nn
import numpy as np
import pydicom
import vtk
from vtk.util import numpy_support

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import (
    get_test_loaders, THSliceBuilder, read_dicom_vtk, calculate_stats, get_slice_builder
)
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model


logger = utils.get_logger('UNet3DPredict')


def get_predictor(model, output_dir, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('pytorch3dunet.unet3d.predictor')
    predictor_class = getattr(m, class_name)

    return predictor_class(model, output_dir, config, **predictor_config)


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


def main():
    logger.info("===========start to inference=============")
    start_time = time.time()

    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config['model'])

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    # use DataParallel if more than 1 GPU available

    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')
        model = model.cuda()
    if torch.cuda.is_available() and not config['device'] == 'cpu':
        model = model.cuda()

    load_model_time = time.time()

    output_dir = config['loaders'].get('output_dir', None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving predictions to: {output_dir}')

    # create predictor instance
    predictor = get_predictor(model, output_dir, config)

    # for test_loader in get_test_loaders(config):
    #     # run the model prediction on the test_loader and save the results in the output_dir
    #     predictor(test_loader)

    # import pdb; pdb.set_trace()
    labels = config["predictor"]["labels"]
    sigmoid_threshold = config["predictor"]["sigmoid_threshold"]
    test_base_path = config["loaders"]["test"]["test_base_path"]
    save_base_path = config["loaders"]["output_dir"]
    slice_builder_config = config["loaders"]["test"]['slice_builder']
    transformer_config = config["loaders"]["test"]['transformer']
    epoch = os.path.splitext(model_path.split("/")[-1])[0]
    uids = os.listdir(test_base_path)
    uids = [i for i in uids if "xls" not in i]
    uids.sort(key=lambda x: int(x))

    time_dict = {}
    for uid in uids[80:90]:
        # if uid not in ["53", "91", "93","12", "35", "36", "37", "38", "39"]: #["5","7", "14", "23"]: # 
        #     continue

        logger.info("\n====dealing with: {}======" .format(uid))
        start_infernce = time.time()
        time_dict[uid] = {}

        #### load data ####
        image_path = os.path.join(test_base_path, uid, "dicom")
        res = read_dicom_vtk(image_path)
        dicom_array = res["array"]
        dicom_array[dicom_array == -3024] = -1024
        dicom_vtk = res["vtk"]
        dimensions = dicom_vtk.GetDimensions()

        #### slice ####
        tmp = time.time()
        slice_builder = get_slice_builder(dimensions[::-1], None, None, slice_builder_config)
        raw_slices = slice_builder.raw_slices
        slice_time = time.time()
        print(f" vtk: {tmp - start_infernce}, slice: {slice_time - tmp}")

        #### inference ####
        stats = calculate_stats(dicom_array, global_normalization=False)
        transformer = transforms.Transformer(transformer_config, stats)
        raw_transform = transformer.raw_transform()
        prediction_map = predictor(dicom_array, raw_transform, raw_slices)

        prediction_map = np.squeeze(prediction_map)
        prediction_map = sigmoid(prediction_map)

        # print(f">={sigmoid_threshold}: {np.argwhere(prediction_map > sigmoid_threshold).shape[0]}")
        prediction_map[prediction_map >= sigmoid_threshold] = 1
        prediction_map[prediction_map < sigmoid_threshold] = 0
        
        end_inference = time.time()

        #### 3D reconstruction ####
        prediction_map = prediction_map.astype(np.uint8)[:,::-1,:]
        vtk_data = numpy_support.numpy_to_vtk(
            np.ravel(prediction_map), dimensions[2], vtk.VTK_UNSIGNED_CHAR
        )

        image = vtk.vtkImageData()
        image.SetDimensions(dicom_vtk.GetDimensions())
        image.SetSpacing(dicom_vtk.GetSpacing())
        image.SetOrigin(dicom_vtk.GetOrigin())
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

        time_dict[uid]["Slice Num"] = dimensions[2]
        time_dict[uid]["Load Model"] = round(load_model_time - start_time, 3)
        time_dict[uid]["VTK"] = round(tmp - start_infernce, 3)
        time_dict[uid]["Slice Builder"] = round(slice_time - tmp, 3)
        time_dict[uid]["Inference"] = round(end_inference - slice_time, 3)
        time_dict[uid]["3D Recon"] = round(end_time - end_inference, 3)
        time_dict[uid]["Total"] = round(end_time - start_infernce + (load_model_time - start_time), 3)

    ### print table
    print_test_time("3DUnet", time_dict)
    print('max GPU memory used: ', torch.cuda.max_memory_allocated(int("0")) / (1024 * 1024 * 1021), 'GB')


def print_test_time(encoder_name, time_dict):
    from prettytable import PrettyTable
    test_samples = list(time_dict.keys())
    x = PrettyTable()
    x.title = "Test time of {}" .format(encoder_name)
    x.field_names = ["Test sample"] + list(time_dict[test_samples[0]].keys())

    for sp in test_samples:
        values = list(time_dict[sp].values())
        tmp = []
        tmp.append(sp)
        tmp.extend(values)
        x.add_rows([
            tmp
        ])
    print(x)


if __name__ == '__main__':
    main()
