import glob
import os
from itertools import chain

import h5py
import json
import numpy as np
import random
import time
import SimpleITK as sitk

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import (
    get_slice_builder, ConfigDataset, calculate_stats, read_dicom_vtk, read_dicom, read_dicom_itk
)
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('HDF5Dataset')


class AbstractHDF5Dataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.

    Args:
        file_path (str): path to H5 file containing raw data as well as labels and per pixel weights (optional)
        phase (str): 'train' for training, 'val' for validation, 'test' for testing
        slice_builder_config (dict): configuration of the SliceBuilder
        transformer_config (dict): data augmentation configuration
        raw_internal_path (str or list): H5 internal path to the raw dataset
        label_internal_path (str or list): H5 internal path to the label dataset
        weight_internal_path (str or list): H5 internal path to the per pixel weights (optional)
        global_normalization (bool): if True, the mean and std of the raw data will be calculated over the whole dataset
    """

    def __init__(
            self, 
            file_path, 
            # base_path,
            uid,
            phase,
            slice_builder_config, 
            transformer_config, 
            shapeInfos,
            raw_internal_path='raw',
            label_internal_path='label', 
            weight_internal_path=None, 
            global_normalization=True,
            # classes="Sphere"
            
    ):
        assert phase in ['train', 'val', 'test']

        # import pdb; pdb.set_trace()
        self.phase = phase
        # self.classes = classes 
        self.file_path = file_path
        self.raw_internal_path = raw_internal_path
        self.label_internal_path = label_internal_path
        self.weight_internal_path = weight_internal_path

        # self.min_value = transformer_config["raw"][0]["min_value"]
        # self.max_value = transformer_config["raw"][0]["max_value"]
        # self.base_path = base_path
        # self.uid = uid
        image_shape = shapeInfos[uid]["image_shape"]
        weight_shape = None

        self.input_file = self.create_h5_file(file_path)

        # self.raw = self.load_dataset(input_file, raw_internal_path)

        # dicom_path = os.path.join(self.base_path, "dicom", self.uid)
        # self.image_itk = read_dicom_itk(dicom_path)

        # mask_path = os.path.join(self.base_path, "mask", self.classes, self.uid)
        # self.mask_itk = read_dicom_itk(mask_path)

        tmp = np.array(np.zeros(image_shape))
        stats = calculate_stats(tmp, global_normalization)  ## 只是为了返回空的stats
        # stats = calculate_stats(self.raw, global_normalization)

        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase != 'test':
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.label_transform()
            # self.label = self.load_dataset(input_file, label_internal_path)
            mask_shape = shapeInfos[uid]["mask_shape"]

            # if weight_internal_path is not None:
            if weight_shape is not None:
                # look for the weight map in the raw file
                # self.weight_map = self.load_dataset(input_file, weight_internal_path)
                self.weight_transform = self.transformer.weight_transform()
                weight_shape = shapeInfos[uid]["weight_shape"]
            else:
                self.weight_map = None

            # self._check_volume_sizes(self.raw, self.label)
            self._check_shape_sizes(image_shape, mask_shape)
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            # self.label = None
            # self.weight_map = None
            mask_shape = None
            weight_shape = None

        ## close h5 file  dulicui
        # self.close_h5_file(input_file)

        # build slice indices for raw and label data sets
        # slice_builder = get_slice_builder(self.raw, self.label, self.weight_map, slice_builder_config)

        start_time = time.time()
        slice_builder = get_slice_builder(image_shape, mask_shape, weight_shape, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices
        print(f"slice time: {time.time() - start_time:.6f} seconds")

        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

        # import pdb; pdb.set_trace()
        # dicom_path = "/mnt/Data/data03/dst/dicom/20220624-100017/"
        # raw = read_dicom(dicom_path)["array"]

    @staticmethod
    def load_dataset(input_file, internal_path):
        ds = input_file[internal_path][:]
        assert ds.ndim in [3, 4], \
            f"Invalid dataset dimension: {ds.ndim}. Supported dataset formats: (C, Z, Y, X) or (Z, Y, X)"
        return ds

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # dicom_path = os.path.join(self.base_path, "dicom", self.uid)
        # # print(f"\n\n dicom_path: {dicom_path}=====")
        # raw = read_dicom(dicom_path)["array"]

        # raw = sitk.GetArrayFromImage(self.image_itk)
        # input_file = self.create_h5_file(self.file_path)
        raw = self.load_dataset(self.input_file, self.raw_internal_path)
        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_slice = raw[raw_idx]

        # raw_slice[raw_slice > self.max_value] = self.max_value
        # raw_slice[raw_slice < self.min_value] = self.min_value

        raw_patch_transformed = self.raw_transform(raw_slice)

        if self.phase == 'test':
            # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
            if len(raw_idx) == 4:
                raw_idx = raw_idx[1:]
            return raw_patch_transformed, raw_idx
        else:
            # get the slice for a given index 'idx'
            # mask_path = os.path.join(self.base_path, "mask", self.classes, self.uid)
            # # print(f"\n\n mask_path: {mask_path}=====")
            # mask = read_dicom(mask_path)["array"]
            # mask = sitk.GetArrayFromImage(self.mask_itk)
            mask = self.load_dataset(self.input_file, self.label_internal_path)
            label_idx = self.label_slices[idx]
            label_patch_transformed = self.label_transform(mask[label_idx])
            # if self.weight_map is not None:
            #     weight_idx = self.weight_slices[idx]
            #     weight_patch_transformed = self.weight_transform(self.weight_map[weight_idx])
            #     return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
            # return the transformed raw and label patches

            # self.close_h5_file(self.input_file)
            return raw_patch_transformed, label_patch_transformed

    def __len__(self):
        return self.patch_count

    @staticmethod
    def create_h5_file(file_path):
        raise NotImplementedError
    
    @staticmethod
    def close_h5_file(handle):
        handle.close()
    
    @staticmethod
    def load_json_file(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    @staticmethod
    def _check_volume_sizes(raw, label):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

        assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'

    ## Add by dulicui
    @staticmethod
    def _check_shape_sizes(image_shape, mask_shape):
        def _volume_shape(shape):
            if len(shape) == 3:
                return shape
            return shape[1:]

        assert len(image_shape) in [3, 4], 'shape must be 3D (DxHxW) or 4D (CxDxHxW)'
        assert len(mask_shape) in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

        assert _volume_shape(image_shape) == _volume_shape(mask_shape), 'Raw and labels have to be of the same size'

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        # base_path = phase_config["base_path"]
        # classes = phase_config["classes"]
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths

        file_paths = cls.traverse_h5_paths(file_paths)
        # file_paths = cls.traverse_dicom_paths(file_paths)
        # file_paths = os.listdir(base_path)

        # # import pdb; pdb.set_trace()
        json_file = phase_config['json_path']
        shapeInfos = cls.load_json_file(json_file)
        # uids = list(shapeInfos.keys())

        # ### ==============MEM is not enough==========
        # num_tmp = len(uids)
        # if phase == "train":
        #     num = 300 #300 #int(num_tmp * 2/3)
        # elif phase == "val":
        #     num = 50 #50 #if int(num_tmp * 1/3) else int(num_tmp * 1/3)
        # else:
        #     num = len(uids)
        # uids = random.sample(uids, num)
        # ### ==============MEM is not enough==========

         ### ==============MEM is not enough==========
        num_tmp = len(file_paths)
        if phase == "train":
            num = 300 #int(num_tmp * 2/3)
        elif phase == "val":
            num = 50 #if int(num_tmp * 1/3) else int(num_tmp * 1/3)
        else:
            num = len(file_paths)
        file_paths = random.sample(file_paths, num)
        ### ==============MEM is not enough==========

        datasets = []
        samples_num = 0
        # for file_path in file_paths:
        for cnt, file_path in enumerate(file_paths):
        # import pdb; pdb.set_trace()
        # for cnt, uid in enumerate(uids):
            try:
                uid = file_path.split("/")[-1].split(".")[0]
                logger.info(f'Loading sample-{cnt + 1} {phase} set from: {uid}...')
                dataset = cls(
                              file_path=file_path,
                            #   base_path=base_path,
                              uid=uid,
                              phase=phase,
                            #   base_path,
                            #   uid,
                            #   phase,
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              raw_internal_path=dataset_config.get('raw_internal_path', 'raw'),
                              label_internal_path=dataset_config.get('label_internal_path', 'label'),
                              weight_internal_path=dataset_config.get('weight_internal_path', None),
                              global_normalization=dataset_config.get('global_normalization', None),
                              shapeInfos=shapeInfos,
                            #   classes=classes,
                )
                datasets.append(dataset)
                samples_num += dataset.patch_count
            except Exception:
                logger.error(f'Skipping {phase} set: {uid}', exc_info=True)
        logger.info(f"\n================{phase}: {samples_num}===============\n")
        return datasets

    @staticmethod
    def traverse_h5_paths(file_paths):
        assert isinstance(file_paths, list)
        results = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # if file path is a directory take all H5 files in that directory
                iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
                for fp in chain(*iters):
                    results.append(fp)
            else:
                results.append(file_path)
        return results


class THStandardHDF5Dataset1(AbstractHDF5Dataset):
    """
    Implementation of the HDF5 dataset which loads the data from the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    # def __init__(self, file_path, phase, slice_builder_config, transformer_config,
    #              raw_internal_path='raw', label_internal_path='label', weight_internal_path=None,
    #              global_normalization=True):
    # super().__init__(file_path=file_path, phase=phase, slice_builder_config=slice_builder_config,
    #                      transformer_config=transformer_config, raw_internal_path=raw_internal_path,
    #                      label_internal_path=label_internal_path, weight_internal_path=weight_internal_path,
    #                      global_normalization=global_normalization)
    def __init__(
            self,
            file_path, 
            # base_path,
            uid,
            phase,
            slice_builder_config, 
            transformer_config, 
            shapeInfos,
            raw_internal_path='raw',
            label_internal_path='label', 
            weight_internal_path=None, 
            global_normalization=True,
            # classes="Sphere"
    ):
        super().__init__(file_path=file_path, uid=uid, phase=phase, slice_builder_config=slice_builder_config, 
                         transformer_config=transformer_config, raw_internal_path=raw_internal_path,
                         label_internal_path=label_internal_path, weight_internal_path=weight_internal_path,
                         shapeInfos=shapeInfos, 
                         global_normalization=global_normalization)
        

    @staticmethod
    def create_h5_file(file_path):
        return h5py.File(file_path, 'r')


class LazyHDF5Dataset(AbstractHDF5Dataset):
    """Implementation of the HDF5 dataset which loads the data lazily. It's slower, but has a low memory footprint."""

    def __init__(self, file_path, phase, slice_builder_config, transformer_config,
                 raw_internal_path='raw', label_internal_path='label', weight_internal_path=None,
                 global_normalization=False):
        super().__init__(file_path=file_path, phase=phase, slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config, raw_internal_path=raw_internal_path,
                         label_internal_path=label_internal_path, weight_internal_path=weight_internal_path,
                         global_normalization=global_normalization)

        logger.info("Using modified HDF5Dataset!")

    @staticmethod
    def create_h5_file(file_path):
        return LazyHDF5File(file_path)


class LazyHDF5File:
    """Implementation of the LazyHDF5File class for the LazyHDF5Dataset."""

    def __init__(self, path, internal_path=None):
        self.path = path
        self.internal_path = internal_path
        if self.internal_path:
            with h5py.File(self.path, "r") as f:
                self.ndim = f[self.internal_path].ndim
                self.shape = f[self.internal_path].shape

    def ravel(self):
        with h5py.File(self.path, "r") as f:
            data = f[self.internal_path][:].ravel()
        return data

    def __getitem__(self, arg):
        if isinstance(arg, str) and not self.internal_path:
            return LazyHDF5File(self.path, arg)

        if arg == Ellipsis:
            return LazyHDF5File(self.path, self.internal_path)

        with h5py.File(self.path, "r") as f:
            data = f[self.internal_path][arg]

        return data
