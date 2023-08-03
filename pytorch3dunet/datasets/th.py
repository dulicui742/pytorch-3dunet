import collections
import os

import imageio
import numpy as np
import torch
import SimpleITK as sitk

from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.utils import ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('THSpereDataset')


def read_dicom(dicom_path):
    # print("Start trans ", dicom_path)
    series_reader = sitk.ImageSeriesReader()
    dicom_names = series_reader.GetGDCMSeriesFileNames(dicom_path)
    series_reader.SetFileNames(dicom_names)
    dicom_img = series_reader.Execute()
    return dicom_img


def resample_sitk(src_img, interpolator=sitk.sitkLinear, dest_spacing=[1., 1., 1.]):
    src_size = src_img.GetSize()
    src_spacing = src_img.GetSpacing()
    dest_size = np.array(src_size) * np.array(src_spacing) / np.array(dest_spacing)
    dest_size = np.round(dest_size).astype(np.int64).tolist()
    print(f"=======src size: {src_size}, dest size: {dest_size}")

    return sitk.Resample(src_img, dest_size, sitk.Transform(), interpolator,
                        src_img.GetOrigin(), dest_spacing, src_img.GetDirection(),
                        0, src_img.GetPixelID())


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


def dsb_prediction_collate(batch):
    """
    Forms a mini-batch of (images, paths) during test time for the DSB-like datasets.
    """
    error_msg = "batch must contain tensors or str; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], collections.Sequence):
        # transpose tuples, i.e. [[1, 2], ['a', 'b']] to be [[1, 'a'], [2, 'b']]
        transposed = zip(*batch)
        return [dsb_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class THSpereDataset(ConfigDataset):
    def __init__(
                    self, 
                    root_dir, 
                    uid_file, 
                    phase, 
                    transformer_config, 
                    expand_dims=True,
                    # stl_classes=["sphere"]
    ):
        assert os.path.isdir(root_dir), f'{root_dir} is not a directory'
        assert phase in ['train', 'val', 'test']

        self.phase = phase
        self.expand_dims = expand_dims

        # load raw images
        uids = []

        # print(f"\n~~~~~~~~~~uid file: {uid_file}")
        with open(uid_file) as f:
            uids = [x.strip() for x in f.readlines()]

        images_dir = os.path.join(root_dir, 'dicom')
        assert os.path.isdir(images_dir)

        self.uids = uids
        self.images_dir = images_dir
        self.masks_dir = os.path.join(root_dir, "mask", "Sphere")
        print(f"~~~~~~~~~~phase: {phase}, uids: {len(self.uids)}")

        # self.image_labels = []
        # for uid in uids:
        #     for filename in os.listdir(images_dir, uid):
        #         self.image_labels.append()



        # self.images, self.paths = self._load_files(images_dir, uids, expand_dims)
        # self.file_path = images_dir

        # # import pdb; pdb.set_trace()
        # # stats = calculate_stats(self.imagraw_transformes, True)
        # stats = calculate_stats(self.images, True)

        # transformer = transforms.Transformer(transformer_config, stats)

        # # load raw images transformer
        # self.raw_transform = transformer.raw_transform()

        # if phase != 'test':
        #     # load labeled images
        #     masks_dir = os.path.join(root_dir, 'mask', "Sphere")
        #     assert os.path.isdir(masks_dir)
        #     self.masks, _ = self._load_files(masks_dir, uids, expand_dims)
        #     assert len(self.images) == len(self.masks)
        #     # load label images transformer
        #     self.masks_transform = transformer.label_transform()
        # else:
        #     self.masks = None
        #     self.masks_transform = None

    # def __getitem__(self, idx):
    #     if idx >= len(self):
    #         raise StopIteration

    #     img = self.images[idx]
    #     if self.phase != 'test':
    #         mask = self.masks[idx]
    #         return self.raw_transform(img), self.masks_transform(mask)
    #     else:
    #         return self.raw_transform(img), self.paths[idx]


    def __getitem__(self, idx):
        # if idx >= len(self):
        #     raise StopIteration

        img = self._load_file(self.images_dir, self.uids[idx], self.expand_dims)
        if self.phase != 'test':
            mask = self._load_file(self.masks_dir, self.uids[idx], self.expand_dims)
            # return self.raw_transform(img), self.masks_transform(mask)
            return img, mask
        else:
            # return self.raw_transform(img), self.paths[idx]
            return img, os.path.join(self.images_dir, "Sphere", self.uids[idx])



    def __len__(self):
        # return len(self.images)
        return len(self.uids)

    @classmethod
    def prediction_collate(cls, batch):
        return dsb_prediction_collate(batch)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load files to process
        uid_file = phase_config['uid_file']
        file_paths = phase_config["file_paths"]
        expand_dims = dataset_config.get('expand_dims', True)
        # return [cls(file_paths[0], phase, transformer_config, expand_dims)]
        # import pdb; pdb.set_trace()
        return [cls(file_paths, uid_file, phase, transformer_config, expand_dims)]

    # @staticmethod
    # def _load_files(dir, expand_dims):
    #     files_data = []
    #     paths = []
    #     import pdb; pdb.set_trace()
    #     for file in os.listdir(dir):
    #         path = os.path.join(dir, file)
    #         img = np.asarray(imageio.imread(path))
    #         if expand_dims:
    #             dims = img.ndim
    #             img = np.expand_dims(img, axis=0)
    #             if dims == 3:
    #                 img = np.transpose(img, (3, 0, 1, 2))

    #         files_data.append(img)
    #         paths.append(path)

    #     return files_data, paths

    

    @staticmethod
    def _load_files(dir, uids, expand_dims):
        files_data = []
        paths = []
        # import pdb; pdb.set_trace()
        for file in uids[:3]:
            path = os.path.join(dir, file)
            # img = np.asarray(imageio.imread(path))

            img_itk = read_dicom(path)
            spacing = img_itk.GetSpacing()
            # img_itk = resample_sitk(img_itk, dest_spacing=[5.,5.,5.])

            img = sitk.GetArrayFromImage(img_itk)

            if expand_dims:
                dims = img.ndim
                img = np.expand_dims(img, axis=0)
                # if dims == 3:
                #     # img = np.transpose(img, (3, 0, 1, 2))
                #     img = np.transpose(img, (1, 0, 2, 3))

            files_data.append(img)
            paths.append(path)

        return files_data, paths

    
    @staticmethod
    def _load_file(dir, uid, expand_dims):
        path = os.path.join(dir, uid)
        img_itk = read_dicom(path)
        spacing = img_itk.GetSpacing()
        # img_itk = resample_sitk(img_itk, dest_spacing=[5.,5.,5.])
        img_itk = resample_sitk_1(img_itk, dest_size=[64, 64, 64])
        img = sitk.GetArrayFromImage(img_itk)

        if expand_dims:
            dims = img.ndim
            img = np.expand_dims(img, axis=0)
    
        return img

