import h5py
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import cv2


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap=plt.cm.gray)
    plt.show()
    # plt.pause(2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    pred_base_path = "./output/res/h5"
    save_base_path = "./output/res/nii"


    uids = os.listdir(pred_base_path)
    for tmp in uids:
        uid = tmp.split("_")[0]
        if uid not in ["30", "46"]:
            continue
        h5_file = os.path.join(pred_base_path, tmp)


        import pdb; pdb.set_trace()
        with h5py.File(h5_file, 'r') as f:
            ds = f["predictions"][:]
            ds = np.squeeze(ds)
            ds = sigmoid(ds)
            print(f">=0.7: {np.argwhere(ds > 0.7).shape[0]}")
            ds[ds >= 0.7] = 1
            ds[ds < 0.7] = 0
            ds_itk = sitk.GetImageFromArray(ds)

        
            sitk.WriteImage(sitk.Cast(ds_itk, sitk.sitkInt16), os.path.join(save_base_path, f"{uid}.nii"))
            # for i in ds.shape[0]:
            #     cv2.imwrite()


    import pdb; pdb.set_trace()
    print("end!")

