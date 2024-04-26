import os
import glob
import numpy as np
import SimpleITK as sitk
from tqdm.contrib import tzip
from collections import Counter

CT_paths = sorted(glob.glob('D:/datasets/hecktor2022/images/*__CT.nii.gz'))
PT_paths = sorted(glob.glob('D:/datasets/hecktor2022/images/*__PT.nii.gz'))
label_paths = sorted(glob.glob('D:/datasets/hecktor2022/labels/*.nii.gz'))

os.makedirs("D:/datasets/hecktor2022/npy_images", exist_ok=True)
os.makedirs("D:/datasets/hecktor2022/npy_labels", exist_ok=True)

hs = []
ws = []
for ct_path, pt_path, label_path in tzip(CT_paths, PT_paths, label_paths):
    name = os.path.basename(ct_path)
    name = name.replace('.nii.gz', '')
    ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
    pt = sitk.GetArrayFromImage(sitk.ReadImage(pt_path)).astype(np.float32)
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.uint8)
    print(np.unique(label))
    if np.sum(label) > 0:
        x,y,z = np.where(label>0)
        zmin = np.min(z)
        zmax = np.max(z)
        ct = ct[:,:,zmin:zmax+1]
        pt = pt[:,:,zmin:zmax+1]
        label = label[:,:,zmin:zmax+1]
        # assert ct.GetSize() == pt.GetSize()
        # assert label.GetSize() == ct.GetSize()

        ct = np.clip(ct, np.percentile(ct, 0.5), np.percentile(ct, 99.5))
        ct = (ct - np.mean(ct)) / np.std(ct)
        pt = np.clip(pt, np.percentile(pt, 0.5), np.percentile(pt, 99.5))
        pt = (pt - np.mean(pt)) / np.std(pt)

        for i in range(label.shape[2]):
        # Have 2 features for each pixel
            c = ct[:,:,i]
            p = pt[:,:,i]
            lab = label[:,:,i]
            img = np.stack((c, p), axis=2)

            print(img.shape)
            hs.append(img.shape[0])
            ws.append(img.shape[1])

        # Save processed images and labels to numpy format files
            np.save(f'D:/datasets/hecktor2022/npy_images/{name}_{i}.npy', img)
            np.save(f'D:/datasets/hecktor2022/npy_labels/{name}_{i}.npy', lab)


already_train_ct_name = [os.path.basename(path).split('__')[0] for path in sorted(glob.glob('D:/datasets/hecktor2022/train/cts/*.npy'))]
already_val_ct_name = [os.path.basename(path).split('__')[0] for path in sorted(glob.glob('D:/datasets/hecktor2022/val/cts/*.npy'))]
print([os.path.basename(path).split('__')[0] for path in CT_paths].index(already_train_ct_name[-1]))



