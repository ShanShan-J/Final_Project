import os.path

import numpy as np
import torch
import torch.nn.functional as F
from nets.unet import Unet
import matplotlib.pyplot as plt
import glob
import scipy
from tqdm.contrib import tzip
import cv2
import SimpleITK as sitk

image_paths = sorted(glob.glob('D:/datasets/Medical_Datasets/Images/*.npy'))
label_paths = sorted(glob.glob('D:/datasets/Medical_Datasets/Labels/*.npy'))

for image_path, label_path in tzip(image_paths, label_paths):
    name = os.path.basename(image_path).split('__')[0]
    idx = os.path.basename(image_path).split('__')[1].split('_')[-1].replace('.npy', '')
    image = np.load(image_path)
    label = np.load(label_path)
    print(image.shape, label.shape)


    image = np.array(image, np.float32).transpose(2, 0, 1)
    nii = sitk.GetArrayFromImage(sitk.ReadImage(f"D:/datasets/hecktor2022/labels/{name}.nii.gz"))

    ori_h, ori_w, _ = nii.shape
    h, w, _ = image.shape


    black = np.zeros_like(label, dtype=np.uint8)
    ct = ((image[0]-np.min(image[0]))/(np.max(image[0])-np.min(image[0]))*255).astype(np.uint8)
    # glay (256, 256) rgb (512, 512, 3)
    ct = np.stack((ct, ct, ct), axis=2)
    # (512, 512, 3)
    pt = ((image[1]-np.min(image[1]))/(np.max(image[1])-np.min(image[1]))*255).astype(np.uint8)
    pt = np.stack((pt, pt, pt), axis=2)

    # label (512, 512) [0, 1, 2] [False, True] = [0, 1]
    if np.sum(label==1) > np.sum(label==2):
        red = ((label==1)*255).astype(np.uint8) # [0, 255]
        green = ((label==2)*255).astype(np.uint8) # [0, 255]
        show = np.stack((black.copy(), black.copy(), black.copy()), axis=2)
        show[:,:,2] = red
        show[:,:,1] = green
        show[:,:,2][label==2] = 0
        # [0, 255, 0], [0, 0, 255]
    else:
        red = ((label == 1) * 255).astype(np.uint8)
        green = ((label == 2) * 255).astype(np.uint8)
        show = np.stack((black.copy(), black.copy(), black.copy()), axis=2)
        show[:,:,1] = green
        show[:,:,2] = red
        show[:,:,1][label==1] = 0
    # print(ct.shape, pt.shape, show.shape)
    # [0, 0, 255] + [0, 0, 255] = [0, 0, 255*0.7+255*0.3]
    ct_show = cv2.addWeighted(ct,0.7, show, 0.3, 0)
    pt_show = cv2.addWeighted(pt,0.7, show, 0.3, 0)

    cv2.imwrite(f"D:/datasets/Medical_Datasets/ShowLabel/ct_{name}_{idx}.png", ct_show)
    cv2.imwrite(f"D:/datasets/Medical_Datasets/ShowLabel/pt_{name}_{idx}.png", pt_show)

