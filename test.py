import os.path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
import scipy
from tqdm.contrib import tzip
import cv2
import SimpleITK as sitk
from nets.unet import Unet

def get_pred(net, image):

    image_data = np.expand_dims(np.array(image, np.float32), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        images = images.cuda()
        pr = net(images)[0]
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        pr = pr.argmax(axis=-1)
        # 2

    image = pr
    return image


net = Unet(num_classes=3, pretrained=False, backbone='vgg')
model_dict = net.state_dict()
model_path = 'logs/best_epoch_weights.pth'

pretrained_dict = torch.load(model_path, map_location = 'cuda:0')
load_key, no_load_key, temp_dict = [], [], {}
for k, v in pretrained_dict.items():
    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
        temp_dict[k] = v
        load_key.append(k)
    else:
        no_load_key.append(k)
model_dict.update(temp_dict)
net.load_state_dict(model_dict)
net = net.cuda()
print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

image_paths = sorted(glob.glob('D:/datasets/Medical_Datasets/Images/*.npy'))
label_paths = sorted(glob.glob('D:/datasets/Medical_Datasets/Labels/*.npy'))
for image_path, label_path in tzip(image_paths, label_paths):
# image_path = "Medical_Datasets/Images/patient001_frame12_02.npy"
    name = os.path.basename(image_path).split('__')[0]
    idx = os.path.basename(image_path).split('__')[1].split('_')[-1].replace('.npy', '')
    image = np.load(image_path)
    label = np.load(label_path)

    nii = sitk.GetArrayFromImage(sitk.ReadImage(f"D:/datasets/hecktor2022/labels/{name}.nii.gz"))
    ori_h, ori_w, _ = nii.shape
    h, w, _ = image.shape

    image = scipy.ndimage.zoom(image, (256 / h, 256 / w, 1), output=None,
                             order=3, mode='nearest')
    label = scipy.ndimage.zoom(label, (256 / h, 256 / w), output=None,
                             order=1, mode='nearest')

    image = np.array(image, np.float32).transpose(2,0,1)
    pred = get_pred(net, image)

    pred = scipy.ndimage.zoom(pred, (ori_h/256, ori_w/256), output=None,
                             order=1, mode='nearest')






