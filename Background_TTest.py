import os
import config
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from preprocessing import Preprocess
from torchvision import transforms
from net_utils import initialise_DNN
from analysis_utils import avg_gradcam
from skimage.segmentation import flood_fill
from scipy.ndimage.filters import gaussian_filter
from scipy import stats

plt.style.use("seaborn")


def get_ffil_img(img):
    img = img.convert("1")
    img_arr = np.array(img).astype(int)
    ffil_arr = flood_fill(img_arr, (0, 0), 2)

    lineinds = tuple(np.transpose(np.where(ffil_arr == 1))[0])
    while len(lineinds) > 0:
        ffil_arr = flood_fill(ffil_arr, lineinds, 0)
        try:
            lineinds = tuple(np.transpose(np.where(ffil_arr == 1))[0])
        except IndexError:
            break
    ffil_arr = flood_fill(ffil_arr, (0, 0), 1)
    # ffil_arr = 1 - ffil_arr

    return ffil_arr
#
# def get_background_num(ddir):
#     df_all = pd.DataFrame([])
#     for file in os.listdir(ddir):
#         fpath = os.path.join(ddir, file)
#         ffill_img = get_ffil_img(fpath)
#         df = pd.DataFrame([{
#             "img_name": file.replace(".bmp", ""),
#             "file_path": fpath,
#             "background_size": np.sum(ffill_img) / (224 * 224)
#         }])
#         df_all = df_all.append(df)
#     return df_all

p = Preprocess(batch_size=1,augment=True,shuffle=False)
data_transforms = transforms.Compose([
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.ToPILImage(),
    transforms.Grayscale()

])

# def binarize(img):


for idx,(torch_img,lbl) in enumerate(p.train_loader):
    shape_lbls = p.test_loader.dataset.samples[idx]
    # torch_img = torch.squeeze(torch_img)
    pil_img = data_transforms(torch.squeeze(torch_img))
    ffil_img = get_ffil_img(pil_img)
    break

