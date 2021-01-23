import os
import config
import torch
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from preprocessing import Preprocess
from torchvision import transforms
from net_utils import initialise_DNN
from analysis_utils import avg_gradcam
from skimage.segmentation import flood_fill
from scipy.ndimage.filters import gaussian_filter
from scipy import stats
from more_utils import set_seed
from copy import copy

matplotlib.use("TkAgg")
# set_seed(2)
plt.style.use("seaborn")


def get_ffil_img(img):
    img = img.convert("1")
    img_arr = np.array(img).astype(int)

    corner_inds = [(0, 0),
                   (img_arr.shape[0] - 1, 0),
                   (0, img_arr.shape[1] - 1),
                   (img_arr.shape[0] - 1, img_arr.shape[1] - 1)]

    ffil_arr = copy(img_arr)
    for ind in corner_inds:
        if img_arr[ind[0], ind[1]] == 0:
            ffil_arr = flood_fill(ffil_arr, ind, 2)

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

data_transforms = transforms.Compose([
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.ToPILImage(),
    transforms.Grayscale()

])


# def binarize(img):

def get_background_proportion_1iter(train_data=True, zoom_augs=False):
    def get_background_proportion(img_arr):
        return np.sum(img_arr) / (img_arr.shape[0] * img_arr.shape[1])

    if zoom_augs:
        p = Preprocess(batch_size=1, scale_factor=0.5, augment=True, shuffle=False)
    else:
        p = Preprocess(batch_size=1, augment=True, shuffle=False)

    if train_data:
        loader = p.train_loader
        classes = p.train_class_names
    else:
        loader = p.test_loader
        classes = p.test_class_names

    df_all = pd.DataFrame([])
    for idx, (torch_img, lbl) in enumerate(loader):
        shape_lbls = loader.dataset.samples[idx]

        shape_name = os.path.basename(shape_lbls[0]).replace(".bmp", "")
        # torch_img = torch.squeeze(torch_img)
        pil_img = data_transforms(torch.squeeze(torch_img))
        ffil_img = get_ffil_img(pil_img)
        bg_pct = get_background_proportion(ffil_img)
        df = pd.DataFrame([{"image": shape_name,
                            "label_num": shape_lbls[1],
                            "label": classes[shape_lbls[1]],
                            "background_pct": bg_pct
                            }])
        df_all = df_all.append(df)
    return df_all


def get_background_proportion_20iter(train_data=True,zoom_augs=False):
    df_all = pd.DataFrame([])
    for i in range(20):
        set_seed(i)
        print(f"Testing iteration {i}...")
        df = get_background_proportion_1iter(train_data=train_data, zoom_augs=zoom_augs)
        df["seed"] = i
        df_all = df_all.append(df)
    return df_all
    # if idx == 3:
    # plt.figure()
    # plt.imshow(ffil_img)
    # break
def ttest_and_plot(df):
    bg_poss = df[df["label"] == "Possible"].background_pct
    bg_imp = df[df["label"] == "Impossible"].background_pct

    plt.hist(bg_poss,color="r")
    plt.hist(bg_imp,color="b")