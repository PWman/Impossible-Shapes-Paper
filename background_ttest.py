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
from initialise_nets import initialise_DNN
from process_results import avg_gradcam
from skimage.segmentation import flood_fill
from scipy.ndimage.filters import gaussian_filter
from scipy import stats
from more_utils import set_seed
from copy import copy
from math import floor, log10

matplotlib.use("TkAgg")
# set_seed(2)
plt.style.use("seaborn")


def get_ffil_img(img):
    img = img.convert("L")
    img_arr = np.array(img).astype(int)
    img_arr[np.where(img_arr > 0)] = 1

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
    for ind in corner_inds:
        ffil_arr = flood_fill(ffil_arr, ind, 1)
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

        # df_raw = get_background_proportion_1iter(train_data=train_data, zoom_augs=zoom_augs)
        # imp_pct = np.mean(df_raw[df_raw["label"] == "Impossible"].background_pct)
        # df_i = pd.DataFrame([{"label": "Impossible", "background_pct": imp_pct}])
        # poss_pct = np.mean(df_raw[df_raw["label"] == "Possible"].background_pct)
        # df_p = pd.DataFrame([{"label": "Possible","background_pct": poss_pct}])
        # df = df_p.append(df_i)
        df["seed"] = i
        df_all = df_all.append(df)
    return df_all
    # if idx == 3:
    # plt.figure()
    # plt.imshow(ffil_img)
    # break


def ttest_and_plot(df):
    def round_sig(x,sig=2):
        return round(x, sig-int(floor(log10(abs(x))))-1)

    bg_poss = df[df["label"] == "Possible"].background_pct
    bg_imp = df[df["label"] == "Impossible"].background_pct

    bins = np.linspace(min(df["background_pct"]),max(df["background_pct"]), 20)


    tval,pval = stats.ttest_ind(bg_poss,bg_imp)
    print(f"pval = {round(pval,6)}")
    x_p, y_p, _ = plt.hist(bg_poss, bins=bins,color="r",alpha=0.5)
    x_i, y_i, _ = plt.hist(bg_imp, bins=bins,color="b",alpha=0.5)

    xmin,xmax = plt.xlim()
    x = np.linspace(xmin,xmax,100)

    mu_p, std_p = stats.norm.fit(bg_poss)
    mu_i, std_i = stats.norm.fit(bg_imp)

    pdf_p = stats.norm.pdf(x,mu_p,std_p)
    pdf_i = stats.norm.pdf(x,mu_i,std_i)

    p_mul = x_p.max()/max(pdf_p)
    i_mul = x_i.max()/max(pdf_i)

    plt.plot(x,pdf_p*p_mul, color="r")
    plt.plot(x,pdf_i*i_mul, color="b")

    plt.legend([
        f"Possible: mu={round_sig(mu_p,2)}, std={round_sig(std_p,2)}",
        f"Impossible: mu={round_sig(mu_i,2)}, std={round_sig(std_i,2)}"
    ])

    plt.title(f"Background T-Test: t-value={round_sig(tval,3)}, p-value={round_sig(pval,2):.2e}")

def get_img_average(df_all):
    df_avg = pd.DataFrame([])
    for img_name in df_all.image.unique():
        data = df_all[df_all["image"] == img_name]
        lbl = data.iloc[0]["label"]
        bg_pct = np.mean(data["background_pct"])
        df = pd.DataFrame([{
            "image":img_name,
            "label":lbl,
            "background_pct": bg_pct
        }])
        df_avg = df_avg.append(df)
    return df_avg

if __name__ == "__main__":
    df_study1 = get_background_proportion_20iter(train_data=True,zoom_augs=False)
    df_study2 = get_background_proportion_20iter(train_data=True,zoom_augs=True)
    df_avg_s1 = get_img_average(df_study1)
    df_avg_s2 = get_img_average(df_study2)

    plt.figure()
    ttest_and_plot(df_avg_s1)
    plt.savefig(os.path.join(config.expt1_dir, "Background T-test.png"))
    plt.figure()
    ttest_and_plot(df_avg_s2)
    plt.savefig(os.path.join(config.expt2_dir, "Background T-test.png"))