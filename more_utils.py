import os
import config
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from copy import copy
from math import floor, log10
from skimage.segmentation import flood_fill
from torchvision import transforms
from preprocessing import Preprocess


def set_seed(seednum):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seednum)
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    random.seed(seednum)


def save_batch(imgs, lbls, bname=None, study_num=2):
    save_path = os.path.join(config.fully_prepro_dir, f"Study {study_num}")
    config.check_make_dir(save_path)
    for count, (img, lbl) in enumerate(zip(imgs, lbls)):
        if bname:
            img_name = f"batch_{bname}_img_{count}_class_{lbl.tolist()}.bmp"
        else:
            img_name = f"img_{count}_class_{lbl.tolist()}.bmp"
        save_name = os.path.join(save_path, img_name)
        transforms.ToPILImage()(img).save(save_name)


def round_sig(x, sig=2):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def cm_arr_to_df(arr, study_num=2):
    shapes_dir = os.path.join(config.prepro_dir, f"Study {study_num}", "Validation")
    shape_categories = os.listdir(shapes_dir)
    df = pd.DataFrame(arr)
    df.columns = [f"Predict {i}" for i in shape_categories]
    df.index = [f"Actual {i}" for i in shape_categories]
    return df


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
    return ffil_arr


def get_background_proportion(img_arr):
    return np.sum(img_arr) / (img_arr.shape[0] * img_arr.shape[1])


# def get_unnormalised_bgsizes(datadir):
#     def get_imgs_subdir(subdir):
#         df_bg_all = pd.DataFrame([])
#         for file in os.listdir(subdir):
#             fpath = os.path.join(subdir, file)
#             img = Image.open(fpath)
#             ffill_img = get_ffil_img(img)
#             bg_proportion = get_background_proportion(ffill_img)
#             df = pd.DataFrame([{
#                 "image": file.replace(".bmp", ""),
#                 "background_proportion": bg_proportion
#             }])
#             df_bg_all = df_bg_all.append(df)
#         return df_bg_all
#
#     df_all = pd.DataFrame([])
#     for dataset in os.listdir(datadir):
#         dataset_dir = os.path.join(datadir, dataset)
#         for lbl in os.listdir(dataset_dir):
#             dpath = os.path.join(datadir, dataset, lbl)
#             df = get_imgs_subdir(dpath)
#             df["label"] = lbl
#             df["dataset"] = dataset
#             df_all = df_all.append(df)
#     return df_all


if __name__ == "__main__":
    set_seed(0)
    for study_num in range(3):
        # Saves fully preprocessed images
        img_dir = os.path.join(config.prepro_dir,f"Study {study_num}")
        p = Preprocess(data_dir=img_dir, augment=True, scale_factor=0.9)
        for idx, (img, lbl) in enumerate(p.train_loader):
            save_batch(img, lbl, bname=f"{idx + 1}train", study_num=study_num)
        for idx, (img, lbl) in enumerate(p.test_loader):
            save_batch(img, lbl, bname=f"{idx + 1}test", study_num=study_num)

        # Saves background segmented images
        bg_save_dir = os.path.join(config.bg_segment_dir,f"Study {study_num}")
        config.check_make_dir(bg_save_dir)
        for dataset in os.listdir(img_dir):
            img_dset_dir = os.path.join(img_dir,dataset)
            for cat in os.listdir(img_dset_dir):
                img_cat_dir = os.path.join(img_dset_dir,cat)
                for img_name in os.listdir(img_cat_dir):
                    img = Image.open(os.path.join(img_cat_dir,img_name))
                    ffil_arr = get_ffil_img(img)
                    ffill_img = Image.fromarray(ffil_arr*255).convert("L")
                    ffill_img.save(os.path.join(bg_save_dir,img_name))