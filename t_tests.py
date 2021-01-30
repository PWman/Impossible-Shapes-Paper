import os
import config
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from preprocessing import Preprocess
from initialise_nets import initialise_DNN
from process_results import avg_gradcam, cm_arr_to_df
from more_utils import get_ffil_img, set_seed
from skimage.segmentation import flood_fill
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import levene, ttest_ind, ttest_rel


def get_acc_df(train_data=False, study_1=True):
    if study_1:
        data_dir = config.raw_dir_expt1
    else:
        data_dir = config.raw_dir_expt2

    if train_data:
        subdir = "train_data"
    else:
        subdir = "validation_data"

    df_imp = pd.DataFrame([])
    df_poss = pd.DataFrame([])


    for net in config.DNNs:
        net_fpath = os.path.join(data_dir, net, "confusion_matrices", subdir)
        imp_scores = []
        poss_scores = []
        for file in os.listdir(net_fpath):
            fpath = os.path.join(net_fpath, file)
            # print(fpath)
            cm_arr = np.load(fpath)
            imp_scores.append(cm_arr[0, 0] / np.sum(cm_arr[0, :]))
            poss_scores.append(cm_arr[1, 1] / np.sum(cm_arr[1, :]))
        df_imp[net] = imp_scores
        df_poss[net] = poss_scores

    return df_imp, df_poss


def get_acc_ttest_table(zoom_augs=False):
    train_imp, train_poss = get_acc_df(train_data=True, zoom_augs=zoom_augs)
    train_imp = np.array(train_imp)
    train_poss = np.array(train_poss)
    # tvals_train, pvals_train = ttest_ind(train_imp, train_poss)

    val_imp, val_poss = get_acc_df(train_data=False, zoom_augs=zoom_augs)
    val_imp = np.array(val_imp)
    val_poss = np.array(val_poss)
    # tvals_val, pvals_val = ttest_ind(val_imp, val_poss)

    imp_arr = np.array([train_imp, val_imp])
    poss_arr = np.array([train_poss, val_poss])

    tvals,pvals = ttest_rel(imp_arr, poss_arr, axis=1)

    df_tval_t = pd.DataFrame([{net: val for net, val in zip(config.DNNs, tvals[0])}])
    df_tval_v = pd.DataFrame([{net: val for net, val in zip(config.DNNs, tvals[1])}])

    df_all_tvals = df_tval_t.append(df_tval_v)
    df_all_tvals.index = ["training", "validation"]

    df_pval_t = pd.DataFrame([{net: val for net, val in zip(config.DNNs,pvals[0])}])
    df_pval_v = pd.DataFrame([{net: val for net, val in zip(config.DNNs,pvals[1])}])

    df_all_pvals = df_pval_t.append(df_pval_v)
    df_all_pvals.index = ["training","validation"]

    return df_all_tvals, df_all_pvals


def save_acc_ttest_results(zoom_augs=False):
    df_tval, df_pval = get_acc_ttest_table(zoom_augs=zoom_augs)
    if zoom_augs:
        save_path = os.path.join(config.expt2_dir,"Accuracy T-test.xlsx")
    else:
        save_path = os.path.join(config.expt1_dir,"Accuracy T-test.xlsx")
    xl_writer = pd.ExcelWriter(save_path, engine="xlsxwriter")
    df_tval.to_excel(xl_writer, sheet_name="t-values")
    df_pval.to_excel(xl_writer, sheet_name="p-values")
    xl_writer.save()



def get_background_proportion_1iter(train_data=True, study_1=True):
    def get_background_proportion(img_arr):
        return np.sum(img_arr) / (img_arr.shape[0] * img_arr.shape[1])

    data_transforms = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.ToPILImage(),
        transforms.Grayscale()

    ])

    if study_1:
        prepro_dir = os.path.join(config.prepro_dir, "Study 1")
    else:
        prepro_dir = os.path.join(config.prepro_dir, "Study 2")

    p = Preprocess(data_dir=prepro_dir,  batch_size=1, augment=True, shuffle=False)

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


def get_background_proportion_20iter(train_data=True,study_1=True):
    df_all = pd.DataFrame([])
    for i in range(20):
        set_seed(i)
        print(f"Testing iteration {i}...")
        df = get_background_proportion_1iter(train_data=train_data, study_1=study_1)
        df["seed"] = i
        df_all = df_all.append(df)
    return df_all
# if __name__ == "__main__":
#     save_acc_ttest_results(zoom_augs=False)
#     save_acc_ttest_results(zoom_augs=True)
