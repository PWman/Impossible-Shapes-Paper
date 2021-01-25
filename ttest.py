import os
import config
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from net_utils import initialise_DNN
from analysis_utils import avg_gradcam, cm_arr_to_df
from skimage.segmentation import flood_fill
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import levene, ttest_ind, ttest_rel


def get_acc_df(train_data=False, zoom_augs=False):
    if train_data:
        subdir = "train_data"
    else:
        subdir = "validation_data"

    df_imp = pd.DataFrame([])
    df_poss = pd.DataFrame([])

    ddir = config.raw_dir
    for net in config.DNNs:
        if zoom_augs:
            net_fpath = os.path.join(ddir, f"{net} sf=0.5", "confusion_matrices", subdir)
        else:
            net_fpath = os.path.join(ddir, net, "confusion_matrices", subdir)
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


def save_ttest_results(zoom_augs=False):
    df_tval, df_pval = get_acc_ttest_table(zoom_augs=zoom_augs)
    if zoom_augs:
        save_path = os.path.join(config.expt2_dir,"Accuracy T-test.xlsx")
    else:
        save_path = os.path.join(config.expt1_dir,"Accuracy T-test.xlsx")
    xl_writer = pd.ExcelWriter(save_path, engine="xlsxwriter")
    df_tval.to_excel(xl_writer, sheet_name="t-values")
    df_pval.to_excel(xl_writer, sheet_name="p-values")
    xl_writer.save()

if __name__ == "__main__":
    save_ttest_results(zoom_augs=False)
    save_ttest_results(zoom_augs=True)

# df_expt1 = get_pval_table(zoom_augs=False)
# df_expt2 = get_pval_table(zoom_augs=True)
#
# save_path = r"C:\Users\Peter\Desktop\Imposs_vs_Poss_Accuracy_T-Test.xlsx"
#
#
# df_expt2.to_excel(xl_writer, sheet_name="Study 2")
# xl_writer.save()