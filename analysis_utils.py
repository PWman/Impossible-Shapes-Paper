import os
import config
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gcam_utils import plot_cam_on_img
from train_test_utils import gcam_all_imgs
from more_utils import plot_and_save, cm_arr_to_df
from torchvision import transforms

plt.style.use("seaborn-bright")


def avg_train_results(net_name):
    avg_results = []
    scores = pd.DataFrame(columns=["acc", "val_acc", "loss", "val_loss"])
    results_path = os.path.join(config.raw_dir, net_name, "train_results")
    for file in os.listdir(results_path):
        # print(file)
        # print(os.path.join(results_path, file))
        result = pd.read_csv(os.path.join(results_path, file), index_col=0)
        end_result = result[result["epoch"] == max(result["epoch"])]  # .drop(columns=["epoch"])
        scores = scores.append(end_result.drop(columns=["epoch"]))
        avg_results.append(result)

    return sum(avg_results) / len(avg_results), scores


def total_cmats(net_name):
    cm_dir = os.path.join(config.raw_dir, net_name, "confusion_matrices")
    train_dir = os.path.join(cm_dir, "train_data")
    val_dir = os.path.join(cm_dir, "validation_data")
    cm_tot_t = np.zeros((2, 2))
    for file in os.listdir(train_dir):
        cm_t = np.load(os.path.join(train_dir, file))
        cm_tot_t = cm_tot_t + cm_t
    cm_tot_v = np.zeros((2, 2))
    for file in os.listdir(val_dir):
        cm_v = np.load(os.path.join(val_dir, file))
        cm_tot_v = cm_tot_v + cm_v
    return cm_tot_t, cm_tot_v


def avg_gradcam(net_name):
    gcam_path = os.path.join(config.raw_dir, net_name, "gradCAM")
    mask_path = os.path.join(gcam_path, "masks")
    gstat_path = os.path.join(gcam_path, "scores")

    MASKS = []
    GSTATS = pd.DataFrame([])
    itercam = zip(os.listdir(mask_path), os.listdir(gstat_path))
    for seed, (arr, csv) in enumerate(itercam):
        mask = np.load(os.path.join(mask_path, arr))
        MASKS.append(mask)
        df = pd.read_csv(os.path.join(gstat_path, csv))
        df["seed_num"] = seed
        GSTATS = GSTATS.append(df)
    GSTATS.to_csv(os.path.join(gcam_path, "all_gradCAMs_info.csv"))

    MASKS = np.array(MASKS)
    seed_idx = GSTATS[GSTATS["avg_include"] == False]["seed_num"]
    img_idx = seed_idx.index
    for iidx, sidx in zip(img_idx, seed_idx):
        # print(np.sum(MASKS[sidx, iidx, :, :]))
        MASKS[sidx, iidx, :, :] = np.nan
    avg_mask = np.nanmean(MASKS, axis=0)
    cols = [ "avg_include", "correct", "nan_array", "prediction"]
    avg_gstats = pd.pivot_table(GSTATS, values=cols,
                                index=["img_name", "img_path"],
                                aggfunc=np.sum)

    avg_gstats.columns = ["n_usable_cams", "n_correct", "n_nans", "n_poss_preds"]
    avg_gstats = pd.DataFrame(avg_gstats.to_records())

    # for idx, img in enumerate(GSTATS["img_name"].unique()):
    #     df = GSTATS[GSTATS["img_name"] == img]
    #     correct_inds = list(df["correct"] == True)
    #     nan_inds = list(df["correct"] == False)
    #     for c,n in correct_inds, nan_inds:
    #         if not (c==1 and n==1):
    #             MASKS[0,0,:,:]


    # avg_mask = np.mean(MASKS, axis=0)
    #
    # GSTATS = pd.DataFrame([])
    # for idx,file in enumerate(os.listdir(gstat_path)):
    #     df = pd.read_csv(os.path.join(gstat_path, file), index_col=0)
    #     df["seed"] = idx
    #     GSTATS = GSTATS.append(df)
    # avg_gstats = pd.pivot_table(GSTATS, values=["correct", "prediction"],
    #                             index=["img_name", "img_path"], aggfunc=np.mean)
    # avg_gstats.columns = ["accuracy", "pct_poss_preds"]
    # avg_gstats = pd.DataFrame(avg_gstats.to_records())
    return avg_mask, avg_gstats


def avg_save_net_results(net_name):

    if "sf=0.5" in net_name:
        result_dir = os.path.join(config.expt2_dir, net_name[:net_name.find("sf=0.5")-1])
    elif "sf=" in net_name:
        result_dir = os.path.join(config.results_basedir, net_name)
    else:
        result_dir = os.path.join(config.expt1_dir, net_name)
    config.check_make_dir(result_dir)

    print("Processing training results...")
    train_results, _ = avg_train_results(net_name)
    train_results.to_csv(os.path.join(result_dir, "Train Results.csv"))
    plot_and_save(train_results, result_dir)

    print("Processing confusion matrices...")
    cm_t, cm_v = total_cmats(net_name)
    cm_path = os.path.join(result_dir, "Confusion Matrices.xlsx")
    cm_writer = pd.ExcelWriter(cm_path, engine="xlsxwriter")
    cm_t = cm_arr_to_df(cm_t)
    cm_v = cm_arr_to_df(cm_v)
    cm_t.to_excel(cm_writer, sheet_name="Training")
    cm_v.to_excel(cm_writer, sheet_name="Validation")
    cm_writer.save()

    print("Processing GradCAM results...")
    avg_masks, avg_gstats = avg_gradcam(net_name)
    gcam_dir = os.path.join(result_dir, "gradCAM")
    config.check_make_dir(gcam_dir)
    avg_gstats.to_csv(os.path.join(gcam_dir, "GradCAM Info.csv"))
    img_paths = list(avg_gstats["img_path"])
    for img, mask in zip(img_paths, avg_masks):
        mask_img = plot_cam_on_img(img, mask)
        img_name = os.path.basename(img)
        mask_img.save(os.path.join(gcam_dir, img_name))

if __name__ == "__main__":
    for net in config.DNNs:
        if "pretrain" in net:
            avg_save_net_results(f"{net} sf=0.5")
