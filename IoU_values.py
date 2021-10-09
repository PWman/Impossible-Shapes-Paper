import os
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from more_utils import get_ffil_img
from process_results import avg_gradcam

plt.style.use("seaborn")


def get_cam_mask(net_name, img_name, study_num=2, train_data=False):
    all_masks, df_camstats = avg_gradcam(net_name, study_num=study_num, train_data=train_data)
    img_idx = int(df_camstats[df_camstats["img_name"] == img_name].index[0])
    mask = all_masks[img_idx]
    if np.amin(mask) != 0:
        mask = mask - np.amin(mask)
    else:
        mask = mask / np.amax(mask)
    return mask


def calculate_IoU(gcam_mask, ROI_mask):
    IoU_mask = np.multiply(gcam_mask, ROI_mask)
    return np.sum(IoU_mask) / np.sum(gcam_mask)


def get_net_IoUs(net_name, study_num=2, train_data=False):
    def get_all_IoUs(img_dir, roi_dir=None, train_data=False):
        df_IoU = pd.DataFrame([])
        for img_fname in os.listdir(img_dir):
            img_fpath = os.path.join(img_dir, img_fname)
            img_name = img_fname.replace(".bmp", "")

            img = Image.open(img_fpath)
            background_arr = get_ffil_img(img)
            gcam_arr = get_cam_mask(net_name, img_fname, train_data=train_data, study_num=study_num)
            background_IoU = calculate_IoU(gcam_arr, background_arr)

            if roi_dir is not None:
                roi_path = os.path.join(roi_dir, f"roi_{img_fname}")
                roi_arr = np.asarray(Image.open(roi_path))
                roi_IoU = calculate_IoU(gcam_arr, roi_arr)
                df = pd.DataFrame([{
                    "Image": img_name,
                    "Impossible Region": roi_IoU,
                    "Background": background_IoU
                }])
            else:
                df = pd.DataFrame([{
                    "Image": img_name,
                    "Background": background_IoU
                }])
            df_IoU = df_IoU.append(df)
        return df_IoU

    img_dir = os.path.join(config.prepro_dir, f"Study {study_num}")
    if train_data:
        dataset = "Training"
    else:
        dataset = "Validation"

    img_dir = os.path.join(img_dir, dataset)
    labels = os.listdir(img_dir)
    # print(labels)
    if study_num > 0:
        roi_dir = os.path.join(config.shapes_basedir, "ROI", f"Study{study_num}", dataset)
    else:
        roi_dir = None
    df_IoU_imp = get_all_IoUs(os.path.join(img_dir, labels[0]), roi_dir, train_data=train_data)
    df_IoU_imp["label"] = labels[0]
    df_IoU_poss = get_all_IoUs(os.path.join(img_dir, labels[1]), train_data=train_data)
    df_IoU_poss["label"] = labels[1]
    df = df_IoU_imp.append(df_IoU_poss)
    return df


def save_all_IoU(study_num=2, train_data=False):
    save_path = os.path.join(config.raw_dir, f"Study {study_num}")
    if train_data:
        save_path = os.path.join(save_path, "All IoU Values Training.xlsx")
    else:
        save_path = os.path.join(save_path, "All IoU Values Validation.xlsx")

    xl_writer = pd.ExcelWriter(save_path, engine="xlsxwriter")
    for net in config.DNNs:
        print(net)
        net_IoU = get_net_IoUs(net, study_num=study_num, train_data=train_data)
        net_IoU.to_excel(xl_writer, sheet_name=f"{net}")
    xl_writer.save()


def get_summary_result(df_in):
    lbls = df_in["label"].unique()
    c1_bg = np.mean(df_in[df_in["label"] == lbls[0]]["Background"])
    c2_bg = np.mean(df_in[df_in["label"] == lbls[1]]["Background"])
    df_out = pd.DataFrame([{
        f"ROI-Background {lbls[0]}": c1_bg,
        f"ROI-Background {lbls[1]}": c2_bg,
    }])
    if "Impossible Region" in df_in.columns:
        imp_roi = np.mean(df_in[df_in["label"] == "Impossible"]["Impossible Region"])
        df_out["ROI Impossible Region"] = imp_roi
    return df_out


def collate_IoU_summary(study_num=2):
    def summarise_file(file_fpath):
        df_all = pd.DataFrame([])
        for DNN in config.DNNs:
            xl = pd.read_excel(file_fpath, sheet_name=DNN)
            df = get_summary_result(xl)
            df.insert(0, "Network", DNN)
            df_all = df_all.append(df)
        return df_all

    save_fpath = os.path.join(config.results_basedir, f"Study {study_num}")
    xl_writer = pd.ExcelWriter(os.path.join(save_fpath, "IoU Values.xlsx"), engine="xlsxwriter")
    raw_path = os.path.join(config.raw_dir, f"Study {study_num}")
    train_fpath = os.path.join(raw_path, "All IoU Values Training.xlsx")
    val_fpath = os.path.join(raw_path, "All IoU Values Validation.xlsx")
    df_train = summarise_file(train_fpath)
    df_train.to_excel(xl_writer, sheet_name="Training")
    df_val = summarise_file(val_fpath)
    df_val.to_excel(xl_writer, sheet_name="Validation")
    xl_writer.save()


if __name__ == "__main__":
    for study_num in range(3):
        save_all_IoU(study_num=study_num, train_data=False)
        save_all_IoU(study_num=study_num, train_data=True)
        collate_IoU_summary(study_num)
