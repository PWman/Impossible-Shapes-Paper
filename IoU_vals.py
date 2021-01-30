import os
import config
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from initialise_nets import initialise_DNN
from process_results import avg_gradcam
from skimage.segmentation import flood_fill
from scipy.ndimage.filters import gaussian_filter
from scipy import stats

plt.style.use("seaborn")


def get_ffil_img(img_path):
    img = Image.open(img_path).convert("1")
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


def get_cam_mask(net_name, img_name):
    all_masks, df_camstats = avg_gradcam(net_name)

    img_idx = int(df_camstats[df_camstats["img_name"] == img_name].index[0])
    mask = all_masks[img_idx]
    mask = mask - np.amin(mask)
    mask = mask / np.amax(mask)

    return mask


# def get_background_IoU_mask(net_name, img_name):
#     img_path = os.path.join(config.prepro_dir,"Validation","Possible",img_name)
#     cam_mask = get_cam_mask(net_name,img_name)
#     ffil_img = get_ffil_img(img_path)
#     IoU_mask = np.multiply(ffil_img,cam_mask)
#     # IoU_mask = IoU_mask/np.amax(IoU_mask)
#     return IoU_mask

def get_imposs_heatmap(coordinates):
    heatmap, xedges, yedges = np.histogram2d([coordinates[0]], [coordinates[1]], bins=(range(225), range(225)))
    heatmap = heatmap.T
    heatmap = gaussian_filter(heatmap, 24)
    heatmap = heatmap / np.amax(heatmap)
    heatmap[np.where(heatmap >= 0.1)] = 1
    heatmap[np.where(heatmap < 0.1)] = 0
    return heatmap


#
# def get_impossibleROI_IoU_mask(net_name, img_name, ROI_heatmap):
#     cam_mask = get_cam_mask(net_name,img_name)
#     IoU_mask = np.multiply(ROI_heatmap,cam_mask)
#     IoU_mask = IoU_mask/np.amax(IoU_mask)
#     return IoU_mask
#
#
# def get_IoU_ratio(mask,threshold,return_img=False):
#     all_regions = np.where(mask != 0)
#     target_regions = np.where(mask > threshold)
#     size_all_regions = len(all_regions[0])
#     size_target_regions = len(target_regions[0])
#     ratio = size_target_regions/size_all_regions
#     if return_img:
#         new_mask = np.zeros((mask.shape))
#         new_mask[target_regions] = 1
#         return ratio,new_mask
#     else:
#         return ratio
val_imgs = os.listdir(os.path.join(config.prepro_dir, "Validation", "Possible"))

imposs_coordinate = {"impossa1": [147, 49],
                     "impossb5": [123, 137],
                     "impossc2": [77, 52],
                     "impossg2": [81, 173],
                     "impossg8": [106, 122],
                     "impossh2": [101, 185],
                     "impossi1": [177, 126],
                     "impossi5": [88, 92]
                     }


def get_IoU(region, gcam):
    if np.amin(gcam) != 0:
        gcam = gcam - np.amin(gcam)
    if np.amax(gcam) != 1:
        gcam = gcam / np.amax(gcam)

    ROI_mask = np.multiply(gcam, region)
    # inv_ROI_mask = np.multiply(gcam,1-region)
    return np.sum(ROI_mask) / np.sum(gcam)

    # s1 = np.sum(ROI_mask)
    # s2 = np.sum(region)
    # r1 = s1/s2
    # s3 = np.sum(inv_ROI_mask)
    # s4 = np.sum(1-region)
    # r2 = s3/s4
    #
    # tot_frac = np.sum(gcam)/(224*224)
    #
    # IoU = r1*(1-r2)
    #
    # return IoU/tot_frac


def get_all_IoUs():
    val_imgs_POSS = os.listdir(os.path.join(config.prepro_dir, "Validation", "Impossible"))

    for expt_dir in [config.expt1_dir, config.expt2_dir]:
        excel_writer = pd.ExcelWriter(os.path.join(expt_dir, "IoU_vals.xlsx"),
                                      engine="xlsxwriter")
        for net in config.DNNs:
            df_IoU = pd.DataFrame([])
            for img_name in val_imgs:
                print(img_name)
                img_file = os.path.join(config.prepro_dir, "Validation", "Possible", img_name)
                img = plt.imread(img_file)

                coordinates = imposs_coordinate[img_name.replace(".bmp", "")]
                ROI_mask = get_imposs_heatmap(coordinates)
                background_mask = get_ffil_img(os.path.join(config.prepro_dir, "Validation", "Possible", img_name))
                cam_mask = get_cam_mask(net, img_name)

                ROI_IoU = get_IoU(ROI_mask, cam_mask)
                background_IoU = get_IoU(background_mask, cam_mask)

                save_name = img_name.replace(".bmp", "")
                df = pd.DataFrame([{"image": save_name, "ROI_IoU": ROI_IoU, "background_IoU": background_IoU}])
                df_IoU = df_IoU.append(df)
                df_IoU.to_excel(excel_writer, sheet_name=net)

            for img_name in val_imgs_POSS:
                img_file = os.path.join(config.prepro_dir, "Validation", "Impossible", img_name)
                img = plt.imread(img_file)

                background_mask = get_ffil_img(os.path.join(config.prepro_dir, "Validation", "Impossible", img_name))
                cam_mask = get_cam_mask(net, img_name)

                background_IoU = get_IoU(background_mask, cam_mask)

                save_name = img_name.replace(".bmp", "")
                df = pd.DataFrame([{"image": save_name, "ROI_IoU": np.nan, "background_IoU": background_IoU}])
                df_IoU = df_IoU.append(df)
                df_IoU.to_excel(excel_writer, sheet_name=net)

        excel_writer.save()
    # plt.imshow(img)


def get_background_num(ddir):
    df_all = pd.DataFrame([])
    for file in os.listdir(ddir):
        fpath = os.path.join(ddir, file)
        ffill_img = get_ffil_img(fpath)
        df = pd.DataFrame([{
            "img_name": file.replace(".bmp", ""),
            "file_path": fpath,
            "background_size": np.sum(ffill_img) / (224 * 224)
        }])
        df_all = df_all.append(df)
    return df_all

def get_background_study1():
    train_dir = os.path.join(config.prepro_dir, "Training")
    df_all = pd.DataFrame([])
    categories = ("Possible", "Impossible")
    for idx, subdir in enumerate(os.listdir(train_dir)):
        df = get_background_num(os.path.join(train_dir, subdir))
        df["category"] = categories[idx]
        df_all = df_all.append(df)
    return df_all

def plot_background_ttest(df_all):
    poss_backgrounds = df_all[df_all["category"] == "Possible"].background_size
    imp_backgrounds = df_all[df_all["category"] == "Impossible"].background_size
    mu_poss, std_poss = stats.norm.fit(poss_backgrounds)
    mu_imp, std_imp = stats.norm.fit(imp_backgrounds)

    plt.hist(poss_backgrounds, color="r", alpha=0.5)
    plt.hist(imp_backgrounds, color="b", alpha=0.5)
    xmin, xmax = plt.xlim()

    x = np.linspace(xmin, xmax, 100)
    pdf_poss = stats.norm.pdf(x, mu_poss, std_poss)
    pdf_imp = stats.norm.pdf(x, mu_imp, std_imp)
    plt.plot(x,pdf_poss,color="r")
    plt.plot(x,pdf_imp,color="b")

    _,poss_ntest = stats.normaltest(poss_backgrounds)
    _,imp_ntest = stats.normaltest(imp_backgrounds)
    leg = [
        f"Possible; mu={round(mu_poss,2)},std={round(std_poss,2)}; normtest p_val={round(poss_ntest,3)}",
        f"Impossible; mu={round(mu_imp,2)},std={round(std_imp,2)}; normtest p_val={round(imp_ntest,3)}"
    ]
    plt.legend(leg)
    tt_tval,tt_pval = stats.ttest_ind(poss_backgrounds,imp_backgrounds)
    # title = f"T-Test Statistic={round(tt_tval,2)}, p-value={tt_pval:e}"
    title = f"T-Test Statistic={round(tt_tval,2)}, p-value={round(tt_pval,5)}"

    plt.title(title,fontsize=14)

# background_mask = get_background_IoU_mask("AlexNet",img_name)
#
#
# background_IoU,background_target_mask = get_IoU_ratio(background_mask,threshold=0.5,return_img=True)
# ROI_IoU,ROI_target_mask = get_IoU_ratio(ROI_mask,threshold=0.5,return_img=True)
#

# SAVE_IoU_IMAGES
# plt.figure()
# plt.imshow(img)
# plt.imshow(ROI_target_mask,cmap="jet",alpha=0.5)
# plt.savefig(os.path.join(r"C:\Users\Peter\Documents\chosen_impossible_locs",f"{save_name}_ROI"))
# plt.figure()
# plt.imshow(img)
# plt.imshow(background_target_mask,cmap="jet",alpha=0.5)
# plt.savefig(os.path.join(r"C:\Users\Peter\Documents\chosen_impossible_locs",f"{save_name}_background"))
#
# SAVE_HEATMAP
# plt.imshow(img)
# plt.imshow(ROI_hmap,cmap="jet",alpha=0.5)
# plt.savefig(os.path.join(r"C:\Users\Peter\Documents\chosen_impossible_locs",f"{save_name}_REALROI"))

# plt.scatter(coordinates[0],coordinates[1])
# plt.imshow(heatmap,alpha=0.5,cmap="jet")
# plt.title(img_name)
# save_name = img_name.replace(".bmp", "")
# plt.savefig(os.path.join(r"C:\Users\Peter\Documents\chosen_impossible_locs",f"{save_name}_hmap"))
#
# CHARLES DATA
#
# val_imgs = os.listdir(os.path.join(config.prepro_dir,"Validation","Possible"))
# val_img_paths = [os.path.join(config.prepro_dir,"Validation","Possible",i) for i in val_imgs]
# val_imgs = [i.replace(".bmp","") for i in val_imgs]
# img_path_mapping = dict(zip(val_imgs,val_img_paths))
#
# df_imp_locs = pd.read_csv(os.path.join(config.shapes_basedir,"impossible_locations.csv"))
#
# CHARLES DATA
# for img_name in df_imp_locs["shape"].unique():
#     data = df_imp_locs[df_imp_locs["shape"] == img_name]
#
#     x = [i*224/350 for i in data["x"]]
#     y = [i*224/350 for i in data["y"]]
#
#     img = plt.imread(img_path_mapping[img_name])
#     plt.figure()
#     plt.imshow(img,alpha=0.5)
#     plt.scatter(x,y)
#
# DISS DATA
# ROI_dir = r"C:\Users\Peter\Desktop\ROI"
# all_data = pd.DataFrame([])
# for file in os.listdir(ROI_dir):
#     if "csv" in file:
#         df = pd.read_csv(os.path.join(ROI_dir,file))
#         df = df.drop(columns="response_time_mouse_response")
#         all_data = all_data.append(df,ignore_index=True)
# all_data = all_data[~all_data.image.str.contains("_")]

# img_mask_dir = os.path.join(config.shapes_basedir,"backgrounds")
# config.check_make_dir(img_mask_dir)
#
# val_dir = os.path.join(config.prepro_dir,"Validation")
#
# for subdir in os.listdir(val_dir):
#     save_dir = os.path.join(img_mask_dir,subdir)
#     config.check_make_dir(save_dir)
#     for img in os.listdir(os.path.join(val_dir,subdir)):
#         ffil_arr = get_ffil_img(os.path.join(val_dir,subdir,img))
#         plt.figure()
#         plt.imshow(ffil_arr)
#


# for img in all_data.image.unique():
#     if img in val_imgs:
#         data = (all_data[all_data.image==img])
#         x = list(data["cursor_x"])
#         y = list(data["cursor_y"])
#         x = [int(i*(224/768) + (224/2)) for i in x]
#         y = [int(i*(224/768) + (224/2)) for i in y]
#         heatmap, xedges, yedges = np.histogram2d(x, y, bins=(range(225), range(225)))
#         heatmap = heatmap.T
#         heatmap = gaussian_filter(heatmap.T, 16)
#         if "imp" in img:
#             img_path = os.path.join(config.prepro_dir,"Validation","Possible",img) ########
#         else:
#             img_path = os.path.join(config.prepro_dir,"Validation","Impossible",img) #######
#         plt.figure()
#         shape_img = plt.imread(img_path)
#         plt.imshow(shape_img)
#         plt.imshow(heatmap,alpha=0.5,cmap="jet")
