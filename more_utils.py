import os
import config
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
from math import floor, log10
from skimage.segmentation import flood_fill
from torchvision import transforms


plt.style.use("seaborn")

def set_seed(seednum):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seednum)
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    random.seed(seednum)
    return

def save_batch(imgs,lbls,bname):
    # save_path = config.check_train_dir
    #
    # if train_data:
    #     save_path = os.path.join(save_path,"Train")
    #     if not os.path.isdir(save_path):
    #         os.mkdir(save_path)
    # else:
    #     save_path = os.path.join(save_path,"Test")
    #     if not os.path.isdir(save_path):
    #         os.mkdir(save_path)
    #
    for count, (img,lbl) in enumerate(zip(imgs,lbls)):
        img_name = f"{bname}_img_{count}_class_{lbl.tolist()}.bmp"
        save_name = os.path.join(config.check_train_dir,img_name)
        transforms.ToPILImage()(img).save(save_name)

def round_sig(x,sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)



def cm_arr_to_df(arr):
    df = pd.DataFrame(arr)
    df.columns = ["Pred Imposs", "Pred Poss"]
    df.index = ["Actual Imposs", "Actual Poss"]
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
    # ffil_arr = 1 - ffil_arr

    return ffil_arr


# def average_results(net_arch):
#     avg_results = []
#     scores = pd.DataFrame(columns=["acc", "val_acc", "loss", "val_loss"])
#     results_path = os.path.join(config.raw_dir, net_arch)
#     for file in os.listdir(results_path):
#         # print(file)
#         # print(os.path.join(results_path, file))
#         result = pd.read_csv(os.path.join(results_path, file), index_col=0)
#         end_result = result[result["epoch"] == max(result["epoch"])]#.drop(columns=["epoch"])
#         scores = scores.append(end_result.drop(columns=["epoch"]))
#         avg_results.append(result)
#
#     return sum(avg_results)/len(avg_results), scores


# def get_dir_sizes(dir):
#     for dirpath, dirnames, filenames in os.walk(dir):
#         models_size = 0
#         results_size = 0
#         total_size = 0
#         for f in filenames:
#             print(f)
#             fp = os.path.join(dirpath, f)
#             # skip if it is symbolic link
#             if not os.path.islink(fp):
#                 total_size += os.path.getsize(fp)
#                 if "models" in fp:
#                     # print(fp)
#
#                     models_size += os.path.getsize(fp)
#                 else:
#                     results_size += os.path.getsize(fp)
#         # return results_size,




# if __name__ == "__main__":

    # collate_cmats(config.expt1_dir)
    # collate_cmats(config.expt2_dir)
    # collate_all_results()
    # collate_all_results(scale_factor=0.5)
    # graph_all_results(config.expt1_dir)
    # graph_all_results(config.expt2_dir)