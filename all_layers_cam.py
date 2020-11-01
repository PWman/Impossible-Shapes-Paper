import os
import config
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from preprocessing import Preprocess
from net_utils import initialise_DNN
from gcam_utils import gcam_all_imgs, plot_cam_on_img
from analysis_utils import avg_gradcam
from more_utils import set_seed
plt.style.use("seaborn")


def get_axnet_gcam_all_layers(net_name):
    if "AlexNet" not in net_name:
        print("Please select AlexNet model")
        return
    model_dir = os.path.join(config.raw_dir,net_name,"models")
    # save_dir = os.path.join(config.raw_dir,f"{net_name} gcam all layers")
    # config.check_make_dir(save_dir)

    target_layers = [1,4,7,9,11]
    p = Preprocess(batch_size=1, augment=False, shuffle=False)
    if "sf=0.5" not in net_name:
        net, _ = initialise_DNN(net_name)
    else:
        net, _ = initialise_DNN(net_name[:net_name.index(net_name.split()[-1])-1])

    dir_names = []
    for t_lr in target_layers:
        print(f"Testing layer {t_lr}")
        net, _ = initialise_DNN("AlexNet")
        save_subdir = os.path.join(config.raw_dir, str(f"{net_name} layer{t_lr}"))
        dir_names.append(os.path.basename(save_subdir))
        config.check_make_dir(save_subdir)
        save_subdir = os.path.join(save_subdir, "gradCAM")
        config.check_make_dir(save_subdir)
        mask_dir = os.path.join(save_subdir, "masks")
        config.check_make_dir(mask_dir)
        gstat_dir = os.path.join(save_subdir, "scores")
        config.check_make_dir(gstat_dir)

        for seed in os.listdir(model_dir):
            print(f"Testing seed {seed}")
            set_seed(int(seed))

            net.load_state_dict(torch.load(os.path.join(model_dir, seed)))#file)))
            cam_arr, df = gcam_all_imgs(p, net, [str(t_lr)])
            np.save(os.path.join(mask_dir, f"{seed}"), cam_arr)
            df.to_csv(os.path.join(gstat_dir, f"{seed}.csv"))

    if "sf=0.5" in net_name:
        result_dir = os.path.join(config.expt2_dir,net_name[:net_name.index(net_name.split()[-1])-1])
    else:
        result_dir = os.path.join(config.expt1_dir, net_name)
    for dir in dir_names:
        avg_masks, avg_gstats = avg_gradcam(dir)
        result_subdir = os.path.join(result_dir, dir)
        config.check_make_dir(result_subdir)
        avg_gstats.to_csv(os.path.join(result_subdir, "GradCAM Info.csv"))
        img_paths = list(avg_gstats["img_path"])
        for img, mask in zip(img_paths, avg_masks):
            mask_img = plot_cam_on_img(img, mask)
            img_name = os.path.basename(img)
            mask_img.save(os.path.join(result_subdir, img_name))
