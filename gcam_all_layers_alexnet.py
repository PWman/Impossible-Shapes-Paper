import os
import config
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from preprocessing import Preprocess
from initialise_nets import initialise_DNN
from gcam_utils import gcam_all_imgs, plot_cam_on_img
from process_results import avg_gradcam
from more_utils import set_seed

plt.style.use("seaborn")


def get_result_target_layer(p, net, target_layer, model_dir, save_dir):
    config.check_make_dir(save_dir)
    mask_dir = os.path.join(save_dir, "masks")
    config.check_make_dir(mask_dir)
    gstat_dir = os.path.join(save_dir, "scores")
    config.check_make_dir(gstat_dir)

    for seed in range(config.num_seeds):
        print(f"Testing seed {seed}")
        set_seed(int(seed))

        net.load_state_dict(torch.load(os.path.join(model_dir, f"{seed}.pt")))
        cam_arr, df = gcam_all_imgs(p, net, [str(target_layer)])
        np.save(os.path.join(mask_dir, f"{seed}"), cam_arr)
        df.to_csv(os.path.join(gstat_dir, f"{seed}.csv"))


def save_average_target_layer(net_name_dir, final_save_dir, study_num=2, train_data=False):
    config.check_make_dir(final_save_dir)
    avg_masks, avg_gstats = avg_gradcam(net_name_dir, study_num=study_num,train_data=train_data)
    avg_gstats.to_csv(os.path.join(final_save_dir, "GradCAM Info.csv"))
    img_paths = list(avg_gstats["img_path"])
    for img, mask in zip(img_paths, avg_masks):
        mask_img = plot_cam_on_img(img, mask)
        img_name = os.path.basename(img)
        mask_img.save(os.path.join(final_save_dir, img_name))


def get_axnet_gcam_all_layers(study_num=2, pretrained=True, train_data=False):
    if pretrained:
        net_name = "AlexNet (pretrained)"
    else:
        net_name = "AlexNet"
    # if study_1:
    #     prepro_dir = os.path.join(config.prepro_dir, "Study 1")
    #     raw_result_dir = os.path.join(config.raw_dir_expt1)
    #     final_savedir = os.path.join(config.expt1_dir, "All Layers GradCAM")
    # else:
    #     prepro_dir = os.path.join(config.prepro_dir, "Study 2")
    #     raw_result_dir = os.path.join(config.raw_dir_expt2)
    #     final_savedir = os.path.join(config.expt2_dir, "All Layers GradCAM")
    prepro_dir = os.path.join(config.prepro_dir, f"Study {study_num}")
    raw_result_dir = os.path.join(config.raw_dir, f"Study {study_num}")
    final_savedir = os.path.join(config.results_basedir, f"Study {study_num}", "All Layers GradCAM")

    if train_data:
        train_vs_val_name = "training"
    else:
        train_vs_val_name = "validation"

    config.check_make_dir(final_savedir)
    final_savedir = os.path.join(final_savedir, net_name)
    config.check_make_dir(final_savedir)

    target_layers = [1, 4, 7, 9, 11]
    p = Preprocess(prepro_dir, batch_size=1, augment=False, shuffle=False)
    net, _ = initialise_DNN(net_name)

    model_dir = os.path.join(raw_result_dir, net_name, "models")
    for t_lr in target_layers:
        print(f"Testing layer {t_lr}")
        net_dir_name = f"{net_name} layer_{t_lr}"
        save_raw_subdir = os.path.join(raw_result_dir, net_dir_name)
        config.check_make_dir(save_raw_subdir)
        save_raw_subdir = os.path.join(save_raw_subdir, "gradCAM")
        config.check_make_dir(save_raw_subdir)
        save_raw_subdir = os.path.join(save_raw_subdir, train_vs_val_name)
        get_result_target_layer(p, net, t_lr, model_dir, save_raw_subdir)

        final_save_subdir = os.path.join(final_savedir, f"Layer_{t_lr}")
        config.check_make_dir(final_save_subdir)
        final_save_subdir = os.path.join(final_save_subdir, train_vs_val_name)
        save_average_target_layer(net_dir_name, final_save_subdir, study_num=study_num,train_data=train_data)


if __name__ == "__main__":
    for study_num in range(3):
        get_axnet_gcam_all_layers(study_num=study_num, pretrained=False, train_data=True)
        get_axnet_gcam_all_layers(study_num=study_num, pretrained=True, train_data=True)
        get_axnet_gcam_all_layers(study_num=study_num, pretrained=False, train_data=False)
        get_axnet_gcam_all_layers(study_num=study_num, pretrained=True, train_data=False)