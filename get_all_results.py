import os
import torch
import config
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from preprocessing import Preprocess
from gcam_utils import save_all_CAMS
from net_utils import initialise_DNN
from train_test_utils import train_test_network
from more_utils import plot_and_save, average_results, collate_all_results, set_seed, cm_arr_to_df
from analysis_utils import avg_save_net_results
# def get_single_result(net_name, net_save_loc=None, scale_factor=None):
#     print("Initialising network...")
#     net, opt = initialise_DNN(net_name)
#     net.to(config.device)
#     p_train = Preprocess(batch_size=config.batch_size,
#                          scale_factor=scale_factor)
#     print("Training network...")
#     results = train_net(p_train, net, opt)
#
#     if net_save_loc is not None:
#         torch.save(net.state_dict(), net_save_loc)
#
#     print("Getting confusion matrices...")
#     p_cmat = Preprocess(batch_size=config.batch_size, augment=False)
#     cm_t, cm_v = get_cm_result(p_cmat, net)
#
#     print("Getting GradCAM results...")
#     p_gcam = Preprocess(batch_size=1, shuffle=False, augment=False)
#     masks, df_cam = gcam_all_imgs(p_gcam, net, target_layer=config.target_layers[net_name])
#
#     return results, [cm_t, cm_v], [masks, df_cam]
#
#
# def save_results_all_seeds(net_name, save_models=False, scale_factor=None):
#     # if save_models:
#     #     if scale_factor is None:
#     #         model_dir = os.path.join(config.model_dir, f"{net_name}")
#     #     else:
#     #         model_dir = os.path.join(config.model_dir, f"{net_name}_sf{scale_factor}")
#     #     config.check_make_dir(model_dir)
#     if scale_factor is None:
#         raw_dir = os.path.join(config.raw_dir, f"{net_name}")
#         if save_models:
#             model_dir = os.path.join(config.model_dir, f"{net_name}")
#             config.check_make_dir(model_dir)
#     else:
#         if save_models:
#             model_dir = os.path.join(config.model_dir, f"{net_name}_sf{scale_factor}")
#             config.check_make_dir(model_dir)
#         raw_dir = os.path.join(config.raw_dir, f"{net_name}_sf{scale_factor}")
#     config.check_make_dir(raw_dir)
#     result_dir = os.path.join(raw_dir,"train_results")
#     cmat_dir = os.path.join(raw_dir,"confusion_matrices")
#     gcam_dir = os.path.join(raw_dir,"gradcam")
#     config.check_make_dir(result_dir)
#     config.check_make_dir(gcam_dir)
#     config.check_make_dir(cmat_dir)
#
#     for seed in range(config.num_seeds):
#         print(f"Testing seed {seed}...")
#         set_seed(seed)
#
#         net, opt = initialise_DNN(net_name)
#         net.to(config.device)
#         p_train = Preprocess(batch_size=config.batch_size,
#                              scale_factor=scale_factor)
#         print("Training network...")
#         results = train_net(p_train, net, opt)
#         p_cmat = Preprocess(batch_size=config.batch_size, augment=False)
#         cm_t, cm_v = get_cm_result(p_cmat, net)
#         p_gcam = Preprocess(batch_size=1, shuffle=False, augment=False)
#         masks, camstats = gcam_all_imgs(p_gcam, net, target_layer=config.target_layers[net_name])
#
#         # results, [cm_t, cm_v], [masks, camstats] = get_single_result(net_name)
#         results.to_csv(os.path.join(result_dir, f"train_results{seed}.csv"))
#         np.save(os.path.join(cmat_dir, f"cmats_train{seed}"), cm_t)
#         np.save(os.path.join(cmat_dir,f"cmats_val{seed}"), cm_v)
#         camstats.to_csv(os.path.join(gcam_dir,f"gcam_info{seed}.csv"))
#         np.save(os.path.join(gcam_dir,f"gcam_masks{seed}"), masks)
#     return
#

def get_result(net_name, scale_factor=None, save_raw=True):
    train_test_network(net_name, scale_factor=scale_factor)
    if scale_factor is not None:
        net_name = f"{net_name} sf={scale_factor}"
    avg_save_net_results(net_name)
    # if not save_raw:
    #     shutil.rmtree(os.path.join(config.raw_dir, net_name))
    # return



if __name__ == "__main__":
    net = "VGG11"
    # get_result(net)
    get_result(net, scale_factor=0.5)
    # nets = ["VGG16", "ResNet18", "ResNet50", "GoogLeNet"]
                                                                                                # nets = [f"{n} (pretrained)" for n in nets]
    # for net in nets:
    #     get_result(net)
    #     get_result(net, scale_factor=0.5)

    # for net in config.DNNs:
    #     if "pretrain" in net:
    #         get_result(net)
    #         get_result(net, scale_factor=0.5)
    # results, [cm_t, cm_v], [masks, camstats] = get_single_result("ResNet18 (pretrained)")

    # ALL_NETS = config.DNNs + [dnn + " (pretrained)" for dnn in config.DNNs]
    # ALL_NETS = ["ResNet50 (pretrained)","GoogLeNet (pretrained)"]
    # p = Preprocess()
    # for net_name in ALL_NETS:
    #     print(f"\nTesting {net_name}...\n")
    #
    #     for seed in range(config.num_seeds):
    #         print(f"Testing seed {seed}...")
    #         set_seed(seed)
    #         net, opt = make_models(net_name)
    #         result = train_net(p,net,opt)
    #         result.to_csv(os.path.join(
    #             config.raw_dir, net_name, f"{seed}.csv"
    #         ))
    #         torch.save(net.state_dict(), os.path.join(
    #             config.model_dir, net_name, str(seed) + ".pt"
    #         ))
    #         save_all_CAMS(net_name,config.target_layers[net_name])
    #
    # p = Preprocess(augment=False)
    # train_dir = os.path.join(config.cm_dir, "Training")
    # val_dir = os.path.join(config.cm_dir, "Validation")
    # config.check_make_dir(train_dir)
    # config.check_make_dir(val_dir)
    #
    # writer_t = pd.ExcelWriter(
    #     os.path.join(config.cm_dir, "All_Train_Confusion_Matrices.xlsx"),
    #     engine="xlsxwriter"
    # )
    # writer_v = pd.ExcelWriter(
    #     os.path.join(config.cm_dir, "All_Validation_Confusion_Matrices.xlsx"),
    #     engine="xlsxwriter"
    # )
    #
    # for net_name in ALL_NETS:
    #     print(f"Testing {net_name}...")
    #     cm_t, cm_v = get_cms_all(p, net_name)
    #     df_t = cm_arr_to_df(cm_t)
    #     df_v = cm_arr_to_df(cm_v)
    #     df_t.to_csv(os.path.join(train_dir, f"{net_name}.csv"))
    #     df_v.to_csv(os.path.join(val_dir, f"{net_name}.csv"))
    #     df_t.to_excel(writer_t, sheet_name=net_name)
    #     df_v.to_excel(writer_v, sheet_name=net_name)
    #
    # writer_t.save()
    # writer_v.save()
    # plt.style.use("seaborn")  # -bright")
    # ALL_SCORES = pd.DataFrame(columns=["name", "acc", "acc_std",
    #                                    "val_acc", "val_acc_std",
    #                                    "loss", "loss_std",
    #                                    "val_loss", "val_loss_std"])
    # RESULT_DICT = {}
    # for net_name in ALL_NETS:
    #     avg_result, net_scores = average_results(os.path.join(config.raw_dir, net_name))
    #     avg_result.to_csv(os.path.join(config.avg_dir, f"{net_name}.csv"))
    #     plot_and_save(avg_result, net_name)
    #     score = pd.DataFrame([{
    #         "name": net_name, "acc": np.mean(net_scores["acc"]), "acc_std": np.std(avg_result["acc"]),
    #         "val_acc": np.mean(net_scores["val_acc"]), "val_acc_std": np.std(avg_result["val_acc"]),
    #         "loss": np.mean(net_scores["loss"]), "loss_std": np.std(avg_result["loss"]),
    #         "val_loss": np.mean(net_scores["val_loss"]), "val_loss_std": np.std(avg_result["val_loss"]),
    #     }])
    #     ALL_SCORES = ALL_SCORES.append(score, sort=False)
    #     RESULT_DICT[net_name] = avg_result
    #     # save_all_CAMS(net_name,config.target_layers[net_name])
    # ALL_SCORES.to_csv(os.path.join(config.table_dir, "All_DNN_Scores.csv"))

    # plt.close("all")
    # plot_all_together(RESULT_DICT)
    # ALL_NETS = config.DNNs
    # ALL_NETS = ["ResNet50"]
    # # ALL_NETS = [dnn + " (pretrained)" for dnn in config.DNNs]
    # for net_name in ALL_NETS:
    #     save_all_CAMS(net_name,config.target_layers[net_name])
    #
