import os
import config
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from net_utils import initialise_DNN
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


def cm_arr_to_df(arr):
    df = pd.DataFrame(arr)
    df.columns = ["Pred Imposs", "Pred Poss"]
    df.index = ["Actual Imposs", "Actual Poss"]
    return df

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

def plot_and_save(results,fpath):
    # PLOT ACCURACY

    plt.clf()
    plt.plot(results["epoch"], results["acc"], "b")
    plt.plot(results["epoch"], results["val_acc"], "r-")
    # plt.title(fpath + " Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training", "Validation"])
    plt.xlim(0, config.num_epochs-1)
    plt.savefig(os.path.join(fpath, "Accuracy.png"))

    # PLOT LOSS
    plt.clf()
    plt.plot(results["epoch"], results["loss"], "b-")
    plt.plot(results["epoch"], results["val_loss"], "r-")
    # plt.title(fname + " Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.xlim(0, config.num_epochs-1)
    plt.savefig(os.path.join(fpath, "Loss.png"))
    plt.close()
    return

def graph_all_results(expt_dir):

    # cm_write = pd.ExcelWriter()
    leg1 = []
    leg2 = []
    for net_name in os.listdir(expt_dir):
        # if ".png" not in net_name:
        if os.path.isdir(os.path.join(expt_dir, net_name)):

            for file in os.listdir(os.path.join(expt_dir, net_name)):
                if "Train Results.csv" in file:
                    result = pd.read_csv(os.path.join(expt_dir, net_name, file))
                    if "pretrained" not in net_name:
                        plt.figure(1)
                        plt.plot(result["epoch"], result["val_acc"])

                        # t1 = "All Validation Accuracy without Pretraining"
                        # plt.title(t1)
                        # plt.ylabel("Validation Accuracy")
                        # plt.xlabel("Epoch")
                        # plt.legend(net_name)
                        plt.figure(2)
                        plt.plot(result["epoch"], result["val_loss"])
                        leg1.append(net_name)

                        # plt.title(t2)
                        # plt.ylabel("Validation Loss")
                        # plt.xlabel("Epoch")
                        # plt.legend(config.DNNs)
                        # plt.ylim(top=1.5)
                    else:
                        plt.figure(3)
                        plt.plot(result["epoch"], result["val_acc"])
                        plt.figure(4)
                        t2 = "All Validation Loss Pretraining"
                        plt.plot(result["epoch"], result["val_loss"])
                        leg2.append(net_name.split(" ")[0])

                    # t3 = "All Validation Accuracy with Pretraining"
                    # plt.title(t3)
                    # plt.ylabel("Validation Accuracy")
                    # plt.xlabel("Epoch")
                    # plt.legend(config.DNNs)
                    # # plt.savefig(os.path.join(config.graph_dir, t3))
                    #
                    # plt.figure(4)
                    # t4 = "All Validation Loss with Pretraining"
                    # plt.title(t4)
                    # plt.ylabel("Validation Loss")
                    # plt.xlabel("Epoch")
                    # plt.legend(config.DNNs)
                    # plt.ylim(top=1)
                    #
                    # plt.savefig(os.path.join(config.graph_dir, t4))
    # plt.savefig(os.path.join(expt_dir, t1))
    # plt.savefig(os.path.join(config.graph_dir, t2))
    plt.figure(1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(leg1)
    plt.savefig(os.path.join(expt_dir, "Validation Accuracies (no pretraining).png"))
    plt.figure(2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(leg2)
    plt.savefig(os.path.join(expt_dir, "Validation Losses (no pretraining).png"))

    plt.figure(3)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(expt_dir, "Validation Accuracies (with pretraining).png"))
    plt.figure(4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(expt_dir, "Validation Losses (with pretraining)"))
    plt.close("all")
    return


def collate_all_results(scale_factor=None):
    net_names = config.DNNs
    if scale_factor is not None:
        net_names = [f"{n} sf={scale_factor}" for n in net_names]

    collated_table = pd.DataFrame([])
    for net in net_names:
        result_path = os.path.join(config.raw_dir, net, "train_results")
        net_results = pd.DataFrame([])
        for file in os.listdir(result_path):
            df = pd.read_csv(os.path.join(result_path, file))
            score = {
                "acc": df[df["epoch"] == 99]["acc"].values[0], # select acc at final epoch
                "acc_std": np.std(df["acc"]),
                "val_acc": df[df["epoch"] == 99]["val_acc"].values[0],
                "val_acc_std": np.std(df["val_acc"]),
                "loss": df[df["epoch"] == 99]["loss"].values[0],
                "loss_std": np.std(df["loss"]),
                "val_loss": df[df["epoch"] == 99]["val_loss"].values[0],
                "val_loss_std": np.std(df["val_acc"])
            }
            net_results = net_results.append(pd.DataFrame([score]))
        result = pd.DataFrame(np.mean(net_results)).transpose()
        if scale_factor is None:
            result["net_name"] = net
        else:
            result["net_name"] = net[:net.find(f"sf={scale_factor}") - 1]
        collated_table = collated_table.append(result)

    if scale_factor is None:
        collated_table.to_csv(os.path.join(config.expt1_dir, "DNN perfomance summary.csv"))
    elif scale_factor == 0.5:
        collated_table.to_csv(os.path.join(config.expt2_dir, "DNN perfomance summary.csv"))

    return collated_table

def collate_cmats(expt_dir):

    train_writer = pd.ExcelWriter(os.path.join(expt_dir, "Confusion Matrices (train images).xlsx"),
                                  engine="xlsxwriter")
    val_writer = pd.ExcelWriter(os.path.join(expt_dir, "Confusion Matrices (validation images).xlsx"),
                                engine="xlsxwriter")
    for net in os.listdir(expt_dir):
        if net in config.DNNs:
            for file in os.listdir(os.path.join(expt_dir, net)):
                if file == "Confusion Matrices.xlsx":
                    rpath = os.path.join(expt_dir, net, file)
                    train_result = pd.read_excel(rpath, "Training", index_col=0)
                    val_result = pd.read_excel(rpath, "Validation", index_col=0)

                    train_result.to_excel(train_writer, sheet_name=net)
                    val_result.to_excel(val_writer, sheet_name=net)

    train_writer.save()
    val_writer.save()
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




if __name__ == "__main__":
    collate_cmats(config.expt1_dir)
    collate_cmats(config.expt2_dir)
    collate_all_results()
    collate_all_results(scale_factor=0.5)
    graph_all_results(config.expt1_dir)
    graph_all_results(config.expt2_dir)