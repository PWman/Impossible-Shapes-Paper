import os
import config
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def cm_arr_to_df(arr):
    df = pd.DataFrame(arr)
    df.columns = ["Pred Imposs", "Pred Poss"]
    df.index = ["Actual Imposs", "Actual Poss"]
    return df

def average_results(net_arch):
    avg_results = []
    scores = pd.DataFrame(columns=["acc", "val_acc", "loss", "val_loss"])
    results_path = os.path.join(config.raw_dir, net_arch)
    for file in os.listdir(results_path):
        # print(file)
        # print(os.path.join(results_path, file))
        result = pd.read_csv(os.path.join(results_path, file), index_col=0)
        end_result = result[result["epoch"] == max(result["epoch"])]#.drop(columns=["epoch"])
        scores = scores.append(end_result.drop(columns=["epoch"]))
        avg_results.append(result)

    return sum(avg_results)/len(avg_results), scores

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

    return

def collate_all_results(expt_dir):

    # cm_write = pd.ExcelWriter()
    leg1 = []
    leg2 = []
    for net_name in os.listdir(expt_dir):
        if ".png" not in net_name:
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
    plt.savefig(os.path.join(expt_dir, "Validation Accuracies (no pretraining).png"))
    plt.figure(2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(expt_dir, "Validation Losses (no pretraining).png"))

    plt.figure(3)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(expt_dir, "Validation Accuracies (with pretraining).png"))
    plt.figure(4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(expt_dir, "Validation Losses (with pretraining)"))
    return

if __name__ == "__main__":
    collate_all_results(config.expt2_dir)