import os
import config
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms

plt.style.use("seaborn-bright")

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

def plot_and_save(results,fname):
    # PLOT ACCURACY
    plt.clf()
    plt.plot(results["epoch"], results["acc"], "b")
    plt.plot(results["epoch"], results["val_acc"], "r-")
    plt.title(fname + " Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training", "Validation"])
    plt.xlim(0, 99)
    plt.savefig(os.path.join(config.acc_dir, fname + ".png"))

    # PLOT LOSS
    plt.clf()
    plt.plot(results["epoch"], results["loss"], "b-")
    plt.plot(results["epoch"], results["val_loss"], "r-")
    plt.title(fname + " Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.xlim(0, 99)
    plt.savefig(os.path.join(config.loss_dir, fname + ".png"))

    return

def set_seed(seednum):
    torch.manual_seed(seednum)
    torch.cuda.manual_seed_all(seednum)
    np.random.seed(seednum)
    random.seed(seednum)
    return

def save_batch(X,bname,train_data):
    save_path = config.check_train_dir

    if train_data:
        save_path = os.path.join(save_path,"Train")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    else:
        save_path = os.path.join(save_path,"Test")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    for count, (img,lbl) in enumerate(X):
        save_name = os.path.join(save_path, bname + "img_" +
                                 str(count) + "_class_" +
                                 str(lbl.tolist()) + ".bmp")
        transforms.ToPILImage()(img).save(save_name)