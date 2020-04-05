import os
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis_functions import average_results, plot_and_save

plt.style.use("seaborn")#-bright")

# raw_dir = os.path.join(config.results_basedir,config.result_dir_names[0])
# avg_dir = os.path.join(config.results_basedir,config.result_dir_names[1])
# graph_dir = os.path.join(config.results_basedir,config.result_dir_names[2])

ALL_SCORES = pd.DataFrame(columns=["name","acc","acc_std",
                                   "val_acc","val_acc_std",
                                   "loss","loss_std",
                                   "val_loss","val_loss_std"])
RESULT_DICT = {}
pretrained_DNNs = [n + " (pretrained)" for n in config.DNNs]
ALL_NET_NAMES = config.DNNs + pretrained_DNNs

for net_name in ALL_NET_NAMES:
    avg_result, net_scores = average_results(os.path.join(config.raw_dir,net_name))

    avg_result.to_csv(os.path.join(config.avg_dir, f"{net_name}.csv"))

    # fit_and_plot(avg_result, config.graph_dir_1, net_name.replace("_pt", " Pretrained"))
    plot_and_save(avg_result, net_name)
    # net_scores = net_scores.dropna()
    # print(net_scores)
    score = pd.DataFrame([{
        "name": net_name, "acc": np.mean(net_scores["acc"]), "acc_std": np.std(avg_result["acc"]),
        "val_acc": np.mean(net_scores["val_acc"]), "val_acc_std": np.std(avg_result["val_acc"]),
        "loss": np.mean(net_scores["loss"]), "loss_std": np.std(avg_result["loss"]),
        "val_loss": np.mean(net_scores["val_loss"]), "val_loss_std": np.std(avg_result["val_loss"]),
    }])

    ALL_SCORES = ALL_SCORES.append(score, sort=False)
    RESULT_DICT[net_name] = avg_result

ALL_SCORES.to_csv(os.path.join(config.table_dir,"All_DNN_Scores.csv"))
plt.close("all")

for net_key, result in RESULT_DICT.items():
    if "pretrained" not in net_key:
        plt.figure(1)
        plt.plot(result["epoch"], result["val_acc"])
        plt.figure(2)
        plt.plot(result["epoch"], result["val_loss"])
    else:
        plt.figure(3)
        plt.plot(result["epoch"], result["val_acc"])
        plt.figure(4)
        plt.plot(result["epoch"], result["val_loss"])

plt.figure(1)
t1 = "All Validation Accuracy without Pretraining"
plt.title(t1)
plt.ylabel("Validation Accuracy")
plt.xlabel("Epoch")
plt.legend(config.DNNs)
plt.savefig(os.path.join(config.graph_dir,t1))

plt.figure(2)
t2 = "All Validation Loss without Pretraining"
plt.title(t2)
plt.ylabel("Validation Loss")
plt.xlabel("Epoch")
plt.legend(config.DNNs)
plt.ylim(top=1.5)

plt.savefig(os.path.join(config.graph_dir,t2))

plt.figure(3)
t3 = "All Validation Accuracy with Pretraining"
plt.title(t3)
plt.ylabel("Validation Accuracy")
plt.xlabel("Epoch")
plt.legend(config.DNNs)
plt.savefig(os.path.join(config.graph_dir,t3))

plt.figure(4)
t4 = "All Validation Loss with Pretraining"
plt.title(t4)
plt.ylabel("Validation Loss")
plt.xlabel("Epoch")
plt.legend(config.DNNs)
plt.ylim(top=1)
plt.savefig(os.path.join(config.graph_dir,t4))
