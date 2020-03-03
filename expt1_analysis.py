import os
import config
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from analysis_functions import average_results, fit_and_plot, plot_all_loss

plt.style.use("seaborn-bright")

all_scores = pd.DataFrame(columns=["name","acc","acc_std",
                                   "val_acc","val_acc_std",
                                   "loss","loss_std",
                                   "val_loss","val_loss_std"])
result_dict = {}
for net_name in config.net_names:
    avg_result, net_scores = average_results(config.raw_dir_1, net_name, config.num_seeds)
    avg_result.to_csv(os.path.join(config.avg_dir_1, net_name + ".csv"))
    fit_and_plot(avg_result, config.graph_dir_1, net_name.replace("_pt", " Pretrained"))

    net_scores = net_scores.dropna()
    # print(net_scores)
    score = pd.DataFrame([{
        "name": net_name, "acc": np.mean(net_scores["acc"]), "acc_std": np.std(avg_result["acc"]),
        "val_acc": np.mean(net_scores["val_acc"]), "val_acc_std": np.std(avg_result["val_acc"]),
        "loss": np.mean(net_scores["loss"]), "loss_std": np.std(avg_result["loss"]),
        "val_loss": np.mean(net_scores["val_loss"]), "val_loss_std": np.std(avg_result["val_loss"]),
    }])
    all_scores = all_scores.append(score, sort=False)
    result_dict[net_name] = avg_result


all_scores.to_csv(os.path.join(config.avg_dir_1, "Network Scores.csv"))

all_scores["legend"] = all_scores["name"].replace({
    "_pt": ""}, regex=True)

scratch_scores = all_scores.iloc[:4]
scratch_path = os.path.join(config.graph_dir_1, "Validation Losses")
plot_all_loss(result_dict,scratch_scores,scratch_path)

pt_scores = all_scores.iloc[4:]
pt_path = os.path.join(config.graph_dir_1,"Validation Losses Pretrained")
plot_all_loss(result_dict,pt_scores,pt_path)

plt.close()