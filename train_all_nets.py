import os
import torch
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import Preprocess
from gcam_utils import save_all_CAMS
from net_utils import make_models, train_net, get_cm_result
from more_utils import set_seed, plot_and_save, average_results, plot_all_together


if __name__ == "__main__":
    # ALL_NETS = config.DNNs + [dnn + " (pretrained)" for dnn in config.DNNs]
    ALL_NETS = ["ResNet50"]
    p = Preprocess()
    for net_name in ALL_NETS:
        print(f"\nTesting {net_name}...\n")

        for seed in range(config.num_seeds):
            print(f"Testing seed {seed}...")
            set_seed(seed)
            net, opt = make_models(net_name)
            result = train_net(p,net,opt)
            result.to_csv(os.path.join(
                config.raw_dir, net_name, f"{seed}.csv"
            ))
            torch.save(net.state_dict(), os.path.join(
                config.model_dir, net_name, str(seed) + ".pt"
            ))
        save_all_CAMS(net_name,)

    plt.style.use("seaborn")  # -bright")
    ALL_SCORES = pd.DataFrame(columns=["name", "acc", "acc_std",
                                       "val_acc", "val_acc_std",
                                       "loss", "loss_std",
                                       "val_loss", "val_loss_std"])
    RESULT_DICT = {}
    for net_name in ALL_NETS:
        avg_result, net_scores = average_results(os.path.join(config.raw_dir, net_name))
        avg_result.to_csv(os.path.join(config.avg_dir, f"{net_name}.csv"))
        plot_and_save(avg_result, net_name)
        score = pd.DataFrame([{
            "name": net_name, "acc": np.mean(net_scores["acc"]), "acc_std": np.std(avg_result["acc"]),
            "val_acc": np.mean(net_scores["val_acc"]), "val_acc_std": np.std(avg_result["val_acc"]),
            "loss": np.mean(net_scores["loss"]), "loss_std": np.std(avg_result["loss"]),
            "val_loss": np.mean(net_scores["val_loss"]), "val_loss_std": np.std(avg_result["val_loss"]),
        }])
        ALL_SCORES = ALL_SCORES.append(score, sort=False)
        RESULT_DICT[net_name] = avg_result
        save_all_CAMS(net_name,config.target_layers[net_name])
    ALL_SCORES.to_csv(os.path.join(config.table_dir, "All_DNN_Scores.csv"))
    plt.close("all")
    plot_all_together(RESULT_DICT)
    ALL_NETS = [dnn + " (pretrained)" for dnn in config.DNNs]
    for net_name in ALL_NETS:
        save_all_CAMS(net_name,config.target_layers[net_name])