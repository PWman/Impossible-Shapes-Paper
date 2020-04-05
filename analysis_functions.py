import os
import config
import pandas as pd
import matplotlib.pyplot as plt

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

# def plot_all(result_dict):
#     legname = []
#     for key,val in result_dict.values():
#         result = result_dict[key]
#         plt.plot(result["epoch"],result["val_acc"])
#         legname.append(key)
#     plt.xlabel("Epoch")
#     plt.ylabel("Validation Accuracy")
#     plt.legend(legname)
#     for key,val in result_dict.values():
#         result = result_dict[key]
#         plt.plot(result["epoch"],result["val_loss"])
#     plt.xlabel("Epoch")
#     plt.ylabel("Validation Loss")
    # plt.clf()
    # if ("lr" or "bs") in best_scores:
    #     best_scores = best_scores.sort_values(by=["lr", "bs"])
    #
    # leg_names = []
    # for index, row in best_scores.iterrows():
    #     result = result_dict[row["name"]]
    #     plt.plot(result["epoch"], result["val_loss"])
    #     leg_names.append(row["legend"])
    #
    # plt.title(os.path.basename(path_name))
    # plt.legend(leg_names)
    # # plt.ylim([None, 1.4])
    # plt.xlim([0, 99])
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.savefig(path_name)

    # return