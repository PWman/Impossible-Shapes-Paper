import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-bright")

def average_results(result_dir, net_arch, num_seeds):
    avg_results = []
    scores = pd.DataFrame(columns=["acc", "val_acc", "loss", "val_loss"])
    for seed in range(num_seeds):
        net_name = net_arch + "_s" + str(seed) + ".csv"
        net = os.path.join(result_dir, net_name)
        result = pd.read_csv(net,index_col=0)

        end_result = result[result["epoch"]==max(result["epoch"])]#.drop(columns=["epoch"])
        scores = scores.append(end_result.drop(columns=["epoch"]))

        avg_results.append(result)

    return sum(avg_results)/len(avg_results), scores


def fit_and_plot(results,save_dir,fname):

    # PLOT ACCURACY
    plt.clf()
    plt.plot(results["epoch"], results["acc"], "b")
    plt.plot(results["epoch"], results["val_acc"], "r-")
    plt.title(fname + " Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training", "Validation"])
    plt.xlim(0, 99)
    plt.savefig(os.path.join(save_dir, "Accuracy", fname + ".png"))

    # PLOT LOSS
    plt.clf()
    plt.plot(results["epoch"], results["loss"], "b-")
    plt.plot(results["epoch"], results["val_loss"], "r-")
    plt.title(fname + " Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.xlim(0, 99)
    plt.savefig(os.path.join(save_dir, "Loss", fname + ".png"))

    return

def get_best(all_scores,num_best):
    scores = all_scores.dropna()
    best_scores = pd.DataFrame(columns=scores.columns)
    for score in range(num_best):
        best = scores[scores["val_loss"] == scores["val_loss"].min()]
        scores = scores[scores["val_loss"] != scores["val_loss"].min()]
        best_scores = best_scores.append(best)
    return best_scores

def plot_all_loss(result_dict,best_scores,path_name):
    plt.clf()
    if ("lr" or "bs") in best_scores:
        best_scores = best_scores.sort_values(by=["lr", "bs"])

    leg_names = []
    for index, row in best_scores.iterrows():
        result = result_dict[row["name"]]
        plt.plot(result["epoch"], result["val_loss"])
        leg_names.append(row["legend"])

    plt.title(os.path.basename(path_name))
    plt.legend(leg_names)
    # plt.ylim([None, 1.4])
    plt.xlim([0, 99])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(path_name)

    return
