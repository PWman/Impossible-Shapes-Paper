import os
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gcam_utils import plot_cam_on_img
from more_utils import cm_arr_to_df


def plot_and_save(results,fpath):
    # PLOT ACCURACY
    plt.style.use("seaborn-bright")

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

def avg_train_results(net_name,study_1=True):
    if study_1:
        results_path = os.path.join(config.raw_dir, "Study 1", net_name, "train_results")
    else:
        results_path = os.path.join(config.raw_dir, "Study 2", net_name, "train_results")

    avg_results = []
    scores = pd.DataFrame(columns=["acc", "val_acc", "loss", "val_loss"])
    for file in os.listdir(results_path):
        result = pd.read_csv(os.path.join(results_path, file), index_col=0)
        end_result = result[result["epoch"] == max(result["epoch"])]  # .drop(columns=["epoch"])
        scores = scores.append(end_result.drop(columns=["epoch"]))
        avg_results.append(result)

    return sum(avg_results) / len(avg_results), scores


def total_cmats(net_name,study_1=True):
    if study_1:
        cm_dir = os.path.join(config.raw_dir, "Study 1", net_name, "confusion_matrices")
    else:
        cm_dir = os.path.join(config.raw_dir, "Study 2", net_name, "confusion_matrices")

    train_dir = os.path.join(cm_dir, "train_data")
    val_dir = os.path.join(cm_dir, "validation_data")
    cm_tot_t = np.zeros((2, 2))
    for file in os.listdir(train_dir):
        cm_t = np.load(os.path.join(train_dir, file))
        cm_tot_t = cm_tot_t + cm_t
    cm_tot_v = np.zeros((2, 2))
    for file in os.listdir(val_dir):
        cm_v = np.load(os.path.join(val_dir, file))
        cm_tot_v = cm_tot_v + cm_v
    return cm_tot_t, cm_tot_v


def avg_gradcam(net_name,study_1=True):
    if study_1:
        gcam_path = os.path.join(config.raw_dir, "Study 1", net_name, "gradCAM", "validation")
    else:
        gcam_path = os.path.join(config.raw_dir, "Study 2", net_name, "gradCAM", "validation")

    mask_path = os.path.join(gcam_path, "masks")
    gstat_path = os.path.join(gcam_path, "scores")

    MASKS = []
    GSTATS = pd.DataFrame([])
    itercam = zip(os.listdir(mask_path), os.listdir(gstat_path))
    for seed, (arr, csv) in enumerate(itercam):
        mask = np.load(os.path.join(mask_path, arr))
        MASKS.append(mask)
        df = pd.read_csv(os.path.join(gstat_path, csv))
        df["seed_num"] = seed
        GSTATS = GSTATS.append(df)
    GSTATS.to_csv(os.path.join(gcam_path, "all_gradCAMs_info.csv"))

    MASKS = np.array(MASKS)
    seed_idx = GSTATS[GSTATS["avg_include"] == False]["seed_num"]
    img_idx = seed_idx.index
    for iidx, sidx in zip(img_idx, seed_idx):
        # print(np.sum(MASKS[sidx, iidx, :, :]))
        MASKS[sidx, iidx, :, :] = np.nan
    avg_mask = np.nanmean(MASKS, axis=0)
    cols = ["avg_include", "correct", "nan_array", "prediction"]
    avg_gstats = pd.pivot_table(GSTATS, values=cols,
                                index=["img_name", "img_path"],
                                aggfunc=np.sum)

    avg_gstats.columns = ["n_usable_cams", "n_correct", "n_nans", "n_poss_preds"]
    avg_gstats = pd.DataFrame(avg_gstats.to_records())

    return avg_mask, avg_gstats


def avg_save_net_results(net_name, study_1=True):

    if study_1:
        save_dir = os.path.join(config.expt1_dir, net_name)
    else:
        save_dir = os.path.join(config.expt2_dir, net_name)
    config.check_make_dir(save_dir)
    print("Processing training results...")
    train_results, _ = avg_train_results(net_name,study_1=study_1)
    train_results.to_csv(os.path.join(save_dir, "Train Results.csv"))
    plot_and_save(train_results, save_dir)

    print("Processing confusion matrices...")
    cm_t, cm_v = total_cmats(net_name,study_1=study_1)
    cm_path = os.path.join(save_dir, "Confusion Matrices.xlsx")
    cm_writer = pd.ExcelWriter(cm_path, engine="xlsxwriter")
    cm_t = cm_arr_to_df(cm_t)
    cm_v = cm_arr_to_df(cm_v)
    cm_t.to_excel(cm_writer, sheet_name="Training")
    cm_v.to_excel(cm_writer, sheet_name="Validation")
    cm_writer.save()

    print("Processing GradCAM results...")
    avg_masks, avg_gstats = avg_gradcam(net_name,study_1=study_1)
    gcam_dir = os.path.join(save_dir, "gradCAM")
    config.check_make_dir(gcam_dir)
    avg_gstats.to_csv(os.path.join(gcam_dir, "GradCAM Info.csv"))
    img_paths = list(avg_gstats["img_path"])
    for img, mask in zip(img_paths, avg_masks):
        mask_img = plot_cam_on_img(img, mask)
        img_name = os.path.basename(img)
        mask_img.save(os.path.join(gcam_dir, img_name))


def graph_all_results(study_1=True):
    plt.style.use("seaborn")
    if study_1:
        expt_dir = config.expt1_dir
        acc_ax_ylim = [0.45, 0.72]
        loss_ax_ylim = []
    else:
        expt_dir= config.expt2_dir
        acc_ax_ylim = [0.4, 0.6]

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
                        plt.figure(2)
                        plt.plot(result["epoch"], result["val_loss"])
                        leg1.append(net_name)
                    else:
                        plt.figure(3)
                        plt.plot(result["epoch"], result["val_acc"])
                        plt.figure(4)
                        plt.plot(result["epoch"], result["val_loss"])
                        leg2.append(net_name.split(" ")[0])

    plt.figure(1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(leg1)
    # plt.title("Validation Accuracies (no pretraining)")
    plt.ylim(acc_ax_ylim)
    plt.savefig(os.path.join(expt_dir, "Validation Accuracies (no pretraining).png"))
    plt.figure(2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(leg2)
    # plt.title("Validation Losses (no pretraining)")
    plt.savefig(os.path.join(expt_dir, "Validation Losses (no pretraining).png"))

    plt.figure(3)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(leg1)
    # plt.title("Validation Accuracies (with pretraining)")
    plt.ylim(acc_ax_ylim)
    plt.savefig(os.path.join(expt_dir, "Validation Accuracies (with pretraining).png"))
    plt.figure(4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(leg2)
    # plt.title("Validation Losses (with pretraining)")
    plt.savefig(os.path.join(expt_dir, "Validation Losses (with pretraining)"))
    plt.close("all")
    return


def collate_net_scores_table(study_1=True):
    net_names = config.DNNs
    # if scale_factor is not None:
    #     net_names = [f"{n} sf={scale_factor}" for n in net_names]
    if study_1:
        raw_dir = os.path.join(config.raw_dir_expt1)
    else:
        raw_dir = os.path.join(config.raw_dir_expt2)

    collated_table = pd.DataFrame([])
    for net in net_names:
        result_path = os.path.join(raw_dir, net, "train_results")
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
        collated_table = collated_table.append(result)

    if study_1:
        collated_table.to_csv(os.path.join(config.expt1_dir, "DNN perfomance summary.csv"))
    else:
        collated_table.to_csv(os.path.join(config.expt2_dir, "DNN perfomance summary.csv"))

    return collated_table

def collate_cmats_xl(study_1=True):
    if study_1:
        expt_dir = config.expt1_dir
    else:
        expt_dir = config.expt2_dir

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
if __name__ == "__main__":
    # for net in config.DNNs:
    #     avg_save_net_results(net, study_1=False)
    #     avg_save_net_results(net, study_1=True)
    collate_net_scores_table(study_1=True)
    collate_net_scores_table(study_1=False)
    graph_all_results(study_1=True)
    graph_all_results(study_1=False)