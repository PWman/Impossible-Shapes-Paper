import os
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gcam_utils import plot_cam_on_img
from more_utils import cm_arr_to_df

plt.style.use("seaborn")


def plot_and_save(results, fpath):
    # PLOT ACCURACY
    plt.style.use("seaborn-bright")

    plt.clf()
    plt.plot(results["epoch"], results["acc"], "b")
    plt.plot(results["epoch"], results["val_acc"], "r-")
    # plt.title(fpath + " Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training", "Validation"])
    plt.xlim(0, config.num_epochs - 1)
    plt.savefig(os.path.join(fpath, "Accuracy.png"))

    # PLOT LOSS
    plt.clf()
    plt.plot(results["epoch"], results["loss"], "b-")
    plt.plot(results["epoch"], results["val_loss"], "r-")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.xlim(0, config.num_epochs - 1)
    plt.savefig(os.path.join(fpath, "Loss.png"))
    plt.close()
    return


def avg_train_results(net_name, study_num=2):
    results_path = os.path.join(config.raw_dir, f"Study {study_num}", net_name, "train_results")
    avg_results = []
    for file in os.listdir(results_path):
        result = pd.read_csv(os.path.join(results_path, file), index_col=0)
        avg_results.append(result)

    return sum(avg_results) / len(avg_results)


def get_raw_scores(net_path, average_score=True):
    results_path = os.path.join(net_path, "train_results")
    scores_df = pd.DataFrame(columns=["acc", "val_acc", "loss", "val_loss"])
    for file in os.listdir(results_path):
        result = pd.read_csv(os.path.join(results_path, file), index_col=0)
        end_result = result[result["epoch"] == max(result["epoch"])]
        scores_df = scores_df.append(end_result.drop(columns=["epoch"]))

    if average_score:
        df_av = pd.DataFrame([{
            "acc": np.mean(scores_df["acc"]),
            "acc_std": np.std(scores_df["acc"]),
            "val_acc": np.mean(scores_df["val_acc"]),
            "val_acc_std": np.std(scores_df["val_acc"]),
            "loss": np.mean(scores_df["loss"]),
            "loss_std": np.std(scores_df["loss"]),
            "val_loss": np.mean(scores_df["val_loss"]),
            "val_loss_std": np.std(scores_df["val_loss"])
        }])
        return df_av
    else:
        return scores_df


def total_cmats(net_name, study_num=2):
    cm_dir = os.path.join(config.raw_dir, f"Study {study_num}", net_name, "confusion_matrices")

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


def avg_gradcam(net_name, study_num=2, train_data=False):
    gcam_path = os.path.join(config.raw_dir, f"Study {study_num}", net_name, "gradCAM")
    if train_data:
        gcam_path = os.path.join(gcam_path, "training")
    else:
        gcam_path = os.path.join(gcam_path, "validation")

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
    avg_mask = np.nanmean(MASKS, axis=0)
    cols = ["correct", "nan_array", "prediction"]
    avg_gstats = pd.pivot_table(GSTATS, values=cols,
                                index=["img_name", "img_path"],
                                aggfunc=np.sum)

    avg_gstats.columns = ["n_correct", "n_nans", "n_poss_preds"]
    avg_gstats = pd.DataFrame(avg_gstats.to_records())

    return avg_mask, avg_gstats


def avg_save_net_results(net_name, study_num=2):
    save_dir = os.path.join(config.results_basedir, f"Study {study_num}", net_name)
    # if study_1:
    #     save_dir = os.path.join(config.expt1_dir, net_name)
    # else:
    #     save_dir = os.path.join(config.expt2_dir, net_name)
    config.check_make_dir(save_dir)
    print("Processing training results...")
    train_results = avg_train_results(net_name, study_num=study_num)
    train_results.to_csv(os.path.join(save_dir, "Train Results.csv"))
    plot_and_save(train_results, save_dir)

    print("Processing confusion matrices...")
    cm_t, cm_v = total_cmats(net_name, study_num=study_num)
    cm_path = os.path.join(save_dir, "Confusion Matrices.xlsx")
    cm_writer = pd.ExcelWriter(cm_path, engine="xlsxwriter")
    cm_t = cm_arr_to_df(cm_t, study_num=study_num)
    cm_v = cm_arr_to_df(cm_v, study_num=study_num)
    cm_t.to_excel(cm_writer, sheet_name="Training")
    cm_v.to_excel(cm_writer, sheet_name="Validation")
    cm_writer.save()

    print("Processing GradCAM results...")
    gcam_dir = os.path.join(save_dir, "gradCAM")
    config.check_make_dir(gcam_dir)

    gcam_dir_val = os.path.join(gcam_dir, "validation")
    config.check_make_dir(gcam_dir_val)
    avg_masks_val, avg_gstats_val = avg_gradcam(net_name, study_num=study_num, train_data=False)
    avg_gstats_val.to_csv(os.path.join(gcam_dir_val, "GradCAM Info.csv"))
    img_paths_val = list(avg_gstats_val["img_path"])
    for img, mask in zip(img_paths_val, avg_masks_val):
        mask_img = plot_cam_on_img(img, mask)
        img_name = os.path.basename(img)
        mask_img.save(os.path.join(gcam_dir_val, img_name))

    gcam_dir_train = os.path.join(gcam_dir, "training")
    config.check_make_dir(gcam_dir_train)
    avg_masks_train, avg_gstats_train = avg_gradcam(net_name, study_num=study_num, train_data=True)
    avg_gstats_train.to_csv(os.path.join(gcam_dir_train, "GradCAM Info.csv"))
    img_paths_train = list(avg_gstats_train["img_path"])
    for img, mask in zip(img_paths_train, avg_masks_train):
        mask_img = plot_cam_on_img(img, mask)
        img_name = os.path.basename(img)
        mask_img.save(os.path.join(gcam_dir_train, img_name))


def graph_all_results(study_num=2):
    def graph_all(col_name, ax_lim=None, pretrained=False):
        plt.figure()
        if pretrained:
            DNNs = [net for net in config.DNNs if "pretrain" in net]
        else:
            DNNs = [net for net in config.DNNs if "pretrain" not in net]
        for net_name in DNNs:
            net_path = os.path.join(expt_dir, net_name)
            if os.path.isdir(net_path):
                for file in os.listdir(net_path):
                    if "Train Results.csv" in file:
                        result = pd.read_csv(os.path.join(expt_dir, net_name, file))
                        plt.plot(result["epoch"], result[col_name])

        plt.xlabel("Epoch")
        if col_name == "val_acc":
            save_title = "Validation Accuracies"
            plt.ylabel("Validation Accuracy")
        else:
            save_title = "Validation Losses"
            plt.ylabel("Validation Loss")
        if pretrained:
            save_title = f"{save_title} (with pretraining)"
            plt.legend([net.split(" ")[0] for net in DNNs])
        else:
            save_title = f"{save_title} (no pretraining)"
            plt.legend(DNNs)
        if ax_lim:
            plt.ylim(ax_lim)
        plt.savefig(os.path.join(expt_dir, save_title))

    plt.style.use("seaborn")
    expt_dir = os.path.join(config.results_basedir, f"Study {study_num}")
    if study_num == 1:
        acc_ax_ylim = [0.45, 0.72]
    elif study_num == 2:
        acc_ax_ylim = [0.4, 0.6]
    elif study_num == 3:
        acc_ax_ylim = [0.4, 1]
    else:
        acc_ax_ylim = None

    graph_all("val_acc", acc_ax_ylim, pretrained=False)
    graph_all("val_acc", acc_ax_ylim, pretrained=True)
    graph_all("val_loss", pretrained=False)
    graph_all("val_loss", pretrained=True)
    plt.close("all")


def collate_net_scores_table(study_num=2):
    def format_df(scores_df):
        def format_lbl(df, col_name, round_val):
            avg = np.round(df[col_name].values[0], round_val)
            std = np.round(df[f"{col_name}_std"].values[0], round_val)
            return f"{avg} " + u"\u00B1" + f" {std}"

        if "pretrain" in scores_df["net_name"]:
            arch_lbl = scores_df.net_name.values[0]
        else:
            arch_lbl = f"{scores_df.net_name.values[0]} (untrained)"
        df_formatted = pd.DataFrame([{
            "DNN Architecture": arch_lbl,
            "Train Accuracy [%]": format_lbl(scores_df * 100, "acc", 1),
            "Validation Accuracy [%]": format_lbl(scores_df * 100, "val_acc", 1),
            "Train Loss": format_lbl(scores_df, "loss", 2),
            "Validation Loss": format_lbl(scores_df, "val_loss", 2),
        }])
        return df_formatted

    raw_dir = os.path.join(config.raw_dir, f"Study {study_num}")
    save_dir = os.path.join(config.results_basedir, f"Study {study_num}", "DNN Performance Summary.xlsx")

    df_unformattted = pd.DataFrame([])
    df_full_formatted = pd.DataFrame([])
    for net_name in config.DNNs:
        net_path = os.path.join(raw_dir, net_name)
        df = get_raw_scores(net_path, average_score=True)
        df.insert(0, "net_name", net_name)
        df_unformattted = df_unformattted.append(df)
        df_full_formatted = df_full_formatted.append(format_df(df))
    xl_writer = pd.ExcelWriter(save_dir, engine="xlsxwriter")
    df_full_formatted.to_excel(xl_writer, sheet_name="Formatted Results")
    df_unformattted.to_excel(xl_writer, sheet_name="Results Unformatted")
    xl_writer.save()


def collate_cmats_xl(study_num=2):

    expt_dir = os.path.join(config.results_basedir, f"Study {study_num}")
    train_cmat_xl_fpath = os.path.join(expt_dir, "Confusion Matrices (train images).xlsx")
    val_cmat_xl_fpath = os.path.join(expt_dir, "Confusion Matrices (validation images).xlsx")
    train_writer = pd.ExcelWriter(train_cmat_xl_fpath, engine="xlsxwriter")
    val_writer = pd.ExcelWriter(val_cmat_xl_fpath, engine="xlsxwriter")
    for net in config.DNNs:
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
    for study_num in range(3):
        for net in config.DNNs:
            avg_save_net_results(net, study_num=study_num)
        collate_net_scores_table(study_num=study_num)
        graph_all_results(study_num=study_num)
        collate_cmats_xl(study_num=study_num)
