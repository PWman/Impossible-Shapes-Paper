import os
import config
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.stats import norm, ttest_ind, ttest_rel, ttest_1samp
from preprocessing import Preprocess
from process_results import get_raw_scores
from more_utils import get_ffil_img, set_seed, round_sig, get_background_proportion

plt.style.use("seaborn")


def cohens_dval(x, y, two_sample=True):
    if two_sample:
        mean_diff = (np.mean(x) - np.mean(y))
        st_dev = np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2.0)
    else:
        mean_diff = (np.mean(x) - y)
        st_dev = np.std(x, ddof=1)
    return mean_diff / st_dev


def get_poss_vs_imposs_acc_df(train_data=False, study_num=2):
    data_dir = os.path.join(config.raw_dir, f"Study {study_num}")
    if train_data:
        subdir = "train_data"
    else:
        subdir = "validation_data"

    df_imp = pd.DataFrame([])
    df_poss = pd.DataFrame([])

    for net in config.DNNs:
        net_fpath = os.path.join(data_dir, net, "confusion_matrices", subdir)
        imp_scores = []
        poss_scores = []
        for file in os.listdir(net_fpath):
            fpath = os.path.join(net_fpath, file)
            # print(fpath)
            cm_arr = np.load(fpath)
            imp_scores.append(cm_arr[0, 0] / np.sum(cm_arr[0, :]))
            poss_scores.append(cm_arr[1, 1] / np.sum(cm_arr[1, :]))

        df_imp[net] = imp_scores
        df_poss[net] = poss_scores
    return df_imp, df_poss


def get_acc_ttest_table(study_num=2):
    def calculate_all_cohen(df_imp, df_poss):
        df_cohen = pd.DataFrame([np.zeros(len(config.DNNs))], columns=config.DNNs)
        for net_name in config.DNNs:
            poss = np.array(df_poss[net_name])
            imp = np.array(df_imp[net_name])
            dval = cohens_dval(imp, poss)
            df_cohen[net_name] = dval
        return df_cohen

    train_imp, train_poss = get_poss_vs_imposs_acc_df(train_data=True, study_num=study_num)
    val_imp, val_poss = get_poss_vs_imposs_acc_df(train_data=False, study_num=study_num)

    imp_arr = np.array([np.array(train_imp), np.array(val_imp)])
    poss_arr = np.array([np.array(train_poss), np.array(val_poss)])

    tvals, pvals = ttest_rel(imp_arr, poss_arr, axis=1)

    df_tval_t = pd.DataFrame([{net: val for net, val in zip(config.DNNs, tvals[0])}])
    df_tval_v = pd.DataFrame([{net: val for net, val in zip(config.DNNs, tvals[1])}])

    df_all_tvals = df_tval_t.append(df_tval_v)
    df_all_tvals.index = ["training", "validation"]

    df_pval_t = pd.DataFrame([{net: val for net, val in zip(config.DNNs, pvals[0])}])
    df_pval_v = pd.DataFrame([{net: val for net, val in zip(config.DNNs, pvals[1])}])
    df_all_pvals = df_pval_t.append(df_pval_v)
    df_all_pvals.index = ["training", "validation"]

    dvals_t = calculate_all_cohen(train_imp, train_poss)
    dvals_v = calculate_all_cohen(val_imp, val_poss)

    df_all_dvals = dvals_t.append(dvals_v)

    return df_all_tvals, df_all_pvals, df_all_dvals


def save_acc_ttest_results(study_num=2):
    df_tval, df_pval, df_dval = get_acc_ttest_table(study_num=study_num)
    save_path = os.path.join(config.results_basedir, f"Study {study_num}",
                             "Impossible vs Possible Accuracy T-test.xlsx")
    xl_writer = pd.ExcelWriter(save_path, engine="xlsxwriter")
    df_tval.to_excel(xl_writer, sheet_name="t-values")
    df_pval.to_excel(xl_writer, sheet_name="p-values")
    df_dval.to_excel(xl_writer, sheet_name="cohen's d-values")
    xl_writer.save()


def get_background_proportion_1iter(train_data=True, study_num=2):
    data_transforms = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.ToPILImage(),
        transforms.Grayscale()
    ])
    prepro_dir = os.path.join(config.prepro_dir, f"Study {study_num}")
    p = Preprocess(data_dir=prepro_dir, scale_factor=0.9, batch_size=1, augment=True, shuffle=False)
    if train_data:
        loader = p.train_loader
        classes = p.train_class_names
    else:
        loader = p.test_loader
        classes = p.test_class_names

    df_all = pd.DataFrame([])
    for idx, (torch_img, lbl) in enumerate(loader):
        shape_lbls = loader.dataset.samples[idx]
        shape_name = os.path.basename(shape_lbls[0]).replace(".bmp", "")
        pil_img = data_transforms(torch.squeeze(torch_img))
        ffil_img = get_ffil_img(pil_img)
        bg_pct = get_background_proportion(ffil_img)
        df = pd.DataFrame([{"image": shape_name,
                            "label_num": shape_lbls[1],
                            "label": classes[shape_lbls[1]],
                            "background_pct": bg_pct
                            }])
        df_all = df_all.append(df)
    return df_all


def get_background_proportion_20iter(train_data=True, study_num=2, save_imgs=False):
    df_all = pd.DataFrame([])
    for i in range(20):
        set_seed(i)
        print(f"Testing iteration {i}...")
        df = get_background_proportion_1iter(train_data=train_data, study_num=study_num)
        df["seed"] = i
        df_all = df_all.append(df)
    return df_all


def background_ttest_and_plot(df_bg, img_mean=False):
    labels = list(df_bg.label.unique())
    if img_mean:
        df_av = pd.DataFrame([])
        for img in df_bg.image.unique():
            data = df_bg[df_bg["image"] == img]
            lbl = data["label"].values[0]
            df = pd.DataFrame([{
                "image": img,
                "label": lbl,
                "background_pct": np.mean(data.background_pct)
            }])
            df_av = df_av.append(df)
        df_bg = df_av
    # print(df_bg)
    bg_c1 = df_bg[df_bg["label"] == labels[0]].background_pct
    bg_c2 = df_bg[df_bg["label"] == labels[1]].background_pct
    bins = np.linspace(min(df_bg["background_pct"]), max(df_bg["background_pct"]), 20)

    tval, pval = ttest_ind(bg_c1, bg_c2)
    dval = cohens_dval(bg_c1, bg_c2)

    plt.figure()
    x_1, y_1, _ = plt.hist(bg_c1, bins=bins, color="r", alpha=0.5)
    x_2, y_2, _ = plt.hist(bg_c2, bins=bins, color="b", alpha=0.5)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    mu_1, std_1 = norm.fit(bg_c1)
    mu_2, std_2 = norm.fit(bg_c2)
    pdf_1 = norm.pdf(x, mu_1, std_1)
    pdf_2 = norm.pdf(x, mu_2, std_2)
    c1_mul = x_1.max() / max(pdf_1)
    c2_mul = x_1.max() / max(pdf_2)
    plt.plot(x, pdf_1 * c1_mul, color="r")
    plt.plot(x, pdf_2 * c2_mul, color="b")
    plt.legend([
        f"{labels[0]}: mu={round_sig(mu_1, 2)}, std={round_sig(std_1, 2)}",
        f"{labels[1]}: mu={round_sig(mu_2, 2)}, std={round_sig(std_2, 2)}"
    ])
    plt.title(f"Background T-Test: t-value={round_sig(tval, 2)},"
              f" p-value={round_sig(pval, 2):.2e};"
              f" Cohen's d-value={round_sig(dval, 2)}")


def save_background_t_test_result(study_num=2, img_mean=False):
    save_fpath = os.path.join(config.results_basedir,f"Study {study_num}", "Background Proportion T-Test")
    df_train = get_background_proportion_20iter(study_num=study_num, train_data=True)
    df_train["dataset"] = "training"
    df_val = get_background_proportion_20iter(study_num=study_num, train_data=False)
    df_val["dataset"] = "validation"
    df_results = df_train.append(df_val)
    background_ttest_and_plot(df_results, img_mean=img_mean)
    plt.savefig(save_fpath)


def get_acc_1samp_ttest(study_num=2):
    data_dir = os.path.join(config.raw_dir,f"Study {study_num}")
    save_dir = os.path.join(config.results_basedir,f"Study {study_num}")
    df_all = pd.DataFrame([])
    all_scores = []
    for net_name in config.DNNs:
        net_path = os.path.join(data_dir, net_name)
        score_df = get_raw_scores(net_path, average_score=False)
        train_dval = cohens_dval(score_df["acc"], 0.5, two_sample=False)
        val_dval = cohens_dval(score_df["val_acc"], 0.5, two_sample=False)
        df_dvals = pd.DataFrame([{
            "net_name": net_name,
            "train_d-value": train_dval,
            "validation_d-value": val_dval
        }])
        df_all = df_all.append(df_dvals)
        all_scores.append(np.array(score_df[["acc", "val_acc"]]))

    tvals, pvals = ttest_1samp(all_scores, 0.5, axis=1)
    df_all["train_t-value"] = tvals[:, 0]
    df_all["validation_t-value"] = tvals[:, 1]
    df_all["train_p-value"] = pvals[:, 0]
    df_all["validation_p-value"] = pvals[:, 1]
    df_all = df_all[["net_name", "train_t-value", "train_p-value", "train_d-value",
                     "validation_t-value", "validation_p-value", "validation_d-value"]]
    df_all.to_csv(os.path.join(save_dir, "Accuracy 1-Sample T-Test.csv"))


if __name__ == "__main__":
    save_acc_ttest_results(study_num=0)
    save_acc_ttest_results(study_num=1)
    save_acc_ttest_results(study_num=2)
    save_background_t_test_result(study_num=0, img_mean=True)
    save_background_t_test_result(study_num=1, img_mean=True)
    save_background_t_test_result(study_num=2, img_mean=True)
    get_acc_1samp_ttest(study_num=0)
    get_acc_1samp_ttest(study_num=1)
    get_acc_1samp_ttest(study_num=2)
