import os
import config
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from preprocessing import Preprocess
from initialise_nets import initialise_DNN
from process_results import avg_gradcam, cm_arr_to_df
from more_utils import get_ffil_img, set_seed, save_batch, round_sig
from skimage.segmentation import flood_fill
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import norm, ttest_ind, ttest_rel, ttest_1samp
from process_results import get_raw_scores

def cohens_dval(x,y,two_sample=True):
    if two_sample:
        mean_diff = (np.mean(x) - np.mean(y))
        st_dev = np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2.0)
    else:
        mean_diff = (np.mean(x) - y)
        st_dev = np.std(x, ddof=1)
    return mean_diff/st_dev

def get_poss_vs_imposs_acc_df(train_data=False, study_1=True):
    if study_1:
        data_dir = config.raw_dir_expt1
    else:
        data_dir = config.raw_dir_expt2

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


def get_acc_ttest_table(study_1=False):
    def calculate_all_cohen(df_imp, df_poss):
        df_cohen = pd.DataFrame([np.zeros(len(config.DNNs))],columns=config.DNNs)
        for net_name in config.DNNs:
            poss = np.array(df_poss[net_name])
            imp = np.array(df_imp[net_name])
            dval = cohens_dval(imp, poss)
            df_cohen[net_name] = dval
        return df_cohen

    train_imp, train_poss = get_poss_vs_imposs_acc_df(train_data=True, study_1=study_1)
    val_imp, val_poss = get_poss_vs_imposs_acc_df(train_data=False, study_1=study_1)

    imp_arr = np.array([np.array(train_imp), np.array(val_imp)])
    poss_arr = np.array([np.array(train_poss), np.array(val_poss)])

    tvals,pvals = ttest_rel(imp_arr, poss_arr, axis=1)

    df_tval_t = pd.DataFrame([{net: val for net, val in zip(config.DNNs, tvals[0])}])
    df_tval_v = pd.DataFrame([{net: val for net, val in zip(config.DNNs, tvals[1])}])

    df_all_tvals = df_tval_t.append(df_tval_v)
    df_all_tvals.index = ["training", "validation"]

    df_pval_t = pd.DataFrame([{net: val for net, val in zip(config.DNNs,pvals[0])}])
    df_pval_v = pd.DataFrame([{net: val for net, val in zip(config.DNNs,pvals[1])}])
    df_all_pvals = df_pval_t.append(df_pval_v)
    df_all_pvals.index = ["training","validation"]

    dvals_t = calculate_all_cohen(train_imp,train_poss)
    dvals_v = calculate_all_cohen(val_imp,val_poss)

    df_all_dvals = dvals_t.append(dvals_v)

    return df_all_tvals, df_all_pvals, df_all_dvals


def save_acc_ttest_results(study_1=True):
    df_tval, df_pval,df_dval = get_acc_ttest_table(study_1=study_1)
    if study_1:
        save_path = os.path.join(config.expt1_dir, "Impossible vs Possible Accuracy T-test.xlsx")
    else:
        save_path = os.path.join(config.expt2_dir, "Impossible vs Possible Accuracy T-test.xlsx")
    xl_writer = pd.ExcelWriter(save_path, engine="xlsxwriter")
    df_tval.to_excel(xl_writer, sheet_name="t-values")
    df_pval.to_excel(xl_writer, sheet_name="p-values")
    df_dval.to_excel(xl_writer, sheet_name="cohen's d-values")
    xl_writer.save()

def get_background_proportion_1iter(train_data=True, study_1=True, save_dir=None):
    def get_background_proportion(img_arr):
        return np.sum(img_arr) / (img_arr.shape[0] * img_arr.shape[1])

    data_transforms = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.ToPILImage(),
        transforms.Grayscale()

    ])

    if study_1:
        prepro_dir = os.path.join(config.prepro_dir, "Study 1")
    else:
        prepro_dir = os.path.join(config.prepro_dir, "Study 2")

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
        # torch_img = torch.squeeze(torch_img)
            # save_batch(torch_img,lbl,os.path.join(save_dir,f"{shape_name}"))
        pil_img = data_transforms(torch.squeeze(torch_img))
        ffil_img = get_ffil_img(pil_img)
        if save_dir is not None:
            plt.clf()
            plt.imshow(ffil_img)
            plt.savefig(os.path.join(save_dir,f"{shape_name}_{idx}"))

        bg_pct = get_background_proportion(ffil_img)
        df = pd.DataFrame([{"image": shape_name,
                            "label_num": shape_lbls[1],
                            "label": classes[shape_lbls[1]],
                            "background_pct": bg_pct
                            }])
        df_all = df_all.append(df)
    return df_all


def get_background_proportion_20iter(train_data=True,study_1=True,save_imgs=False):
    df_all = pd.DataFrame([])
    for i in range(20):
        set_seed(i)
        print(f"Testing iteration {i}...")
        if save_imgs:
            save_dir = os.path.join(config.check_train_dir, f"sim_seed{i}_study1{study_1}")
            config.check_make_dir(save_dir)
            df = get_background_proportion_1iter(train_data=train_data, study_1=study_1,save_dir=save_dir)
        else:
            df = get_background_proportion_1iter(train_data=train_data, study_1=study_1)

        df["seed"] = i
        df_all = df_all.append(df)
    return df_all

def background_ttest_and_plot(df_bg,seed_mean=False):

    # def mean_across_seeds()
    if seed_mean:
        df_av = pd.DataFrame([])
        for seed in df_bg.seed.unique():
            data = df_bg[df_bg["seed"]==seed]
            b_i = np.mean(data[data["label"] == "Impossible"].background_pct)
            b_p = np.mean(data[data["label"] == "Possible"].background_pct)

            df_i = pd.DataFrame([{"seed":seed,"label":"Impossible","background_pct":b_i}])
            df_p = pd.DataFrame([{"seed":seed,"label":"Possible","background_pct":b_p}])
            df_av = df_av.append(df_i.append(df_p))
        df_bg = df_av

    bg_poss = df_bg[df_bg["label"] == "Possible"].background_pct
    bg_imp = df_bg[df_bg["label"] == "Impossible"].background_pct
    bins = np.linspace(min(df_bg["background_pct"]), max(df_bg["background_pct"]), 20)

    tval,pval = ttest_ind(bg_imp,bg_poss)
    dval = cohens_dval(bg_imp,bg_poss)
    x_p, y_p, _ = plt.hist(bg_poss, bins=bins,color="r",alpha=0.5)
    x_i, y_i, _ = plt.hist(bg_imp, bins=bins,color="b",alpha=0.5)

    xmin,xmax = plt.xlim()
    x = np.linspace(xmin,xmax,100)

    mu_p, std_p = norm.fit(bg_poss)
    mu_i, std_i = norm.fit(bg_imp)

    pdf_p = norm.pdf(x,mu_p,std_p)
    pdf_i = norm.pdf(x,mu_i,std_i)

    p_mul = x_p.max()/max(pdf_p)
    i_mul = x_i.max()/max(pdf_i)

    plt.plot(x,pdf_p*p_mul, color="r")
    plt.plot(x,pdf_i*i_mul, color="b")

    plt.legend([
        f"Possible: mu={round_sig(mu_p,2)}, std={round_sig(std_p,2)}",
        f"Impossible: mu={round_sig(mu_i,2)}, std={round_sig(std_i,2)}"
    ])

    plt.title(f"Background T-Test: t-value={round_sig(tval,2)},"
              f" p-value={round_sig(pval,2):.2e};"
              f" Cohen's d-value={round_sig(dval,2)}")

def save_background_t_test_result(study_1=True,train_data=True,seed_mean=False):
    if study_1:
        save_fpath = os.path.join(config.expt1_dir,"Background Proportion T-Test")
    else:
        save_fpath = os.path.join(config.expt2_dir,"Background Proportion T-Test")

    df_results = get_background_proportion_20iter(study_1=study_1, train_data=train_data)
    background_ttest_and_plot(df_results,seed_mean=seed_mean)
    plt.savefig(save_fpath)

def get_acc_1samp_ttest(study_1=True):
    if study_1:
        data_dir = config.raw_dir_expt1
        save_dir = config.expt1_dir
    else:
        data_dir = config.raw_dir_expt2
        save_dir = config.expt2_dir

    df_all = pd.DataFrame([])
    for net_name in config.DNNs:
        net_path = os.path.join(data_dir,net_name)
        score_df = get_raw_scores(net_path, average_score=False)
        train_tval, train_pval = ttest_1samp(score_df["acc"], 0.5)
        train_dval = cohens_dval(score_df["acc"], 0.5,two_sample=False)
        val_tval, val_pval = ttest_1samp(score_df["val_acc"], 0.5)
        val_dval = cohens_dval(score_df["val_acc"],0.5,two_sample=False)
        df = pd.DataFrame([{
            "net_name": net_name,
            "train_t-value": train_tval,
            "train_p-value": train_pval,
            "train_d-value": train_dval,
            "validation_t-value": val_tval,
            "validation_p-value": val_pval,
            "validation_d-value": val_dval
        }])
        df_all = df_all.append(df)

    df_all.to_csv(os.path.join(save_dir,"Accuracy 1-Sample T-Test.csv"))


if __name__ == "__main__":
    save_acc_ttest_results(study_1=True)
    save_acc_ttest_results(study_1=False)
    df_s1_train = get_background_proportion_20iter(study_1=True, train_data=True)
    df_s2_train = get_background_proportion_20iter(study_1=False, train_data=True)

    save_background_t_test_result(study_1=True,seed_mean=True)
    save_background_t_test_result(study_1=False,seed_mean=True)

    plt.figure()
    background_ttest_and_plot(df_s2_train)
    get_acc_1samp_ttest(study_1=True)
    get_acc_1samp_ttest(study_1=False)