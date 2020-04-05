import os
import torch
import random
import config
import numpy as np
import pandas as pd
from torchvision import transforms
from preprocessing import Preprocess
from sklearn.metrics import confusion_matrix
from other_funcs import set_seed, save_batch
from training_functions import feed_net
from initialise_nets import make_models

def get_cm_result(p, model,opt):

    cm_tot_t = np.zeros((2,2))
    for img,lbl in p.train_loader:
        cm_t = feed_net(model,opt,img,lbl,
                      train_net=False,
                      confusion_matrices=True)
        cm_tot_t = cm_tot_t + cm_t
    cm_tot_v = np.zeros((2,2))
    for img,lbl in p.test_loader:
        cm_v = feed_net(model,opt,img,lbl,
                      train_net=False,
                      confusion_matrices=True)
        cm_tot_v = cm_tot_v + cm_v
        # print(cm_tot_t,cm_tot_v)

    return cm_tot_t, cm_tot_v


def get_cms_all(p, net_name):
    model_path = os.path.join(config.model_dir,net_name)
    # print(net_name)
    if "pretrain" in net_name:
        # print(net_name[:net_name.index(" ")])
        net,opt = make_models(net_name[:net_name.index(" ")], pretrain=True)
    else:
        net,opt = make_models(net_name, pretrain=False)
    net.to(config.device)
    cm_tot_t = 0
    cm_tot_v = 0
    for file in os.listdir(model_path):
        net.load_state_dict(torch.load(os.path.join(model_path, file)))
        cm_t, cm_v = get_cm_result(p,net,opt=None)
        cm_tot_t = cm_tot_t + cm_t
        cm_tot_v = cm_tot_v + cm_v

    return cm_tot_t, cm_tot_v

def cm_arr_to_df(arr):
    df = pd.DataFrame(arr)
    df.columns = ["Pred Imposs", "Pred Poss"]
    df.index = ["Actual Imposs", "Actual Poss"]
    return df

pretrained_DNNs = [n + " (pretrained)" for n in config.DNNs]
ALL_NET_NAMES = config.DNNs + pretrained_DNNs
# ALL_NET_NAMES = pretrained_DNNs

p = Preprocess()


train_dir = os.path.join(config.cm_dir, "Training")
val_dir = os.path.join(config.cm_dir, "Validation")
config.check_make_dir(train_dir)
config.check_make_dir(val_dir)


writer_t = pd.ExcelWriter(
    os.path.join(config.cm_dir,"All_Train_Confusion_Matrices.xlsx"),
    engine="xlsxwriter"
)
writer_v = pd.ExcelWriter(
    os.path.join(config.cm_dir,"All_Validation_Confusion_Matrices.xlsx"),
    engine="xlsxwriter"
)

for net_name in ALL_NET_NAMES:
    print(f"Testing {net_name}...")
    cm_t, cm_v = get_cms_all(p,net_name)
    df_t = cm_arr_to_df(cm_t)
    df_v = cm_arr_to_df(cm_v)
    df_t.to_csv(os.path.join(train_dir, f"{net_name}.csv"))
    df_v.to_csv(os.path.join(val_dir, f"{net_name}.csv"))
    df_t.to_excel(writer_t, sheet_name=net_name)
    df_v.to_excel(writer_v, sheet_name=net_name)

writer_t.save()
writer_v.save()