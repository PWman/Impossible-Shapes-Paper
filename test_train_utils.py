import os
import config
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from torchvision import models
from torchvision import transforms
from preprocessing import Preprocess
from sklearn.metrics import confusion_matrix
from net_utils import initialise_DNN
# from more_utils import set_seed, save_batch
from gcam_utils import GradCAM


def feed_net(net, opt, img_batch, lbl_batch,
             train_net=False,
             confusion_matrices=False):
    img_batch = img_batch.to(config.device)
    lbl_batch = lbl_batch.to(config.device)
    if train_net:
        net.train()
    else:
        net.eval()
    net_out = net(img_batch)
    # print(len(net_out))

    if len(net_out[0]) == 2:
        loss = config.loss_fun(net_out, lbl_batch)
    else:
        losses = []
        for i in net_out:
            losses.append(config.loss_fun(i, lbl_batch))
        loss = losses[0] + 0.3 * (sum(losses[1:]))
        net_out = net_out[0]

    if train_net:
        loss.backward()
        opt.step()
        opt.zero_grad()

    lbl_batch = lbl_batch.tolist()
    net_out = net_out.tolist()
    # print(net_out,lbl_batch)
    predicts = [i.index(max(i)) for i in net_out]
    matches = [i == j for i, j in zip(predicts, lbl_batch)]

    acc = sum(matches) / len(matches)

    if confusion_matrices:

        cm = confusion_matrix(lbl_batch, predicts, labels=[0, 1])

        return cm
    else:
        return acc, float(loss)


def train_epoch(p, net, opt):
    train_scores = []
    for img_batch, lbl_batch in p.train_loader:
        acc, loss = feed_net(net, opt, img_batch, lbl_batch, train_net=True)
        train_scores.append([acc, loss])

    valid_scores = []
    for x in range(2):
        for vimg_batch, vlbl_batch in p.test_loader:
            vacc, vloss = feed_net(net, opt, vimg_batch, vlbl_batch, train_net=False)
            valid_scores.append([vacc, vloss])

    return np.mean(train_scores, axis=0), np.mean(valid_scores, axis=0)


def train_net(p, net, opt):
    num_epochs = config.num_epochs
    net.to(config.device)
    # batch_size = config.batch_size
    # net,opt = initialise_DNN(net_name)

    results = pd.DataFrame(columns=["epoch", "acc", "loss",
                                    "val_acc", "val_loss"])
    for epoch in range(num_epochs):
        [t_acc, t_loss], [v_acc, v_loss] = train_epoch(p, net, opt)

        r = pd.DataFrame([{
            "epoch": epoch,
            "acc": t_acc,
            "val_acc": v_acc,
            "loss": t_loss,
            "val_loss": v_loss,
        }])
        results = results.append(r, sort=True)
        print(f"Epoch {epoch} Complete")
        print(f"Acc = {round(t_acc, 2)} Val Acc = {round(v_acc, 2)}")

    return results


def get_cm_result(p, model):
    cm_tot_t = np.zeros((2, 2))
    for img, lbl in p.train_loader:
        cm_t = feed_net(model, None, img, lbl,
                        train_net=False,
                        confusion_matrices=True)
        cm_tot_t = cm_tot_t + cm_t
    cm_tot_v = np.zeros((2, 2))
    for img, lbl in p.test_loader:
        cm_v = feed_net(model, None, img, lbl,
                        train_net=False,
                        confusion_matrices=True)
        cm_tot_v = cm_tot_v + cm_v
        # print(cm_tot_t,cm_tot_v)
    return cm_tot_t, cm_tot_v


def gcam_all_imgs(p, net, target_layer):
    cam_array = []
    df_camstats = pd.DataFrame([])
    camnet = GradCAM(net, target_layer)
    for idx, (img, lbl) in enumerate(p.test_loader):
        shape_lbls = p.test_loader.dataset.samples[idx]
        # print(f"Testing GradCAM for {os.path.basename(shape_lbls[0])}")
        mask = camnet(img)
        cam_array.append(mask)

        if int(lbl) == int(camnet.pred):
            correct = True
        else:
            correct = False
        df = pd.DataFrame([{
            "img_name": os.path.basename(shape_lbls[0]),
            "img_path": shape_lbls[0],
            "correct": correct,
            "label": int(lbl),
            "prediction": int(camnet.pred)
        }])
        df_camstats = df_camstats.append(df)
        # print(f"lbl{int(lbl)},pred{int(camnet.pred)}")
    return cam_array, df_camstats
