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
        loss = losses[0] + 0.3*(sum(losses[1:]))
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
        # print(predicts)
        cm = confusion_matrix(lbl_batch, predicts)
        # print(cm)
        # print(predicts)#,"\n",lbl_batch)
        return cm
    else:
        return acc, float(loss)


def train_epoch(net, train_loader, valid_loader, opt):

    accs_tot = 0
    loss_tot = 0
    count_t = 0
    for img_batch, lbl_batch in train_loader:
        accs, loss = feed_net(net, opt, img_batch, lbl_batch, train_net=True)
        accs_tot = accs_tot + accs
        loss_tot = loss_tot + loss
        count_t += 1

    av_acc = accs_tot/count_t
    av_loss = loss_tot/count_t

    vaccs_tot = 0
    vloss_tot = 0
    count_v = 0
    for x in range(2):
        for vimg_batch, vlbl_batch in valid_loader:
            vaccs, vloss = feed_net(net, opt,
                                    vimg_batch, vlbl_batch,
                                    train_net=False)
            vaccs_tot = vaccs_tot + vaccs
            vloss_tot = vloss_tot + vloss
            count_v += 1

    av_vacc = vaccs_tot/count_v
    av_vloss = vloss_tot/count_v

    # print(accs_tot)
    # print()

    return av_acc,av_loss,av_vacc,av_vloss

def train_net(p, net,opt,view=False):

    net.to(config.device)
    train_loader = p.train_loader
    valid_loader = p.test_loader

    results = pd.DataFrame(columns=["epoch", "acc", "loss",
                                    "val_acc", "val_loss"])
    if view:
            for idx, (img, lbl) in enumerate(train_loader):
                save_batch(zip(img, lbl), f"batch_{idx}_", train_data=True)
            for idx, (img, lbl) in enumerate(valid_loader):
                save_batch(zip(img, lbl), f"batch_{idx}_", train_data=False)

    for epoch in range(config.num_epochs):
        t_acc, t_loss, v_acc, v_loss = train_epoch(net,train_loader,valid_loader,opt)

        r = pd.DataFrame([{
            "epoch": epoch,
            "acc": t_acc,
            "val_acc": v_acc,
            "loss": t_loss,
            "val_loss": v_loss,
        }])

        results = results.append(r, sort=True)

        print(f"Epoch {epoch} Complete")
        print(f"Acc = {round(t_acc,2)} Val Acc = {round(v_acc,2)}")
    return results



if __name__ == "__main__":
    from torch import nn
    from torch import optim
    from torchvision import models

    p = Preprocess(img_size=224, batch_size=16, split=0.2)
    net = models.googlenet(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    net.fc = nn.Linear(1024, 2)
    opt = optim.Adam(net.fc.parameters())
    results = train_net(p,net,opt,view=True)

    # cm_t, cm_v = get_cm_result(net)
