import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms

def set_seed(seednum):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seednum)
    torch.cuda.manual_seed_all(seednum)
    np.random.seed(seednum)
    random.seed(seednum)
    return

def save_batch(X,bname,train):
    save_path = os.path.join(os.getcwd(), "Shapes", "Saved Train Images")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if train:
        save_path = os.path.join(save_path,"Train")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    else:
        save_path = os.path.join(save_path,"Test")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    for count, (img,lbl) in enumerate(X):
        save_name = os.path.join(save_path, bname + "img_" +
                                 str(count) + "_class_" +
                                 str(lbl.tolist()) + ".bmp")
        transforms.ToPILImage()(img).save(save_name)

    return


def feed_net(net, loss_func, opt, img, lbl, device, train=False):

    img = img.to(device)
    lbl = lbl.to(device)
    if train:
        net.train()
    else:
        net.eval()
    net_out = net(img)

    if len(net_out) == len(lbl):
        loss = loss_func(net_out, lbl)
    else:
        losses = []
        for i in net_out:
            losses.append(loss_func(i, lbl))
        loss = losses[0] + 0.3*(sum(losses[1:]))
        net_out = net_out[0]

    if train:
        loss.backward()
        opt.step()
        opt.zero_grad()

    preds = [i.index(max(i)) for i in net_out.tolist()]
    matches = [i == j for i, j in zip(preds,lbl.tolist())]
    acc = sum(matches)/len(matches)

    return acc,float(loss)

def train(p, net, loss_func, opt, device,epochs,view=False):
    results = pd.DataFrame(columns=["epoch", "acc", "loss", "val_acc", "val_loss"])
    train_data = p.train_loader
    valid_data = p.test_loader
    net = net.to(device)

    if view:
        for idx, (img, lbl) in enumerate(train_data):
            save_batch(zip(img, lbl), "batch_" + str(idx) + "_",train=True)
        for idx, (img, lbl) in enumerate(valid_data):
            save_batch(zip(img, lbl), "batch_" + str(idx) + "_",train=False)

    for epoch in range(epochs):
        acc_total = 0
        loss_total = 0
        val_acc_total = 0
        val_loss_total = 0
        t_count = 0
        for img, lbl in train_data:
            acc, loss = feed_net(net, loss_func, opt,
                            img, lbl, device,
                            train=True)
            acc_total += acc
            loss_total += loss
            t_count += 1

        v_count = 0
        for i in range(2):
            for img_v, lbl_v in valid_data:
                val_acc, val_loss = feed_net(net, loss_func, opt,
                                        img_v, lbl_v, device,
                                        train=False)
                val_acc_total += val_acc
                val_loss_total += val_loss
                v_count += 1

        result = pd.DataFrame({
            "epoch": [epoch], "acc": [acc_total/t_count],"val_acc": [val_acc_total/v_count],
            "loss": [loss_total/t_count],"val_loss": [val_loss_total/v_count]
        })
        results = pd.concat([result, results],sort=False)
        print(f"Epoch {epoch} Complete")
        print(f"Acc = {round(result.acc[0],2)} Val Acc = {round(result.val_acc[0],2)}")

    return results