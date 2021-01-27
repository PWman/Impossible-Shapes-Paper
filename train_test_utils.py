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
from initialise_nets import initialise_DNN
from more_utils import set_seed, save_batch
from gcam_utils import gcam_all_imgs


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


def train_all_nets(net_name, scale_factor=None):  # save_models=True,
    # if save_models:
    if scale_factor is None:
        result_dir = os.path.join(config.raw_dir, f"{net_name}")
        p = Preprocess()
        # model_dir = os.path.join(config.model_dir, f"{net_name}")
    else:
        result_dir = os.path.join(config.raw_dir, f"{net_name} sf={scale_factor}")
        p = Preprocess(scale_factor=scale_factor)

        # model_dir = os.path.join(config.model_dir, f"{net_name} sf={scale_factor}")

    config.check_make_dir(result_dir)
    model_dir = os.path.join(result_dir, "models")
    config.check_make_dir(model_dir)
    result_dir = os.path.join(result_dir, "train_results")
    config.check_make_dir(result_dir)

    for seed in range(config.num_seeds):
        print(f"Testing {net_name} seed {seed}...")
        set_seed(seed)
        print("Initialising network...")
        net, opt = initialise_DNN(net_name)
        print("Training network...")
        result = train_net(p, net, opt)
        result.to_csv(os.path.join(
            result_dir, f"{seed}.csv"
        ))
        # if save_models:
        torch.save(net.state_dict(), os.path.join(model_dir, f"{seed}"))

    return


def get_cm_result(p, model):
    model.eval()
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



def save_all_cmats(net_name):
    print("Getting confusion matrices...")
    net_path = os.path.join(config.raw_dir, net_name, "models")

    save_dir = os.path.join(config.raw_dir, net_name, "confusion_matrices")
    train_dir = os.path.join(save_dir, "train_data")
    val_dir = os.path\
        .join(save_dir, "validation_data")
    config.check_make_dir(save_dir)
    config.check_make_dir(train_dir)
    config.check_make_dir(val_dir)

    if "sf=" in net_name:
        net_name = net_name[:net_name.find("sf=0.5") - 1]
        # net_name = net_name.split(" ")[0]

    p = Preprocess(augment=False)
    net, opt = initialise_DNN(net_name)
    net.to(config.device)

    for seed, file in enumerate(os.listdir(net_path)):
        net.load_state_dict(torch.load(os.path.join(net_path, file)))
        cm_t, cm_v = get_cm_result(p, net)
        np.save(os.path.join(train_dir, f"{seed}"), cm_t)
        np.save(os.path.join(val_dir, f"{seed}"), cm_v)


def save_all_gcams(net_name,train_data=False):
    print("Getting GradCAM results...")

    net_path = os.path.join(config.raw_dir, net_name, "models")
    gcam_dir = os.path.join(config.raw_dir, net_name, "gradCAM")
    config.check_make_dir(gcam_dir)
    if train_data:
        gcam_dir = os.path.join(gcam_dir,"training")
    else:
        gcam_dir = os.path.join(gcam_dir,"validation")

    gstat_dir = os.path.join(gcam_dir, "scores")
    mask_dir = os.path.join(gcam_dir, "masks")
    config.check_make_dir(gcam_dir)
    config.check_make_dir(gstat_dir)
    config.check_make_dir(mask_dir)

    if "sf=" in net_name:
        net_name = net_name[:net_name.find("sf=0.5") - 1]

    p = Preprocess(batch_size=1, augment=False, shuffle=False)
    net, opt = initialise_DNN(net_name)
    net.to(config.device)

    for seed, file in enumerate(os.listdir(net_path)):
        set_seed(int(seed))
        net.load_state_dict(torch.load(os.path.join(net_path, file)))
        if "GoogLeNet" in net_name and "pretrain" not in net_name:
            net.aux_logits = False
            net.aux1 = None
            net.aux2 = None
        if train_data:
            cam_array, cstats = gcam_all_imgs(p, net, config.target_layers[net_name],train_data=True)
        else:
            cam_array, cstats = gcam_all_imgs(p, net, config.target_layers[net_name])
        np.save(os.path.join(mask_dir, f"{seed}"), cam_array)
        cstats.to_csv(os.path.join(gstat_dir, f"{seed}.csv"))


def train_test_network(net_name, scale_factor=None):
    if scale_factor is None:
        train_all_nets(net_name)
    else:
        train_all_nets(net_name, scale_factor=scale_factor)
        net_name = f"{net_name} sf={scale_factor}"
    save_all_cmats(net_name)
    save_all_gcams(net_name)
    # save_all_gcams(net_name,train_data=True)
    return


if __name__ == "__main__":
    # for net in config.DNNs:
    #     train_test_network(net)
    #     train_test_network(net, scale_factor=0.5)
    #     save_all_gcams(f"{net} sf=0.5")

    # train_test_network("AlexNet")
    train_test_network("VGG11")
    # train_test_network("VGG16", scale_factor=0.5)
    # train_test_network("ResNet18", scale_factor=0.5)
    # train_test_network("ResNet50", scale_factor=0.5)
    # train_test_network("GoogLeNet", scale_factor=0.5)

    # train_test_network("AlexNet", scale_factor=0.5)
    # train_test_network("VGG11", scale_factor=0.5)
    # train_test_network("VGG16", scale_factor=0.5)
    # train_test_network("ResNet18", scale_factor=0.5)
    # train_test_network("ResNet50", scale_factor=0.5)
    # train_test_network("GoogLeNet", scale_factor=0.5)

    # train_test_network("AlexNet (pretrained)", scale_factor=0.5)
    # train_test_network("VGG11 (pretrained)", scale_factor=0.5)
    # train_test_network("VGG16 (pretrained)", scale_factor=0.5)
    # train_test_network("ResNet18 (pretrained)", scale_factor=0.5)
    # train_test_network("ResNet50 (pretrained)", scale_factor=0.5)
    # train_test_network("GoogLeNet (pretrained)", scale_factor=0.5)
    # save_all_cmats("VGG11 sf=0.5")
    # save_all_gcams("VGG11 sf=0.5")

    # train_test_network("GoogLeNet", scale_factor=0.5)
    # for net in config.DNNs:
    #     if "pretrain" in net:0
    #         print(net)
    #         save_all_cmats(f"{net} sf=0.5")
    #
    #         ,m-
    #         save_all_gcams(f"{net} sf=0.5")
    # save_all_cmats("VGG11 sf=0.5")
    # save_all_gcams("VGG11 sf=0.5")

            # train_test_network(net)
            # train_test_network(net, scale_factor=0.5)