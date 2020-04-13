import config
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from torchvision import models
from torchvision import transforms
from preprocessing import Preprocess
from sklearn.metrics import confusion_matrix
from more_utils import set_seed, save_batch


def init_net(model,pretrain=True):
    try:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,2)
        if pretrain:
            for param in model.features.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
    except AttributeError:
        if pretrain:
            for param in model.parameters():
                param.requires_grad = False
        else:
            try:
                model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features,2)
                model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features,2)
            except AttributeError:
                pass
        model.fc = nn.Linear(model.fc.in_features, 2)
    # for child in model.modules():
    #     child.track_running_stats = False
    return model

def init_opt(model,pretrain=True):
    if pretrain:
        try:
            opt = optim.Adam(model.classifier.parameters())
        except AttributeError:
            opt = optim.Adam(model.fc.parameters())
    else:
        opt = optim.Adam(model.parameters())
    return opt


def get_model(model_name):
    if "pretrained" in model_name:
        pretrain = True
    else:
        pretrain = False
    model_name = model_name.split(" ")[0]
    if pretrain:
        if model_name == "AlexNet":
            out_model = init_net(models.alexnet(pretrained=True),
                                 pretrain=True)
        elif model_name == "VGG11":
            out_model = init_net(models.vgg11(pretrained=True),
                                 pretrain=True)
        elif model_name == "VGG16":
            out_model = init_net(models.vgg16(pretrained=True),
                                 pretrain=True)
        elif model_name == "ResNet18":
            out_model = init_net(models.resnet18(pretrained=True),
                                 pretrain=True)
        elif model_name == "ResNet50":
            out_model = init_net(models.resnet50(pretrained=True),
                                 pretrain=True)
        elif model_name == "GoogLeNet":
            out_model = init_net(models.googlenet(pretrained=True),
                                 pretrain=True)
        try:
            opt = init_opt(out_model, pretrain=pretrain)
            return out_model, opt
        except NameError:
            print("Net name not recognised/supported")
            return
    else:
        if model_name == "AlexNet":
            out_model = init_net(models.alexnet(pretrained=False),
                                 pretrain=False)
        elif model_name == "VGG11":
            out_model = init_net(models.vgg11(pretrained=False),
                                 pretrain=False)
        elif model_name == "VGG16":
            out_model = init_net(models.vgg16(pretrained=False),
                                 pretrain=False)
        elif model_name == "ResNet18":
            out_model = init_net(models.resnet18(pretrained=False),
                                 pretrain=False)
        elif model_name == "ResNet50":
            out_model = init_net(models.resnet50(pretrained=False),
                                 pretrain=False)
        elif model_name == "GoogLeNet":
            out_model = init_net(models.googlenet(pretrained=False),
                                 pretrain=False)
        try:
            opt = init_opt(out_model, pretrain=pretrain)
            return out_model, opt
        except NameError:
            print("Net name not recognised/supported")
            return


def make_models(model_names):

    available_nets = config.DNNs
    MODEL_DICTS = [
        {"net_name": n, "net": None, "opt": None} for n in available_nets
    ]
    if type(model_names) == str:
        try:
            net, opt = get_model(model_names)
            return net, opt
        except TypeError:
            print("Error: Model name not recognised")
            return
    elif type(model_names) == list:
        # model_dicts = []
        all_nets = []
        all_opts = []
        for net in model_names:
            try:
                net, opt = get_model(net)
                all_nets.append(net)
                all_opts.append(opt)

            except TypeError:
                print(f"Warning: Model name f{net} not recognised."
                      f" Skipping to next entry...")
        return all_nets, all_opts
    else:
        return


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

        cm = confusion_matrix(lbl_batch, predicts, labels=[0,1])

        return cm
    else:
        return acc, float(loss)

def get_cm_result(p, model):

    cm_tot_t = np.zeros((2,2))
    for img,lbl in p.train_loader:
        cm_t = feed_net(model,None,img,lbl,
                      train_net=False,
                      confusion_matrices=True)
        cm_tot_t = cm_tot_t + cm_t
    cm_tot_v = np.zeros((2,2))
    for img,lbl in p.test_loader:
        cm_v = feed_net(model,None,img,lbl,
                      train_net=False,
                      confusion_matrices=True)
        cm_tot_v = cm_tot_v + cm_v
        # print(cm_tot_t,cm_tot_v)
    return cm_tot_t, cm_tot_v

def train_epoch(p, net, opt):

    accs_tot = 0
    loss_tot = 0
    count_t = 0
    for img_batch, lbl_batch in p.train_loader:
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
        for vimg_batch, vlbl_batch in p.test_loader:
            vaccs, vloss = feed_net(net, opt,
                                    vimg_batch, vlbl_batch,
                                    train_net=False)
            vaccs_tot = vaccs_tot + vaccs
            vloss_tot = vloss_tot + vloss
            count_v += 1

    av_vacc = vaccs_tot/count_v
    av_vloss = vloss_tot/count_v

    return av_acc,av_loss,av_vacc,av_vloss


# def train_net(net_name, seed_num, num_epochs=None, batch_size=16, EStop=None):
#     if num_epochs == None:
#         num_epochs = config.num_epochs
#
#     set_seed(seed_num)
#     net, opt = make_models(net_name)
#     net.to(config.device)
#     p = Preprocess(batch_size=batch_size)
#     train_results = pd.DataFrame(columns=["epoch", "acc", "loss",
#                                           "val_acc", "val_loss"])
#     if EStop:
#         for epoch in num_epochs:
#             acc, loss, v_acc, v_loss = train_epoch(p, net, opt)
#             r = pd.DataFrame([{
#                 "epoch": epoch,
#                 "acc": acc,
#                 "val_acc": v_acc,
#                 "loss": loss,
#                 "val_loss": v_loss,
#             }])
#             train_results = train_results.append(r, sort=True)
#
#     else:
#         for epoch in num_epochs:
#             acc, loss, v_acc, v_loss = train_epoch(p, net, opt)
#             r = pd.DataFrame([{
#                 "epoch": epoch,
#                 "acc": acc,
#                 "val_acc": v_acc,
#                 "loss": loss,
#                 "val_loss": v_loss,
#             }])
#             train_results = train_results.append(r, sort=True)
#
#     return train_results



def train_net(net_name, num_epochs=None, batch_size=16, EStop=None,view=False):
    if num_epochs == None:
        num_epochs = config.num_epochs

    net, opt = make_models(net_name)
    net.to(config.device)
    p = Preprocess(batch_size=batch_size)

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

    if EStop:
        print("yes")
    else:
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
