import config
from torch import nn
from torch import optim
from torchvision import models

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


def get_model(model_name, pretrain=False):
    if "pretrained" in model_name:
        pretrain = True

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

        # try:
        #     opt = init_opt(out_model, pretrain=pretrain)
        #     return out_model, opt
        # except NameError:
        #     print("Model name not recognised")
        #     return

def make_models(model_names,pretrain=False):


    available_nets = config.DNNs
    MODEL_DICTS = [
        {"net_name": n, "net": None, "opt": None} for n in available_nets
    ]
    if type(model_names) == str:
        try:
            net, opt = get_model(model_names,pretrain=pretrain)
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
                net, opt = get_model(net,pretrain=pretrain)
                all_nets.append(net)
                all_opts.append(opt)

            except TypeError:
                print(f"Warning: Model name f{net} not recognised."
                      f" Skipping to next entry...")
        return all_nets, all_opts
    else:
        return