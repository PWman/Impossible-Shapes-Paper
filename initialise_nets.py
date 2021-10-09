import config
from torch import nn
from torch import optim
from torchvision import models


def init_net(model, pretrain=True):
    try:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
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
                model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, 2)
                model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, 2)
            except AttributeError:
                pass
        model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def init_opt(model, pretrain=True):
    if pretrain:
        try:
            opt = optim.Adam(model.classifier.parameters())
        except AttributeError:
            opt = optim.Adam(model.fc.parameters())
    else:
        opt = optim.Adam(model.parameters())
    return opt


def get_model(model_name):
    if "pretrain" in model_name:
        pretrain = True
    else:
        pretrain = False
    model_name = model_name.split(" ")[0]
    if "AlexNet" in model_name:
        out_model = init_net(models.alexnet(pretrained=pretrain),
                             pretrain=pretrain)
    elif "VGG11" in model_name:
        out_model = init_net(models.vgg11_bn(pretrained=pretrain),
                             pretrain=pretrain)
    elif "VGG16" in model_name:
        out_model = init_net(models.vgg16_bn(pretrained=pretrain),
                             pretrain=pretrain)
    elif "ResNet18" in model_name:
        out_model = init_net(models.resnet18(pretrained=pretrain),
                             pretrain=pretrain)
    elif "ResNet50" in model_name:
        out_model = init_net(models.resnet50(pretrained=pretrain),
                             pretrain=pretrain)
    elif "GoogLeNet" in model_name:
        out_model = init_net(models.googlenet(pretrained=pretrain),
                             pretrain=pretrain)
        for child in out_model.modules():
            child.track_running_stats = False
    else:
        print("Net name not recognised/supported")
        return
    return out_model


def initialise_DNN(model_name):
    available_nets = config.DNNs
    if type(model_name) == str:
        if model_name in available_nets:
            net = get_model(model_name)
            if "pretrain" in model_name:
                opt = init_opt(net, pretrain=True)
            else:
                opt = init_opt(net, pretrain=False)

            return net, opt
        else:
            print("Net name not recognised/supported")
    else:
        print("Please input DNN name as a string")
        return
