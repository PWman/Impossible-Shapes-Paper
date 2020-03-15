import os
import config
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from preprocessing import Preprocess
from training_functions import set_seed, train

def initialise_networks():

    alex = models.alexnet(pretrained=False)
    alex.classifier[-1] = nn.Linear(4096, 2)
    vgg = models.vgg16(pretrained=False)
    vgg.classifier[-1] = nn.Linear(4096, 2)

    vgg_s = models.vgg11(pretrained=False)
    vgg_s.classifier[-1] = nn.Linear(4096, 2)

    resnet = models.resnet50(pretrained=False)
    resnet.fc = nn.Linear(2048, 2)
    resnet_s = models.resnet18(pretrained=False)
    resnet_s.fc = nn.Linear(512, 2)

    incept = models.googlenet(pretrained=False)
    incept.aux1.fc2 = nn.Linear(1024, 2)
    incept.aux2.fc2 = nn.Linear(1024, 2)
    incept.fc = nn.Linear(1024, 2)

    alex_pt = models.alexnet(pretrained=True)
    alex_pt.classifier[-1] = nn.Linear(4096, 2)
    for param in alex_pt.features.parameters():
        param.requires_grad = False
    for param in alex_pt.classifier.parameters():
        param.requires_grad = True


    vgg_pt = models.vgg16(pretrained=True)
    vgg_pt.classifier[-1] = nn.Linear(4096, 2)
    for param in vgg_pt.features.parameters():
        param.requires_grad = False
    for param in vgg_pt.classifier.parameters():
        param.requires_grad = True


    vgg_pt_s = models.vgg11(pretrained=True)
    vgg_pt_s.classifier[-1] = nn.Linear(4096, 2)
    for param in vgg_pt_s.features.parameters():
        param.requires_grad = False
    for param in vgg_pt_s.classifier.parameters():
        param.requires_grad = True

    resnet_pt = models.resnet50(pretrained=True)
    for param in resnet_pt.parameters():
        param.requires_grad = False
    resnet_pt.fc = nn.Linear(2048,2)

    resnet_pt_s = models.resnet18(pretrained=True)
    for param in resnet_pt_s.parameters():
        param.requires_grad = False
    resnet_pt_s.fc = nn.Linear(512,2)


    incept_pt = models.googlenet(pretrained=True)
    for param in incept_pt.parameters():
        param.requires_grad = False
    incept_pt.fc = nn.Linear(1024,2)

    nets = [
        {"name": config.net_names[0], "net": alex, "train": alex},
        {"name": config.net_names[1], "net": vgg_s, "train": vgg_s},
        {"name": config.net_names[2], "net": vgg, "train": vgg},
        {"name": config.net_names[3], "net": resnet_s, "train": resnet_s},
        {"name": config.net_names[4], "net": resnet, "train": resnet},
        {"name": config.net_names[5], "net": incept, "train": incept},
        {"name": config.net_names[6], "net": alex_pt, "train": alex_pt.classifier},
        {"name": config.net_names[7], "net": vgg_pt_s, "train": vgg_pt_s.classifier},
        {"name": config.net_names[8], "net": vgg_pt, "train": vgg_pt.classifier},
        {"name": config.net_names[9], "net": resnet_pt_s, "train": resnet_pt_s.fc},
        {"name": config.net_names[10], "net": resnet_pt, "train": resnet_pt.fc},
        {"name": config.net_names[11], "net": incept_pt, "train": incept_pt.fc}
    ]
    return nets

pre = Preprocess(img_size=224, batch_size=16, split=0.2, colour=True)
loss_fun = nn.CrossEntropyLoss()

for seed in range(config.num_seeds):
    set_seed(seed)
    print("\nTESTING SEED  " + str(seed+1) + "/" + str(config.num_seeds) + "...\n")
    nets = initialise_networks()

    for net in nets:
        print("Testing " + net["name"] + "...")
        file_name = net["name"] + "_s" + str(seed) + ".csv"
        opt = optim.Adam(net["train"].parameters())
        result = train(pre,net["net"],loss_fun, opt,
                       config.device, epochs=config.num_epochs)
        result.to_csv(os.path.join(config.raw_dir_1, file_name))
