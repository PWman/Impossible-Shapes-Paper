import os
import torch
import config
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torchvision import models, transforms
from preprocessing import Preprocess
from training_functions import train, set_seed
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam

def get_imgs(img_path):
    img = Image.open(img_path)
    normed_img = transformations(img).view(-1, 3, 224, 224).to(config.device)
    return img, normed_img

def get_cams(net,datadir,config):

    for param in net.parameters():
        param.requires_grad = True
    net = net.to(config.device)
    net.eval()

    cam = GradCAM.from_config(model_type="resnet", arch=net, layer_name="layer4")
    cam_pp = GradCAMpp.from_config(model_type="resnet", arch=net, layer_name="layer4")

    all_names = []
    all_preds = []
    all_masks = []
    all_masks_pp = []
    for class_idx, class_name in enumerate(datadir):
        subdir = os.path.join(datadir,class_name)
        for img_name in os.listdir(subdir):

            img, normed_img = get_imgs(os.path.join(subdir, img_name))
            net_out = net(normed_img).tolist()[0]
            all_preds.append(net_out.index(max(net_out)))

            mask, _ = cam(normed_img, class_idx=class_idx)
            mask_pp, _ = cam_pp(normed_img, class_idx=class_idx)

            all_names.append(img_name)
            all_masks.append(mask)
            all_masks_pp.append(mask_pp)


    return zip(all_names, all_masks, all_masks_pp)


transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

results = pd.read_csv(os.path.join(config.results_dir_3,"Raw Scores.csv"))

score_dict = {}
for seed in results["seed"]:
    x = results[results["seed"] == seed]
    correct = x.iloc[0][2] + x.iloc[1][3]
    score_dict[seed] = correct

best_seed = max(score_dict, key=score_dict.get)

best_params_path = os.path.join(
    config.param_dir,
    config.best_params["name"]
    + "_s" + str(best_seed)
    + ".pt"
)

net = models.resnet18(pretrained=True)
for param in net.parameters():
    param.requires_grad = False
net.fc = nn.Linear(512, 2)
net.load_state_dict(torch.load(best_params_path))


loss_func = nn.CrossEntropyLoss()
p = Preprocess(img_size=224, batch_size=config.best_params["bs"], split=0.2, colour=True)
opt = optim.SGD(net.fc.parameters(), lr=config.best_params["bs"], momentum=0.9, weight_decay=0.0001)

best_cams = os.path.join(config.cam_dir,"Best")
best_gc_dir = os.path.join(best_cams,"GradCAM")
best_gcpp_dir = os.path.join(best_cams,"GradCAM++")
if not os.path.isdir(best_cams):
    os.mkdir(best_cams)
if not os.path.isdir(best_gc_dir):
    os.mkdir(best_gc_dir)
if not os.path.isdir(best_gcpp_dir):
    os.mkdir(best_gcpp_dir)

val_dir = os.path.join(os.getcwd(), "Shapes", "Shapes_Preprocessed", "Validation")

cam_result = get_cams(net,val_dir,config)