import os
import torch
import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torchvision import models, transforms
from preprocessing import Preprocess
from training_functions import train, set_seed
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam

def get_imgs(img_path,data_transforms):
    img = Image.open(img_path).resize((224,224)).convert("RGB")
    normed_img = data_transforms(img).view(-1, 3, 224, 224).to(config.device)
    return img, normed_img

def init_cams():
    cam = GradCAM.from_config(model_type="resnet",
                              arch=net,
                              layer_name="layer4")

    cam_pp = GradCAMpp.from_config(model_type="resnet",
                                   arch=net,
                                   layer_name="layer4")
    return cam,cam_pp

# def get_cams(cam,cam_pp,normed_img,class_idx):
#     mask, _ = cam(normed_img, class_idx=class_idx)
#     mask_pp, _ = cam_pp(normed_img, class_idx=class_idx)
#     return mask, mask_pp


def feed_for_cm_cam(seed,grad_cam=False,train_data=False):

    net.eval()

    cm = pd.DataFrame(columns=["seed", "actual_class",
                               "imp_preds", "poss_preds"])

    if grad_cam:
        gc_dict = {}
        cam, cam_pp = init_cams()

    if train_data:
        my_dir = os.path.join(config.prepro_dir,"Training")
    else:
        my_dir = os.path.join(config.prepro_dir,"Validation")

    for class_idx, class_name in enumerate(os.listdir(my_dir)):
        subdir = os.path.join(my_dir, class_name)
        predictions = []
        for shape_name in os.listdir(subdir):
            img, normed_img = get_imgs(os.path.join(subdir,shape_name),transformations)
            # img = Image.open(os.path.join(subdir, shape_name)).resize((224, 224)).convert("RGB")
            # normed_img = transformations(img).view(-1, 3, 224, 224).to(config.device)
            net_out = net(normed_img.to(config.device)).tolist()[0]
            pred = net_out.index(max(net_out))
            predictions.append(pred)


            if grad_cam:
                # lbl_name = shape_name.replace(".bmp",
                #                               "class" + str(class_idx)
                #                               + "pred" + str(pred))
                mask, _ = cam(normed_img, class_idx=class_idx)
                mask_pp,_ = cam(normed_img,class_idx=class_idx)
                gc_dict[shape_name.replace(".bmp","")] = [mask.cpu(),mask_pp.cpu(),class_idx,pred]

        poss_preds = sum(predictions)
        imp_preds = len(predictions) - sum(predictions)
        cm_row = pd.DataFrame([{"seed": seed,
                                "actual_class": class_name,
                                "imp_preds": imp_preds,
                                "poss_preds": poss_preds}])

        cm = cm.append(cm_row, sort=True)

    if grad_cam:
        return cm, gc_dict
    else:
        return cm

def average_cm(all_cm_scores):
    all_imp = all_cm_scores[all_cm_scores["actual_class"] == "Impossible"]
    all_poss = all_cm_scores[all_cm_scores["actual_class"] == "Possible"]

    true_positive = sum(all_imp["imp_preds"])
    true_negative = sum(all_poss["poss_preds"])

    false_positive = sum(all_poss["imp_preds"])
    false_negative = sum(all_imp["poss_preds"])

    predicted_imp = pd.DataFrame([{
        "prediction": "Impossible",
        "actual_imp": true_positive,
        "actual_poss": false_positive,
    }])
    predicted_poss = pd.DataFrame([{
        "prediction": "Possible",
        "actual_imp": false_negative,
        "actual_poss": true_negative
    }])

    confusion_matrix = pd.concat([predicted_imp, predicted_poss])

    return confusion_matrix

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


p = Preprocess(img_size=224, batch_size=config.best_params["bs"], split=0.2, colour=True)
loss_func = nn.CrossEntropyLoss()

all_cms_train = pd.DataFrame(columns=["seed", "actual_class",
                                      "imp_preds", "poss_preds"])
all_cms_val = pd.DataFrame(columns=["seed", "actual_class",
                                      "imp_preds", "poss_preds"])
cam_dict_list = []
for seed in range(config.num_seeds):
    print("\nTESTING SEED  " + str(seed+1) + "/" + str(config.num_seeds) + "...\n")
    set_seed(seed)
    net = models.resnet50(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    net.fc = nn.Linear(2048, 2)
    opt = optim.SGD(net.fc.parameters(),lr=config.best_params["lr"], momentum=0.9,weight_decay=0.0001)
    result = train(p, net, loss_func, opt, config.device, epochs=100)

    for param in net.parameters():
        param.requires_grad = True

    cm_t = feed_for_cm_cam(seed, train_data=True)
    all_train_cms = all_cms_train.append(cm_t, sort=True)

    cm_v, cam_dict = feed_for_cm_cam(seed, grad_cam=True)
    all_cms_val = all_cms_val.append(cm_v, sort=True)

    cam_dict_list.append(cam_dict)


val_dir = os.path.join(config.prepro_dir,"Validation")



for subdir in os.listdir(val_dir):
    subdir = os.path.join(val_dir,subdir)
    for img_name in os.listdir(subdir):
        img = Image.open(os.path.join(subdir,img_name)).resize((224,224))
        CAMS = []
        CAMS_PP = []
        corr_count = 0
        for cam in cam_dict_list:
            x = cam[img_name.replace(".bmp","")]
            CAMS.append(x[0].cpu().view(224,224).numpy())
            CAMS_PP.append(x[1].cpu().view(224,224).numpy())
            if x[2] == x[3]:
                corr_count += 1
        avg_hmap = sum(CAMS)/len(CAMS)
        avg_hmap_pp = sum(CAMS_PP)/len(CAMS_PP)

        plt.imshow(img, alpha=0.5, cmap="gray")
        plt.imshow(avg_hmap, alpha=0.5,cmap="jet")
        plt.title("GradCAM for " + img_name.replace(".bmp", " ")
                  + str(corr_count) + "/20 Correct")
        plt.savefig(os.path.join(config.cam_dir, "GradCAM",
                                 img_name.replace(".bmp",".png")))
        # plt.show()
        plt.clf()

        plt.imshow(img, alpha=0.5, cmap="gray")
        plt.imshow(avg_hmap_pp, alpha=0.5,cmap="jet")
        plt.title("GradCAM for " + img_name.replace(".bmp", " ")
                  + str(corr_count) + "/20 Correct")
        plt.savefig(os.path.join(config.cam_dir, "GradCAM++",
                                 img_name.replace(".bmp",".png")))
        plt.clf()

# shape_names = []
# AVG_HMAPS = []
# corr_count = 0
# val_dir = os.path.join(config.prepro_dir,"Validation")
# for subdir in os.listdir(val_dir):
#
#     for shape in os.listdir(os.path.join(val_dir,subdir)):
#         img,normed_img = get_imgs(os.path.join(val_dir, subdir, shape))
#         img = Image.open(os.path.join(val_dir, subdir, shape))
#         cam_list = []
#         for expt in cam_dict_list:
#             x = expt[shape.replace(".bmp","")]
#             if x[1] == x[2]:
#                 cam_img = x[0].cpu().view(224,224).numpy()
#                 cam_list.append(cam_img)
#             print(len(cam_list))
#
#         if len(cam_list)>10:
#             plt.figure()
#             avg_hmap = sum(cam_list)/len(cam_list)
#             plt.imshow(img,alpha=0.5,cmap="gray")
#             plt.imshow(avg_hmap,alpha=0.5)
#             plt.title("GradCAM for " + shape.replace(".bmp"," ")
#                       + str(len(cam_list)) + "/20 Correct")
#             AVG_HMAPS.append(avg_hmap)




# shape_names = []
# AVG_HMAPS = []
# corr_count = 0
# val_dir = os.path.join(config.prepro_dir,"Validation")
# for subdir in os.listdir(val_dir):
#     for shape in os.listdir(os.path.join(val_dir,subdir)):
#         img = Image.open(os.path.join(val_dir, subdir, shape))
#         cam_list = []
#         for expt in cam_dict_list:
#             x = expt[shape.replace(".bmp","")]
#             if x[1] == x[2]:
#                 cam_img = x[0].cpu().view(224,224).numpy()
#                 cam_list.append(cam_img)
#             print(len(cam_list))
#
#         if len(cam_list)>10:
#             plt.figure()
#             avg_hmap = sum(cam_list)/len(cam_list)
#             plt.imshow(img,alpha=0.5,cmap="gray")
#             plt.imshow(avg_hmap,alpha=0.5)
#             plt.title("GradCAM for " + shape.replace(".bmp"," ")
#                       + str(len(cam_list)) + "/20 Correct")
#             plt.savefig("")
#             AVG_HMAPS.append(avg_hmap)
