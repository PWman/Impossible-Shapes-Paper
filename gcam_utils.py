import os
import config
from tqdm import tqdm
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from torch.autograd import Function
from torchvision import models, transforms
from more_utils import set_seed
from preprocessing import Preprocess
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from net_utils import initialise_DNN
"""
Parts of the code here (i.e. GradCAM, FeatureExtractor and ModelOutputs objects) were modified from Jacob Gildenblat's (jacobgil) Pytorch implementation of GradCAM.
The original code is under MIT license and is available at the link below:
https://github.com/jacobgil/pytorch-grad-cam
"""


#
class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            # print(name)
            # print(module)

            x = module(x)
            if name in self.target_layers:
                # print(f"TARGET_LAYER={name}")
                # print(module)
                x.register_hook(self.save_gradient)
                outputs += [x]

        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = self.model.avgpool(output)
        # output = output.view(output.size(0), -1)
        try:
            output = self.model.classifier(output)
        except RuntimeError:
            output = self.model.classifier(output.view(output.size(0), -1))
        return target_activations, output


class GradCAM(nn.Module):

    def __init__(self, model, target_layers):
        super(GradCAM, self).__init__()
        if torch.cuda.is_available():
            model.cuda()  # to(config.device)
        for layer in model.parameters():
            layer.requires_grad = True
        model.eval()
        try:
            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = model.classifier
        except AttributeError:
            # try:
            final_pool_idx = list(dict(model.named_children()).keys()).index('avgpool')
            self.features = nn.Sequential(*list(model.children())[:final_pool_idx])
            self.avgpool = model.avgpool
            self.classifier = model.fc  # nn.Sequential(*list(model.children())[final_pool_idx+1:])

        self.extractor = ModelOutputs(self, target_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x.view(1, -1))
        return x

    def __call__(self, input, index=None):
        if torch.cuda.is_available():
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        # features, output = self.extractor(input.to(config.device))
        self.output = output

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
            self.pred = index

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if torch.cuda.is_available():
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        # one_hot = torch.sum(one_hot.to(config.device) * output)

        self.features.zero_grad()
        self.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def plot_cam_on_img(img_path, mask):
    # if np.amax(mask) == 1:
    # print(mask.shape)
    if np.amin(mask) != 0:
        mask = mask - np.amin(mask)
    if np.amax(mask) != 1:
        mask = mask / np.amax(mask)

    cmap = plt.get_cmap("jet")
    # print(mask.shape)
    mask = cmap(mask)
    mask = Image.fromarray(np.uint8(mask * 225))
    # print(mask.shape)
    shape_img = Image.open(img_path).resize((224, 224)).convert("RGBA")
    mask_img = Image.blend(shape_img, mask, 0.5)
    return mask_img


def gcam_all_imgs(p, net, target_layer):
    cam_array = []
    net.eval()
    df_camstats = pd.DataFrame([])
    camnet = GradCAM(net, target_layer)
    for idx, (img, lbl) in enumerate(p.test_loader):
        shape_lbls = p.test_loader.dataset.samples[idx]
        # print(f"Testing GradCAM for {os.path.basename(shape_lbls[0])}")
        mask = camnet(img)
        cam_array.append(mask)
        # print(camnet.output)

        if np.isnan(np.sum(mask)):
            nan_nums = True
        else:
            nan_nums = False
        if int(lbl) == int(camnet.pred):
            correct = True
        else:
            correct = False
        if correct and not nan_nums:
            avg_inc = True
        else:
            avg_inc = False

        df = pd.DataFrame([{
            "img_name": os.path.basename(shape_lbls[0]),
            "img_path": shape_lbls[0],
            "net_output": list(camnet.output.cpu().detach().numpy()[0]),
            "label": bool(lbl),
            "prediction": bool(camnet.pred),
            "correct": correct,
            "nan_array": nan_nums,
            "avg_include": avg_inc

        }])
        # print(df[["avg_include", "nan_array", "correct"]])
        df_camstats = df_camstats.append(df)
        # print(f"lbl{int(lbl)},pred{int(camnet.pred)}")
    return cam_array, df_camstats
# def get_conv_layer(net):
#     for i, c in enumerate(net.children()):
#         if len(c._modules) > 0:
#             print((c._modules))




# def get_cam_all_layers(net_name):
#     model_dir = os.path.join(config.raw_dir,net_name,"models")
#     net,_ = initialise_DNN(net_name)
#     p = Preprocess(batch_size=1,shuffle=False)
#     for file in model_dir:
#         net_name


#
# def gcam_all_imgs(p,net,target_layer):
#     CAM_DICTS = []
#     camnet = GradCAM(net,target_layer)
#     for idx, (img,lbl) in enumerate(p.test_loader):
#         shape_lbls = p.test_loader.dataset.samples[idx]
#         mask = camnet(img)
#         cam_dict = {
#             "img_path": shape_lbls[0],
#             "lbl": int(lbl),
#             "pred": camnet.pred,
#             "mask": mask
#         }
#         CAM_DICTS.append(cam_dict)
#         print(f"lbl{int(lbl)},pred{int(camnet.pred)}")
#     return CAM_DICTS
#
#
# def get_cam_all_seed(net_name,target_layer):
#
#     p = Preprocess(batch_size=1,augment=False,shuffle=False)
#     ALL_CAMDICTS = []
#     model_dir = os.path.join(config.model_dir,net_name)
#     net, _ = initialise_DNN(net_name)
#     # for sdict in tqdm(os.listdir(os.path.join(config.model_dir, pt_netname))):
#     for sdict in os.listdir(model_dir):
#
#         net.load_state_dict(torch.load(os.path.join(
#             model_dir, sdict
#         )))
#         print(sdict)
#         cam_dict = gcam_all_imgs(p,net,target_layer)
#         ALL_CAMDICTS = ALL_CAMDICTS + cam_dict
#
#     return ALL_CAMDICTS
#
#
# def avg_CAM(ALL_CAMDICT):
#     df = pd.DataFrame(ALL_CAMDICT)
#
#     AVG_CAMS = []
#     for img in df["img_path"].unique():
#         data = df[df["img_path"] == img]
#
#         ncorrect = 0
#         all_masks = []
#         for idx,row in data.iterrows():
#             if row["lbl"] == row["pred"]:
#                 ncorrect += 1
#                 all_masks.append(row["mask"])
#
#         avg_mask = np.mean(all_masks,axis=0)
#         avg_dict = {
#             "img_path": img,
#             "ncorrect": ncorrect,
#             "avg_mask": avg_mask
#         }
#         AVG_CAMS.append(avg_dict)
#     return AVG_CAMS
#
# def save_all_CAMS(net_name,target_layer):
#
#     save_dir = os.path.join(config.gradcam_dir,net_name)
#     config.check_make_dir(save_dir)
#
#     ALL_CAM = get_cam_all_seed(net_name,target_layer=target_layer)
#     AVG_CDICT = avg_CAM(ALL_CAM)
#     for dic in AVG_CDICT:
#         if dic["ncorrect"] > 0:
#             ncorrect = dic["ncorrect"]Q
#             img_name = os.path.basename(dic["img_path"]).replace(".bmp","")
#
#             mask_img = plot_cam_on_img(dic["img_path"],dic["avg_mask"])
#             mask_img.save(os.path.join(save_dir,f"{img_name}_correct{ncorrect}.png"))
#     return
#

