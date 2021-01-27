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
from initialise_nets import initialise_DNN
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


def gcam_all_imgs(p, net, target_layer,train_data=False):
    def gcam_all(net, loader):
        cam_array = []
        df_camstats = pd.DataFrame([])
        camnet = GradCAM(net, target_layer)
        for idx, (img, lbl) in enumerate(loader):
            shape_lbls = loader.dataset.samples[idx]
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
        return cam_array, df_camstats

    net.eval()
    if train_data:
        CAM_ARR, DF = gcam_all(net, p.train_loader)
    else:
        CAM_ARR, DF = gcam_all(net, p.test_loader)
        # print(f"lbl{int(lbl)},pred{int(camnet.pred)}")
    return CAM_ARR, DF