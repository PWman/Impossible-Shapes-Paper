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
from preprocessing import Preprocess
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
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
            output = self.model.classifier(output.view(output.size(0),-1))
        return target_activations, output


class GradCAM(nn.Module):

    def __init__(self,model,target_layers):
        super(GradCAM, self).__init__()
        if torch.cuda.is_available():
            model.cuda()#to(config.device)
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
            self.classifier = model.fc #nn.Sequential(*list(model.children())[final_pool_idx+1:])

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


def gcam_all_imgs(p,net,target_layer):
    CAM_DICTS = []
    camnet = GradCAM(net,target_layer)
    for idx, (img,lbl) in enumerate(p.test_loader):
        shape_lbls = p.test_loader.dataset.samples[idx]
        mask = camnet(img)
        cam_dict = {
            "img_path": shape_lbls[0],
            "lbl": int(lbl),
            "pred": camnet.pred,
            "mask": mask
        }
        CAM_DICTS.append(cam_dict)
        print(f"lbl{int(lbl)},pred{int(camnet.pred)}")
    return CAM_DICTS

