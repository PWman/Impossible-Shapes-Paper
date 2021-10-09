import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt

"""
Parts of the code here (i.e. GradCAM, FeatureExtractor and ModelOutputs objects) were modified from Jacob Gildenblat's (jacobgil) Pytorch implementation of GradCAM.
The original code is under MIT license and is available at the link below:
https://github.com/jacobgil/pytorch-grad-cam
"""


#
class FeatureExtractor:
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
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]

        return outputs, x


class ModelOutputs:
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = self.model.avgpool(output)
        try:
            output = self.model.classifier(output)
        except RuntimeError:
            output = self.model.classifier(output.view(output.size(0), -1))
        return target_activations, output


class GradCAM(nn.Module):
    def __init__(self, model, target_layers):
        super(GradCAM, self).__init__()
        if torch.cuda.is_available():
            model.cuda()
        for layer in model.parameters():
            layer.requires_grad = True
        model.eval()
        try:
            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = model.classifier
        except AttributeError:
            final_pool_idx = list(dict(model.named_children()).keys()).index('avgpool')
            self.features = nn.Sequential(*list(model.children())[:final_pool_idx])
            self.avgpool = model.avgpool
            self.classifier = model.fc
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
        self.output = output

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        self.pred = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if torch.cuda.is_available():
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
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
    if np.amin(mask) != 0:
        mask = mask - np.amin(mask)
    if np.amax(mask) != 1:
        mask = mask / np.amax(mask)
    cmap = plt.get_cmap("jet")
    mask = cmap(mask)
    mask = Image.fromarray(np.uint8(mask * 225))
    shape_img = Image.open(img_path).resize((224, 224)).convert("RGBA")
    mask_img = Image.blend(shape_img, mask, 0.5)
    return mask_img


def gcam_all_imgs(p, net, target_layer, train_data=False):
    def gcam_all(net, loader):
        cam_array = []
        df_camstats = pd.DataFrame([])
        camnet = GradCAM(net, target_layer)
        for idx, (img, lbl) in enumerate(loader):
            shape_lbls = loader.dataset.samples[idx]
            mask = camnet(img, index=int(lbl))
            cam_array.append(mask)
            if np.isnan(np.sum(mask)):
                nan_nums = True
            else:
                nan_nums = False
            if int(lbl) == int(camnet.pred):
                correct = True
            else:
                correct = False
            df = pd.DataFrame([{
                "img_name": os.path.basename(shape_lbls[0]),
                "img_path": shape_lbls[0],
                "net_output": list(camnet.output.cpu().detach().numpy()[0]),
                "label": bool(lbl),
                "prediction": bool(camnet.pred),
                "correct": correct,
                "nan_array": nan_nums,
            }])
            df_camstats = df_camstats.append(df)
        return cam_array, df_camstats
    net.eval()
    for param in net.parameters():
        param.requires_grad = True
    if train_data:
        CAM_ARR, DF = gcam_all(net, p.train_loader)
    else:
        CAM_ARR, DF = gcam_all(net, p.test_loader)
    return CAM_ARR, DF
