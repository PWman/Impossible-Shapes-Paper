import config
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms
from initialise_nets import make_models
from preprocessing import Preprocess

class FeatureExtractor():
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
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
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
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


# def preprocess_image(img):
#     means = [0.485, 0.456, 0.406]
#     stds = [0.229, 0.224, 0.225]
#
#     preprocessed_img = img.copy()[:, :, ::-1]
#     for i in range(3):
#         preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
#         preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
#     preprocessed_img = \
#         np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
#     preprocessed_img = torch.from_numpy(preprocessed_img)
#     preprocessed_img.unsqueeze_(0)
#     input = preprocessed_img.requires_grad_(True)
#     return input


# def show_cam_on_image(img, mask):
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names):
        self.model = model
        for layer in self.model.parameters():
            layer.requires_grad = True
        self.model.eval()
        self.model.to(config.device)

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if torch.cuda.is_available():
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        self.output = output
        # if self.cuda:
        #     features, output = self.extractor(input.cuda())
        # else:
        #     features, output = self.extractor(input)

        if index == None:
            print()
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if torch.cuda.is_available():
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
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




# def deprocess_image(img):
#     """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
#     img = img - np.mean(img)
#     img = img / (np.std(img) + 1e-5)
#     img = img * 0.1
#     img = img + 0.5
#     img = np.clip(img, 0, 1)
#     return np.uint8(img*255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.

    net,opt = make_models("VGG16",pretrain=True)
    net.load_state_dict(torch.load(r"D:\peter\FinalShapes\Models\VGG16 (pretrained)\0.pt"))
    grad_cam = GradCam(model=net,
                       target_layer_names=["29"])

    p = Preprocess(batch_size=1,shuffle=False)
    for img,lbl in p.test_loader:
        break

    mask = grad_cam(img)
    # img = cv2.imread(args.image_path, 1)
    # img = np.float32(cv2.resize(img, (224, 224))) / 255
    # input = preprocess_image(img)
    #
    # # If None, returns the map for the highest scoring category.
    # # Otherwise, targets the requested index.
    # target_index = None
    # mask = grad_cam(input, target_index)
    #
    # show_cam_on_image(img, mask)
    #
    #
    # cam_mask = cv2.merge([mask, mask, mask])
    #
    # cv2.imwrite('cam_gb.jpg', cam)