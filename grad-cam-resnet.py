import os
import  config
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from preprocessing import Preprocess
from training_functions import train

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
        self.feature_extractor = FeatureExtractor(nn.Sequential(*list(model.children())[:-1]), target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        #####################################################
        output = self.model.fc(output).cuda()
        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask,save_loc,img_name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(os.path.join(save_loc,img_name), np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, device):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):

        features, output = self.extractor(input.to(self.device))

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        # print(self.extractor.get_gradients())
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

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam-1.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)



#
# vgg_model =  models.vgg11()
# vgg_model.classifier[-1] = nn.Linear(4096, 2)
# vgg_model.load_state_dict(torch.load(
#     os.path.join(r"C:\Users\peter\OneDrive\Documents\MyProjPub\best_params.pt")
# ))
# vgg_model.eval()
#
# grad_cam = GradCam(model=vgg_model, target_layer_names=["19"], device=config.device)

pre = Preprocess(img_size=224, batch_size=16, split=0.2, colour=True)

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 2)

loss_fun = nn.CrossEntropyLoss()
opt = optim.Adam(model.fc.parameters())
result = train(pre,model,loss_fun,opt,config.device,epochs=10)


grad_cam = GradCam(model=model, target_layer_names=["4"], device=config.device)

paths = [os.path.join(r"C:\Users\peter\OneDrive\Documents\MyProjPub\Shapes\Shapes_Preprocessed\Validation",
                      "Impossible"),
         os.path.join(r"C:\Users\peter\OneDrive\Documents\MyProjPub\Shapes\Shapes_Preprocessed\Validation",
                      "Possible")]

save_loc = os.path.join(r"C:\Users\peter\OneDrive\Documents\MyProjPub","GRADCAM RESULTS")
if not os.path.isdir(save_loc):
    os.mkdir(save_loc)

for class_idx, path in enumerate(paths):
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path,file),1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)

        net_out = list(np.squeeze(model(input.to(config.device)).tolist()))
        pred = net_out.index(max(net_out))

        if pred == class_idx:
            target_index = class_idx
            mask = grad_cam(input, target_index)
            show_cam_on_image(img, mask,save_loc,file)