import os
import config
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
# import torch.optim as optim
from preprocessing import Preprocess
from torchvision import models

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
        try:
            self.feature_extractor = FeatureExtractor(model.features,target_layers)
        except AttributeError:
            self.feature_extractor = FeatureExtractor(nn.Sequential(*list(model.children())[:-1]), target_layers)

        #nn.Sequential(*list(model.children())[:-1]),
        # target_layers)
        # print(nn.Sequential(*list(model.children())[:-1]))
    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        #####################################################
        try:
            output = self.model.classifier(output).cuda()
        except AttributeError:
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
#
#
# def show_cam_on_image(img, mask,save_loc,img_name):
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     cv2.imwrite(os.path.join(save_loc,img_name), np.uint8(255 * cam))
#

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

def get_cams_all(net,grad_cam):
    val_dir = os.path.join(config.prepro_dir,"Validation")


    ALL_CAMS = []
    for class_idx, path in enumerate(os.listdir(val_dir)):
        path = os.path.join(val_dir, path)
        print(path)
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file), 1)
            img = np.float32(cv2.resize(img, (224, 224))) / 255
            input = preprocess_image(img)

            net_out = list(np.squeeze(net(input.to(config.device)).tolist()))
            pred = net_out.index(max(net_out))

            if pred ==  class_idx:
                correct = 1
            else:
                correct = 0
            mask = grad_cam(input, class_idx)
            cam_dict = {"image": file, "mask": mask, "pred": pred, "correct":correct}
            ALL_CAMS.append(cam_dict)

    return ALL_CAMS
            # cam_dict[file] = net_out
            # predictions.append(pred)

if __name__ == "__main__":
    # net = models.vgg16(pretrained=True)
    # net.classifier[-1] = nn.Linear(4096, 2)
    # mdir = os.path.join(config.model_dir,"VGG16 (pretrained)")
    net = models.resnet50(pretrained=False)
    net.fc = nn.Linear(2048, 2)
    mdir = os.path.join(config.model_dir,"ResNet50")
    ALL_CAMS = []
    for param_file in os.listdir(mdir):
        net.load_state_dict(torch.load(os.path.join(
            mdir, param_file
        )))
        gradcam = GradCam(model=net,
                          target_layer_names=["4"],
                          device=config.device)
        # gradcam = GradCam(model=net,
        #                    target_layer_names=["29"],
        #                    device=config.device)#,use_cuda=True)#,
        # device=config.device
        cam_dicts = get_cams_all(net,gradcam)
        ALL_CAMS.append(cam_dicts)


    # all_imgs = os.listdir(os.path.join(val_dir,"Impossible")) + os.listdir(os.path.join("Possible"))
    val_dir = os.path.join(config.prepro_dir,"Validation")

    all_imgs = []
    for category in os.listdir(val_dir):
        for file in os.listdir(os.path.join(val_dir,category)):
            all_imgs.append(os.path.join(val_dir,category,file))

    for idx,img in enumerate(all_imgs):
        # MASK_SUM = np.zeros((224,224))
        MASK_SUM = []
        correct = 0
        for result in ALL_CAMS:
            correct = correct + result[idx]["correct"]
            mask = result[idx]["mask"]
            MASK_SUM.append(mask)

        plt.imshow(cv2.resize(cv2.imread(img),(224,224)),alpha=0.5)
        plt.imshow(np.mean(MASK_SUM,axis=0),alpha=0.5,cmap="jet")

        plt.title(f"Correct = {correct}")
        plt.savefig(os.path.join(config.gradcam_dir,os.path.basename(img).replace(".bmp",".png")))
        plt.clf()
    # p = Preprocess(augment=False)
    # net = models.vgg16(pretrained=True)
    # net.classifier[-1] = nn.Linear(4096, 2)
    # net.load_state_dict(torch.load(r"D:\peter\FinalShapes\Models\VGG16 (pretrained)\2.pt"))
    # for param in net.parameters():
    #     param.requires_grad = True
    # grad_cam = GradCam(model=net, target_layer_names=["29"],device=config.device)#,use_cuda=True)#, device=config.device
    # val_dir = os.path.join(config.prepro_dir,"Validation")
    #
    #
    # for class_idx, path in enumerate(os.listdir(val_dir)):
    #     path = os.path.join(val_dir, path)
    #     print(path)
    #     for file in os.listdir(path):
    #         img = cv2.imread(os.path.join(path, file), 1)
    #         img = np.float32(cv2.resize(img, (224, 224))) / 255
    #         input = preprocess_image(img)
    #
    #         net_out = list(np.squeeze(net(input.to(config.device)).tolist()))
    #         pred = net_out.index(max(net_out))
    #
    #         if pred == class_idx:
    #             mask = grad_cam(input, class_idx)
    #             show_cam_on_image(img, mask, config.gradcam_dir, file)
    """
def get_cams_all(net, grad_cam):
val_dir = os.path.join(config.prepro_dir, "Validation")

ALL_CAMS = []
for class_idx, path in enumerate(os.listdir(val_dir)):
    path = os.path.join(val_dir, path)
    print(path)
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(pa th, file), 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)

        net_out = list(np.squeeze(net(input.to(config.device)).tolist()))
        pred = net_out.index(max(net_out))

        mask = grad_cam(input, class_idx)
        cam_dict = {"image": file, "mask": mask, "pred": pred}
        ALL_CAMS.append(cam_dict)

return ALL_CAMS
"""
# from initialise_nets import make_models
# p = Preprocess(batch_size=1)
# net,opt = make_models("VGG16",pretrain=True)
# net.load_state_dict(torch.load(r"D:\peter\FinalShapes\Models\VGG16 (pretrained)\0.pt"))
#
# grad_cam = GradCam(model=net,
#                    target_layer_names=["29"])

# file_name = f"{os.path.basename(img)}_correct{ncorrect}"
# shape_img = Image.open(img).resize((224, 224)).convert("RGBA")
#
# hmap = np.mean(MASK_SUM, axis=0)
# cmap = plt.get_cmap("jet")
# hmap = cmap(hmap / np.amax(hmap))
# hmap = Image.fromarray(np.uint8(hmap * 225))
#
# mask_im = Image.blend(shape_img, hmap, 0.5)