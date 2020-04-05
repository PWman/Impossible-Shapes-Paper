import os
import config
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
# import torch.optim as optim
from torchvision import models, transforms
from preprocessing import Preprocess
from training_functions import train_net
import matplotlib.pyplot as plt
from PIL import Image
from initialise_nets import make_models
# net = models.googlenet(pretrained=True)


# net = models.vgg16()
# net.classifier[-1] = nn.Linear(4096,2)
# net.load_state_dict(torch.load(r"D:\peter\FinalShapes\Models\VGG16 (pretrained)\0.pt"))

class GradCAM(nn.Module):
    def __init__(self,net):

        super(GradCAM, self).__init__()
        self.net = net
        self.gradients = None
        try:
            self.features = self.net.features
            self.pooling = self.net.avgpool
            self.classifier = self.net.classifier
            print("1")
        except AttributeError:
            print("2")
            ### Need to change  this to get working with all
            if type(net) == models.resnet.ResNet:
                self.features = nn.Sequential(*list(net.children())[:-2])
                self.pooling = self.net.avgpool #nn.Sequential(*list(net.children())[-3])
                self.classifier = nn.Sequential(*list(net.children())[-1:])
            else: #type(net) == models.googlenet.GoogLeNet:
                self.features = nn.Sequential(*list(net.children())[:-3])
                # print(self.features)
                self.pooling = net.avgpool
                self.classifier = nn.Sequential(*list(net.children())[-1:])

    def activations_hook(self,grad):
        self.gradients = grad

    def forward(self,x):
        x = self.features(x)#.view(-1,3,224,224))
        h = x.register_hook(self.activations_hook)
        # print(x.view(1,-1).shape)
        x = self.pooling(x)
        x = self.classifier(x.view(1, -1))
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features(x)

def get_cam(camnet, torch_img):
    # camnet = GradCAM(net)
    piltrans = transforms.ToPILImage()
    # torch_img = img.view(-1,3,224,224)

    net_out = camnet(torch_img)#)
    pred = net_out.argmax(dim=1)
    # get the gradient of the output with respect to the parameters of the model
    net_out[:, pred].backward()
    # pull the gradients out of the model
    gradients = camnet.get_activations_gradient()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # get the activations of the last convolutional layer
    activations = camnet.get_activations(torch_img).detach()
    # weight the channels by corresponding gradients
    for i in range(len(pooled_gradients)):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()# average the channels of the activations
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    # normalize the heatmap
    # plt.matshow(heatmap.squeeze())# draw the heatmap
    hmap_pil = piltrans(heatmap)
    hmap = hmap_pil.resize((224,224),resample=Image.BICUBIC)
    return np.array(hmap), pred

def cams_all_imgs(net,loader):
    net = GradCAM(net)
    CAM_DICT = []
    for idx,(img,lbl) in enumerate(loader):
        # net.classifier.zero_grad()
        # net.feaatures.zero_grad()
        hmap, pred = get_cam(net,img)
        img_path, _ = loader.dataset.samples[idx]
        if lbl == pred:
            correct = 1
        else:
            correct = 0
        print(f"pred={pred}, lbl={lbl}")
        CAM_DICT.append({
            "image_path": img_path, "correct": correct, "CAM": hmap
        })
    return CAM_DICT

def get_avg_cam(net_name,pretrain=False):

    if pretrain:
        sdir = os.path.join(config.gradcam_dir, f"{net_name} (pretrained)")
        mdir = os.path.join(config.model_dir, f"{net_name} (pretrained)")

    else:
        sdir = os.path.join(config.gradcam_dir,net_name)
        mdir = os.path.join(config.model_dir, net_name)
    vdir = os.path.join(config.prepro_dir,"Validation")

    config.check_make_dir(sdir)

    p = Preprocess(batch_size=1,augment=False,shuffle=False)
    # net_name = "ResNet50"
    net,opt = make_models(net_name,pretrain=pretrain)

    for param in net.parameters():
        param.requires_grad = True
    net.eval()

    ALL_CAMS = []
    for file in os.listdir(mdir):
        net.load_state_dict(torch.load(os.path.join(mdir, file)))
        cams = cams_all_imgs(net,p.test_loader)
        ALL_CAMS.append(cams)

    all_imgs = []
    for category in os.listdir(vdir):
        for file in os.listdir(os.path.join(vdir,category)):
            all_imgs.append(os.path.join(vdir,category,file))

    for idx, img in enumerate(all_imgs):
        # MASK_SUM = np.zeros((224,224))
        MASK_SUM = []
        correct = 0
        for result in ALL_CAMS:

            correct = correct + result[idx]["correct"]
            if result[idx]["correct"] == 1:
                mask = result[idx]["CAM"]
                MASK_SUM.append(mask)
        if correct>0:
            img_name = os.path.basename(img).replace(".bmp","")
            file_name = f"{img_name}_correct{correct}.png"
            shape_img = Image.open(img).resize((224,224)).convert("RGBA")

            hmap = np.mean(MASK_SUM,axis=0)
            cmap = plt.get_cmap("jet")
            hmap = cmap(hmap / np.amax(hmap))
            hmap = Image.fromarray(np.uint8(hmap*225))

            mask_img = Image.blend(shape_img,hmap,0.5)
            mask_img.save(os.path.join(sdir,file_name))

    return

if __name__ == "__main__":
    # net = models.resnet50()
    # net.fc = nn.Linear(2048, 2)
    # # net.load_state_dict(torch.load(r"D:\peter\FinalShapes\Models\ResNet50 (pretrained)\0.pt"))
    # get_avg_cam("VGG11",pretrain=True)
    get_avg_cam("GoogLeNet",pretrain=True)
    get_avg_cam("VGG11",pretrain=True)
    get_avg_cam("ResNet50",pretrain=False)
    # piltrans = transforms.ToPILImage()
    # val_dir = os.path.join(config.prepro_dir,"Validation")
    #
    #
    # p = Preprocess(batch_size=1,augment=False,shuffle=False)
    #
    # net_name = "ResNet50"
    # net,opt = make_models(net_name,preeetrain=False)
    # for param in net.parameters():
    #     param.requires_grad = True
    #
    # net.eval()
    # ALL_CAMS = []
    # mdir = os.path.join(config.model_dir,net_name)
    # for file in os.listdir(mdir):
    #     net.load_state_dict(torch.load(os.path.join(mdir, file)))
    #     cams = cams_all_imgs(net,p.test_loader)
    #     ALL_CAMS.append(cams)
    #
    #
    # all_imgs = []
    # for category in os.listdir(val_dir):
    #     for file in os.listdir(os.path.join(val_dir,category)):
    #         all_imgs.append(os.path.join(val_dir,category,file))
    #
    # for idx,img in enumerate(all_imgs):
    #     # MASK_SUM = np.zeros((224,224))
    #     MASK_SUM = []
    #     correct = 0
    #     for result in ALL_CAMS:
    #         correct = correct + result[idx]["correct"]
    #         mask = result[idx]["CAM"]
    #         MASK_SUM.append(mask)
    #
    #     plt.imshow(cv2.resize(cv2.imread(img),(224,224)),alpha=0.5)
    #     plt.imshow(np.mean(MASK_SUM,axis=0),alpha=0.5,cmap="jet")
    #
    #     plt.title(f"Correct = {correct}")
    #     plt.savefig(os.path.join(config.gradcam_dir,os.path.basename(img).replace(".bmp",".png")))
    #     plt.clf()
