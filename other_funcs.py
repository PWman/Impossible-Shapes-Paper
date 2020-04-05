import os
import config
import torch
import random
import numpy as np
from torchvision import transforms


def set_seed(seednum):
    torch.manual_seed(seednum)
    torch.cuda.manual_seed_all(seednum)
    np.random.seed(seednum)
    random.seed(seednum)
    return

def save_batch(X,bname,train_data):
    save_path = config.check_train_dir
    # if not os.path.isdir(save_path):
    #     os.mkdir(save_path)

    if train_data:
        save_path = os.path.join(save_path,"Train")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    else:
        save_path = os.path.join(save_path,"Test")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    for count, (img,lbl) in enumerate(X):
        save_name = os.path.join(save_path, bname + "img_" +
                                 str(count) + "_class_" +
                                 str(lbl.tolist()) + ".bmp")
        transforms.ToPILImage()(img).save(save_name)