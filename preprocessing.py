import os
import config
import random
import numpy as np
import matplotlib.pyplot as plt
from math import floor,ceil
from torch import utils
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class Preprocess:
    def __init__(self, data_dir,
                 img_size=224, batch_size=16,
                 augment=True, scale_factor=None,
                 shuffle=True):
        # INITIALISE PARAMETERS
        self.img_size = img_size
        self.batch_size = batch_size
        self.data_dir = data_dir

        # CREATE DATA AUGMENTER
        if augment:
            if scale_factor is None:
                scale_factor = (1, 1)
            elif isinstance(scale_factor, (float, int)):
                if scale_factor < 1:
                    scale_factor = (scale_factor, 1)
                else:
                    scale_factor = (1, scale_factor)
            else:
                print("Please input scale factor as float or int")

            data_transforms = transforms.Compose([
                transforms.RandomAffine(degrees=360, scale=scale_factor, resample=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            data_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        train_dir = os.path.join(self.data_dir, "Training")
        test_dir = os.path.join(self.data_dir, "Validation")

        self.train_dataset = datasets.ImageFolder(train_dir, data_transforms)
        self.train_loader = utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        self.train_class_names = self.train_dataset.classes
        self.train_len = len(self.train_dataset)

        self.test_dataset = datasets.ImageFolder(test_dir, data_transforms)
        self.test_loader = utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=shuffle)
        self.test_class_names = self.test_dataset.classes
        self.test_len = len(self.test_dataset)



# def edit_imgs(split=0.2):
def get_test_train_split(split=0.2):
    source_dir = config.original_dir
    poss_imgs = []
    for file in os.listdir(os.path.join(source_dir,"Possible")):
        if ("_" not in file) and (".bmp" in file):
            poss_imgs.append(file)
    imposs_imgs = []
    for file in os.listdir(os.path.join(source_dir, "Impossible")):
        if ("_" not in file) and (".bmp" in file):
            imposs_imgs.append(file)
    all_img_names = list(zip(imposs_imgs,poss_imgs))
    random.shuffle(all_img_names)
    train_imgs = all_img_names[int(round(split*40)):]
    val_imgs = all_img_names[:int(round(split*40))]

    return train_imgs, val_imgs

def get_pad_dist(img):
    def find_furtherst(lines,origin):
        max_dist = 0
        for r,c in zip(lines[0],lines[1]):
            dist = np.sqrt((r-origin[0])**2 + (c-origin[1])**2)
            if dist>max_dist:
                max_dist = dist
        return max_dist

    im_arr = np.array(img.convert("1"))
    lines = np.where(im_arr != 0)
    origin_coordinates = ((im_arr.shape[0]+1)/2, (im_arr.shape[1]+1)/2)

    max_dist = find_furtherst(lines,origin_coordinates)

    return ceil(max_dist - im_arr.shape[0]/2)

def edit_background(img,pad_val):
    if pad_val>0:
        img_edit = ImageOps.expand(img, border=pad_val, fill=0)
    elif pad_val<0:
        pad_val = floor(np.abs(pad_val)/2)
        lt = pad_val
        rb = 224-pad_val
        img_edit = img.crop((lt,lt,rb,rb))
    else:
        img_edit = img
    return img_edit


def preprocess_save_imgs(images,save_dir,rescale_background=False):
    poss_save_path = os.path.join(save_dir, "Possible") ###########
    imp_save_path = os.path.join(save_dir, "Impossible")
    config.check_make_dir(poss_save_path)
    config.check_make_dir(imp_save_path)

    for imp_img, poss_img in images:
        imposs_loc = os.path.join(config.original_dir,"Impossible",imp_img)
        poss_loc = os.path.join(config.original_dir,"Possible",poss_img)
        img_p = ImageOps.invert(Image.open(poss_loc)).resize((224,224), Image.BICUBIC)
        img_i = ImageOps.invert(Image.open(imposs_loc)).resize((224,224), Image.BICUBIC)

        if rescale_background:
            pad_p = get_pad_dist(img_p)
            img_p = edit_background(img_p, pad_p).resize((224,224), Image.BICUBIC)
            pad_i = get_pad_dist(img_i)
            img_i = edit_background(img_i, pad_i).resize((224,224), Image.BICUBIC)

        img_p.save(os.path.join(poss_save_path, poss_img))
        img_i.save(os.path.join(imp_save_path, imp_img))

    return


if __name__ == "__main__":
    random.seed(0)
    prepro_expt1_dir = os.path.join(config.prepro_dir, "Study 1")
    train_dir_expt1 = os.path.join(prepro_expt1_dir, "Training")
    val_dir_expt1 = os.path.join(prepro_expt1_dir, "Validation")
    config.check_make_dir(prepro_expt1_dir)
    config.check_make_dir(train_dir_expt1)
    config.check_make_dir(val_dir_expt1)

    prepro_expt2_dir = os.path.join(config.prepro_dir, "Study 2")
    train_dir_expt2 = os.path.join(prepro_expt2_dir, "Training")
    val_dir_expt2 = os.path.join(prepro_expt2_dir, "Validation")
    config.check_make_dir(prepro_expt2_dir)
    config.check_make_dir(train_dir_expt2)
    config.check_make_dir(val_dir_expt2)

    train_imgs, val_imgs = get_test_train_split()
    preprocess_save_imgs(train_imgs,save_dir=train_dir_expt1,rescale_background=False)
    preprocess_save_imgs(val_imgs,save_dir=val_dir_expt1,rescale_background=False)

    preprocess_save_imgs(train_imgs,save_dir=train_dir_expt2,rescale_background=True)
    preprocess_save_imgs(val_imgs,save_dir=val_dir_expt2,rescale_background=True)

