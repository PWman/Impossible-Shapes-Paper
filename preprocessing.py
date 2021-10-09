import os
import config
import random
import numpy as np
from math import floor, ceil
from torch import utils
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class Preprocess:
    def __init__(self, data_dir,
                 img_size=224, batch_size=16,
                 augment=True, scale_factor=None,
                 shuffle=True):
        self.img_size = img_size
        self.batch_size = batch_size
        self.data_dir = data_dir
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


def get_test_train_split(source_dir, split=0.2):
    categories = os.listdir(source_dir)
    category_1_imgs = []
    for file in os.listdir(os.path.join(source_dir, categories[0])):
        if ("_" not in file) and (".bmp" in file):
            category_1_imgs.append(file)
    category_2_imgs = []
    for file in os.listdir(os.path.join(source_dir, categories[1])):
        if ("_" not in file) and (".bmp" in file):
            category_2_imgs.append(file)
    all_img_names = list(zip(category_1_imgs, category_2_imgs))
    random.shuffle(all_img_names)
    train_imgs = all_img_names[int(round(split * 40)):]
    val_imgs = all_img_names[:int(round(split * 40))]
    return train_imgs, val_imgs


def get_pad_dist(img):
    def find_furtherst(lines, origin):
        max_dist = 0
        for r, c in zip(lines[0], lines[1]):
            dist = np.sqrt((r - origin[0]) ** 2 + (c - origin[1]) ** 2)
            if dist > max_dist:
                max_dist = dist
        return max_dist

    im_arr = np.array(img.convert("1"))
    lines = np.where(im_arr != 0)
    origin_coordinates = ((im_arr.shape[0] + 1) / 2, (im_arr.shape[1] + 1) / 2)
    max_dist = find_furtherst(lines, origin_coordinates)

    return ceil(max_dist - im_arr.shape[0] / 2)


def edit_background(img, pad_val):
    if pad_val > 0:
        img_edit = ImageOps.expand(img, border=pad_val, fill=0)
    elif pad_val < 0:
        pad_val = floor(np.abs(pad_val) / 2)
        lt = pad_val
        rb = 224 - pad_val
        img_edit = img.crop((lt, lt, rb, rb))
    else:
        img_edit = img
    return img_edit


def preprocess_save_imgs(images, source_dir, save_dir, rescale_background=False):
    categories = os.listdir(source_dir)

    category1_source_path = os.path.join(source_dir, categories[0])
    category2_source_path = os.path.join(source_dir, categories[1])
    category1_save_path = os.path.join(save_dir, categories[0])
    category2_save_path = os.path.join(save_dir, categories[1])
    config.check_make_dir(category1_save_path)
    config.check_make_dir(category2_save_path)

    for cat1_img, cat2_img in images:
        cat1_loc = os.path.join(category1_source_path, cat1_img)
        cat2_loc = os.path.join(category2_source_path, cat2_img)
        img_1 = ImageOps.invert(Image.open(cat1_loc)).resize((224, 224), Image.BICUBIC)
        img_2 = ImageOps.invert(Image.open(cat2_loc)).resize((224, 224), Image.BICUBIC)
        if rescale_background:
            pad_1 = get_pad_dist(img_1)
            img_1 = edit_background(img_1, pad_1).resize((224, 224), Image.BICUBIC)
            pad_2 = get_pad_dist(img_2)
            img_2 = edit_background(img_2, pad_2).resize((224, 224), Image.BICUBIC)
        img_1.save(os.path.join(save_dir, categories[0], cat1_img))
        img_2.save(os.path.join(save_dir, categories[1], cat2_img))


def create_preprocessed_directories(study_num):
    prepro_dir = os.path.join(config.prepro_dir, f"Study {study_num}")
    train_dir = os.path.join(prepro_dir, "Training")
    val_dir = os.path.join(prepro_dir, "Validation")
    config.check_make_dir(prepro_dir)
    config.check_make_dir(train_dir)
    config.check_make_dir(val_dir)


if __name__ == "__main__":
    random.seed(0)

    for study_num in range(3):
        create_preprocessed_directories(study_num)
        prepro_expt_dir = os.path.join(config.prepro_dir, f"Study {study_num}")
        train_dir = os.path.join(prepro_expt_dir, "Training")
        val_dir = os.path.join(prepro_expt_dir, "Validation")
        config.check_make_dir(prepro_expt_dir)
        config.check_make_dir(train_dir)
        config.check_make_dir(val_dir)
        if study_num>0:
            main_orig_dir = os.path.join(config.original_dir, "Main Study")
            train_imgs, val_imgs = get_test_train_split(main_orig_dir)
            preprocess_save_imgs(train_imgs, main_orig_dir, train_dir, rescale_background=False)
            preprocess_save_imgs(val_imgs, main_orig_dir, val_dir, rescale_background=False)
            preprocess_save_imgs(train_imgs, main_orig_dir, train_dir, rescale_background=True)
            preprocess_save_imgs(val_imgs, main_orig_dir, val_dir, rescale_background=True)
        else:
            ctrl_orig_dir = os.path.join(config.original_dir, "Control")
            ctrl_train, ctrl_val = get_test_train_split(ctrl_orig_dir)
            preprocess_save_imgs(ctrl_train, ctrl_orig_dir, train_dir, rescale_background=True)
            preprocess_save_imgs(ctrl_val, ctrl_orig_dir, val_dir, rescale_background=True)

