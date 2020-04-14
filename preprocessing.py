import os
import config
import random
from torch import utils
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class Preprocess():
    # img_size = 224
    # batch_size = 16
    # split = 0.2
    # colour = False

    def __init__(self, img_size=224, batch_size=16, split=0.2,
                 augment=True,shuffle=True):
        # INITIALISE PARAMETERS
        self.img_size = img_size
        self.batch_size = batch_size
        self.split = split
        self.new_dir = config.prepro_dir

        # CREATE PREPROCESSED IMAGES
        if len(os.listdir(self.new_dir)) < 2:
            self.edit_imgs()

        # CREATE DATA AUGMENTER
        if augment:
            data_transforms = transforms.Compose([
                transforms.RandomRotation(360),
                transforms.RandomHorizontalFlip(),
                ################################
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            data_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        train_dir = os.path.join(self.new_dir, "Training")
        test_dir = os.path.join(self.new_dir, "Validation")

        self.train_dataset = datasets.ImageFolder(train_dir, data_transforms)
        self.train_loader = utils.data.DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=shuffle)
        self.train_class_names = self.train_dataset.classes
        self.train_len = len(self.train_dataset)

        self.test_dataset = datasets.ImageFolder(test_dir, data_transforms)
        self.test_loader = utils.data.DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=shuffle)
        self.test_class_names = self.test_dataset.classes
        self.test_len = len(self.test_dataset)


    def edit_imgs(self):
        # ITERATE THROUGH CATEGORIES
        source_dir = config.original_dir
        random.seed(0)


        train_dir = os.path.join(self.new_dir,"Training")
        val_dir = os.path.join(self.new_dir,"Validation")
        if not os.path.isdir(train_dir):
            os.mkdir(train_dir)
        if not os.path.isdir(val_dir):
            os.mkdir(val_dir)

        poss_imgs = []
        imposs_imgs = []

        for file in os.listdir(os.path.join(source_dir,"Possible")):
            if ("_" not in file) and (".bmp" in file):
                poss_imgs.append(file)
        for file in os.listdir(os.path.join(source_dir, "Impossible")):
            if ("_" not in file) and (".bmp" in file):
                imposs_imgs.append(file)

        images = list(zip(poss_imgs,imposs_imgs))
        random.shuffle(images)
        train_imgs = images[int(round(self.split*40)):]
        val_imgs = images[:int(round(self.split*40))]

        train_path_p = os.path.join(self.new_dir, "Training", "Possible")
        if not os.path.isdir(train_path_p):
            os.mkdir(train_path_p)
        train_path_i = os.path.join(self.new_dir, "Training", "Impossible")
        if not os.path.isdir(train_path_i):
            os.mkdir(train_path_i)

        for poss, imposs in train_imgs:
            poss_loc = os.path.join(os.path.join(source_dir, "Possible", poss))
            imposs_loc = os.path.join(os.path.join(source_dir, "Impossible", imposs))

            img_p = ImageOps.invert(Image.open(poss_loc)).resize((224,224), Image.BICUBIC)
            img_i = ImageOps.invert(Image.open(imposs_loc)).resize((224,224), Image.BICUBIC)

            img_p.save(os.path.join(train_path_p, poss))
            img_i.save(os.path.join(train_path_i, imposs))

        val_path_p = os.path.join(self.new_dir, "Validation", "Possible")
        if not os.path.isdir(val_path_p):
            os.mkdir(val_path_p)
        val_path_i = os.path.join(self.new_dir, "Validation", "Impossible")
        if not os.path.isdir(val_path_i):
            os.mkdir(val_path_i)

        for poss, imposs in val_imgs:
            poss_loc = os.path.join(os.path.join(source_dir, "Possible", poss))
            imposs_loc = os.path.join(os.path.join(source_dir, "Impossible", imposs))
##########################################################################################

            img_p = ImageOps.invert(Image.open(poss_loc)).resize((224,224),Image.BICUBIC)
            img_i = ImageOps.invert(Image.open(imposs_loc)).resize((224,224),Image.BICUBIC)

            img_p.save(os.path.join(val_path_p, poss))
            img_i.save(os.path.join(val_path_i, imposs))

if __name__ == "__main__":
    p = Preprocess(img_size=224, batch_size=16, split=0.2)
