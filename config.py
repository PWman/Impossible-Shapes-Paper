import os
import torch


def check_make_dir(data_dir):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

batch_size = 16
num_seeds = 20
num_epochs = 100
loss_fun = torch.nn.CrossEntropyLoss()

target_layers = {
    "AlexNet": ["11"],
    "VGG11": ["19"],
    "VGG16": ["29"],
    "ResNet18": ["7"],
    "ResNet50": ["7"],
    "GoogLeNet": ["15"],
    "AlexNet (pretrained)": ["11"],
    "VGG11 (pretrained)": ["19"],
    "VGG16 (pretrained)": ["29"],
    "ResNet18 (pretrained)": ["7"],
    "ResNet50 (pretrained)": ["7"],
    "GoogLeNet (pretrained)": ["15"]
}
DNNs = list(target_layers.keys())

results_basedir = os.path.join(os.getcwd(), "Results")
shapes_basedir = os.path.join(os.getcwd(), "Shapes")
check_make_dir(results_basedir)
check_make_dir(shapes_basedir)

raw_dir = os.path.join(results_basedir, "Raw")
original_dir = os.path.join(shapes_basedir, "Original")
prepro_dir = os.path.join(shapes_basedir, "Preprocessed")
check_make_dir(raw_dir)
check_make_dir(original_dir)
check_make_dir(prepro_dir)

image_checks = os.path.join(shapes_basedir, "Image_Checking")
fully_prepro_dir = os.path.join(image_checks,"Fully_Preprocessed")
bg_segment_dir = os.path.join(image_checks,"Background_Segmentation")
check_make_dir(image_checks)
check_make_dir(fully_prepro_dir)
check_make_dir(bg_segment_dir)

for study_num in range(3):
    check_make_dir(os.path.join(raw_dir, f"Study {study_num}"))
    check_make_dir(os.path.join(results_basedir, f"Study {study_num}"))
