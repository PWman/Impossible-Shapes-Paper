import os
import torch

# import numpy as np
def check_make_dir(data_dir):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


DNNs = ["AlexNet", "VGG11", "VGG16", "ResNet18", "ResNet50", "GoogLeNet"]
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
    "GoogLeNet (pretrained)": ["15"],
}

shapes_basedir = os.path.join(os.getcwd(), "Shapes")
results_basedir = os.path.join(os.getcwd(), "Results")
check_make_dir(shapes_basedir)
check_make_dir(results_basedir)

original_dir = os.path.join(shapes_basedir,"Original")
prepro_dir = os.path.join(shapes_basedir, "Preprocessed")
check_train_dir = os.path.join(shapes_basedir,"Check_Training_Images")
check_make_dir(original_dir)
check_make_dir(prepro_dir)
check_make_dir(check_train_dir)



table_dir = os.path.join(results_basedir, "Tables")
raw_dir = os.path.join(table_dir, "Raw")
avg_dir = os.path.join(table_dir, "Averaged")
check_make_dir(table_dir)
check_make_dir(raw_dir)
check_make_dir(avg_dir)

graph_dir = os.path.join(results_basedir, "Graphs")
acc_dir = os.path.join(graph_dir, "Accuracy")
loss_dir = os.path.join(graph_dir, "Loss")
check_make_dir(graph_dir)
check_make_dir(acc_dir)
check_make_dir(loss_dir)
#
cm_dir = os.path.join(table_dir,"Confusion_Matrices")
gradcam_dir = os.path.join(results_basedir,"GradCAM")
check_make_dir(cm_dir)
check_make_dir(gradcam_dir)

model_dir = os.path.join(os.getcwd(),"Models")
check_make_dir(model_dir)

[check_make_dir(os.path.join(model_dir, net)) for net in DNNs]
[check_make_dir(os.path.join(model_dir, net + " (pretrained)")) for net in DNNs]

#
[check_make_dir(os.path.join(raw_dir, net)) for net in DNNs]
[check_make_dir(os.path.join(raw_dir, net + " (pretrained)")) for net in DNNs]

