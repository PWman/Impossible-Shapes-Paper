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

model_dir = os.path.join(os.getcwd(),"Models")
shapes_basedir = os.path.join(os.getcwd(), "Shapes")
results_basedir = os.path.join(os.getcwd(), "Results")

original_dir = os.path.join(shapes_basedir,"Original")
prepro_dir = os.path.join(shapes_basedir, "Preprocessed")
check_train_dir = os.path.join(shapes_basedir,"Check_Training_Images")

table_dir = os.path.join(results_basedir, "Tables")
raw_dir = os.path.join(table_dir, "Raw")
avg_dir = os.path.join(table_dir, "Averaged")

graph_dir = os.path.join(results_basedir, "Graphs")
acc_dir = os.path.join(graph_dir, "Accuracy")
loss_dir = os.path.join(graph_dir, "Loss")

gradcam_dir = os.path.join(results_basedir,"GradCAM")
cm_dir = os.path.join(table_dir,"Confusion_Matrices")
# raw_cm_dir = os.path.join(cm_dir,"Raw")
# avg_cm_dir = os.path.join(cm_dir, "Averaged")
check_make_dir(cm_dir)
check_make_dir(gradcam_dir)
# check_make_dir(raw_cm_dir)
# check_make_dir(avg_cm_dir)


check_make_dir(model_dir)
check_make_dir(shapes_basedir)
check_make_dir(results_basedir)

check_make_dir(prepro_dir)
check_make_dir(check_train_dir)

check_make_dir(table_dir)
check_make_dir(raw_dir)
check_make_dir(avg_dir)

check_make_dir(graph_dir)
check_make_dir(acc_dir)
check_make_dir(loss_dir)

[check_make_dir(os.path.join(model_dir, net)) for net in DNNs]
[check_make_dir(os.path.join(model_dir, net + " (pretrained)")) for net in DNNs]

[check_make_dir(os.path.join(raw_dir, net)) for net in DNNs]
[check_make_dir(os.path.join(raw_dir, net + " (pretrained)")) for net in DNNs]

"""
shape_dir_names = ["original", "preprocessed", "check_training"]
[check_make_dir(os.path.join(shapes_basedir, subdir)) for subdir in shape_dir_names]




acc_dir = os.path.join(graph_dir, "accuracy")
loss_dir = os.path.join(graph_dir,"loss")
check_make_dir(acc_dir)
check_make_dir(loss_dir)
# result_dir_names = ["tables", "graphs"]
#
# [check_make_dir(os.path.join(results_basedir, subdir)) for subdir in result_dir_names]


[check_make_dir(os.path.join(model_dir, net)) for net in DNNs]
[check_make_dir(os.path.join(table_dir, net)) for net in DNNs]
"""