import os
import torch
import numpy as np

def check_make_dir(data_dir):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)


num_seeds = 20
num_epochs = 100

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    torch.device("cpu")

net_names = ["AlexNet", "VGG16", "ResNet50", "GoogLeNet", "AlexNet Pretrained",
             "VGG16 Pretrained", "ResNet50 Pretrained", "GoogLeNet Pretrained"]
# net_names = ["AlexNet", "VGG11", "ResNet18", "AlexNet_pt", "VGG11_pt", "ResNet18_pt"]

param_dict = {
    "lr": [0.01, 0.001, 1e-4, 1e-5, 1e-6],
    "bs": [2, 4, 8, 16, 32, 64]
}

best_params = {
    "name": "lr0.001_bs32",
    "lr": 0.001,
    "bs": 16
}


shapes_dir = os.path.join(os.getcwd(), "Shapes")
preprodir = os.path.join(os.getcwd(), "")


results_dir_1 = os.path.join(os.getcwd(),"Expt1 results")
results_dir_2 = os.path.join(os.getcwd(),"Expt2 results")
results_dir_3 = os.path.join(os.getcwd(),"Expt3 results")
check_make_dir(results_dir_1)
check_make_dir(results_dir_2)
check_make_dir(results_dir_3)

raw_dir_1 = os.path.join(results_dir_1,"Raw data")
raw_dir_2 = os.path.join(results_dir_2,"Raw data")
check_make_dir(raw_dir_1)
check_make_dir(raw_dir_2)

avg_dir_1 = os.path.join(results_dir_1,"Average results")
avg_dir_2 = os.path.join(results_dir_2,"Average results")
check_make_dir(avg_dir_1)
check_make_dir(avg_dir_2)

graph_dir_1 = os.path.join(results_dir_1, "Graphs")
graph_dir_2 = os.path.join(results_dir_2, "Graphs")
check_make_dir(graph_dir_1)
check_make_dir(graph_dir_2)

acc_dir_1 = os.path.join(graph_dir_1, "Accuracy")
acc_dir_2 = os.path.join(graph_dir_2, "Accuracy")
check_make_dir(acc_dir_1)
check_make_dir(acc_dir_2)

loss_dir_1 = os.path.join(graph_dir_1, "Loss")
loss_dir_2 = os.path.join(graph_dir_2,  "Loss")
check_make_dir(loss_dir_1)
check_make_dir(loss_dir_2)

cam_dir = os.path.join(results_dir_3,"grad-CAM")
check_make_dir(cam_dir)

param_dir = os.path.join(results_dir_3,"net_params")
check_make_dir(param_dir)