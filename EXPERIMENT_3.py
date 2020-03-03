import os
import torch
import config
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torchvision import models, transforms
from preprocessing import Preprocess
from training_functions import train, set_seed

def get_cm(my_dir):
    cm = pd.DataFrame(columns=["seed", "actual_class",
                               "imp_preds", "poss_preds"])
    for class_idx, class_name in enumerate(os.listdir(my_dir)):
        subdir = os.path.join(my_dir, class_name)
        predictions = []

        for shape in os.listdir(subdir):
            img = Image.open(os.path.join(subdir, shape)).resize((224, 224)).convert("RGB")
            normed_img = transformations(img).view(-1, 3, 224, 224).to(device)
            net_out = net(normed_img.to(device)).tolist()[0]
            pred = net_out.index(max(net_out))
            predictions.append(pred)

        poss_preds = sum(predictions)
        imp_preds = len(predictions) - sum(predictions)

        cm_row = pd.DataFrame([{"seed": seed,
                                "actual_class": class_name,
                                "imp_preds": imp_preds,
                                "poss_preds": poss_preds}])

        cm = cm.append(cm_row, sort=True)

    return cm


def average_cm(all_cm_scores):
    all_imp = all_cm_scores[all_cm_scores["actual_class"] == "Impossible"]
    all_poss = all_cm_scores[all_cm_scores["actual_class"] == "Possible"]

    true_positive = sum(all_imp["imp_preds"])
    true_negative = sum(all_poss["poss_preds"])

    false_positive = sum(all_poss["imp_preds"])
    false_negative = sum(all_imp["poss_preds"])

    predicted_imp = pd.DataFrame([{
        "prediction": "Impossible",
        "actual_imp": true_positive,
        "actual_poss": false_positive,
    }])
    predicted_poss = pd.DataFrame([{
        "prediction": "Possible",
        "actual_imp": false_negative,
        "actual_poss": true_negative
    }])

    confusion_matrix = pd.concat([predicted_imp, predicted_poss])

    return confusion_matrix

# best_bs = 32
# best_lr = 0.001

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



p = Preprocess(img_size=224, batch_size=config.best_params["bs"], split=0.2, colour=True)
device = torch.device("cuda:0")
loss_func = nn.CrossEntropyLoss()


train_dir = os.path.join(os.getcwd(), "Shapes", "Shapes_Preprocessed", "Training")
val_dir = os.path.join(os.getcwd(), "Shapes", "Shapes_Preprocessed", "Validation")

all_cms_train = pd.DataFrame(columns=["seed", "actual_class",
                                      "imp_preds", "poss_preds"])

all_cms_val = pd.DataFrame(columns=["seed", "actual_class",
                                      "imp_preds", "poss_preds"])

for seed in range(config.num_seeds):

    print("\nTESTING SEED  " + str(seed+1) + "/" + str(config.num_seeds) + "...\n")

    set_seed(seed)
    net = models.resnet18(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    net.fc = nn.Linear(512, 2)
    net = net.to(device)

    opt = optim.SGD(net.fc.parameters(),lr=config.best_params["lr"], momentum=0.9,weight_decay=0.0001)
    result = train(p, net, loss_func, opt, device, epochs=100)

    cm_train = get_cm(train_dir)
    cm_val = get_cm(val_dir)

    all_cms_train = all_cms_train.append(cm_train, sort=True)
    all_cms_val = all_cms_val.append(cm_val, sort=True)

    torch.save(net.state_dict(),os.path.join(config.param_dir, "params_s" + str(seed) + ".pt"))

    net.eval()

all_cms_train.to_csv(os.path.join(config.results_dir_3, "Raw_CMs_Training.csv"))
all_cms_val.to_csv(os.path.join(config.results_dir_3, "Raw_CMs_Validation.csv"))

confusion_matrix_training = average_cm(all_cms_train)
confusion_matrix_validation = average_cm(all_cms_val)

confusion_matrix_training.to_csv(os.path.join(config.results_dir_3, "CM_Total_Training.csv"))
confusion_matrix_validation.to_csv(os.path.join(config.results_dir_3, "CM_Total_Validation.csv"))