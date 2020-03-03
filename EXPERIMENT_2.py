import os
import config
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from preprocessing import Preprocess
from training_functions import train, set_seed


loss_fun = nn.CrossEntropyLoss()

for seed in range(config.num_seeds):
    print("\nTESTING SEED  " + str(seed+1) + "/" + str(config.num_seeds) + "...\n")
    set_seed(seed)
    for lr in config.param_dict["lr"]:
        for bs in config.param_dict["bs"]:

            p = Preprocess(img_size=224, batch_size=bs, split=0.2, colour=True)

            print("Testing with lr = " + str(lr) + ", batch size = " + str(bs))
            file_name = "lr" + str(lr) + "_bs" + str(bs) + "_s" + str(seed) + ".csv"


            gnet = models.googlenet(pretrained=True)
            for param in gnet.parameters():
                param.requires_grad = False
            gnet.fc = nn.Linear(1024, 2)

            opt = optim.SGD(gnet.fc.parameters(), lr=lr, momentum=0.9,weight_decay=0.0001)
            result = train(p, gnet, loss_fun, opt, config.device, epochs=config.num_epochs)
            result.to_csv(os.path.join(config.raw_dir_2, file_name))