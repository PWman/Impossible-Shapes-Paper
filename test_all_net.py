import os
import config
import torch
from torch import nn
from torch import optim
from torchvision import models
from preprocessing import Preprocess
from training_functions import set_seed, train_net
from initialise_nets import make_models


if __name__ == "__main__":
    # ALL_NETS = config.DNNs
    ALL_NETS = ["GoogLeNet"]
    p = Preprocess()
    for net_name in ALL_NETS:
        print(f"\nTesting {net_name}...\n")

        for seed in range(config.num_seeds):
            print(f"Testing seed {seed}...")
            set_seed(seed)
            net, opt = make_models(net_name, pretrain=False)
            result = train_net(p,net,opt)
            result.to_csv(os.path.join(
                config.raw_dir, net_name, f"{seed}.csv"
            ))
            torch.save(net.state_dict(), os.path.join(
                config.model_dir, net_name, str(seed) + ".pt"
            ))

            net_name_pt = f"{net_name} (pretrained)"
            net, opt = make_models(net_name, pretrain=True)
            result = train_net(p,net,opt)
            result.to_csv(os.path.join(
                config.raw_dir, net_name_pt, f"{seed}.csv"
            ))
            torch.save(net.state_dict(), os.path.join(
                config.model_dir, net_name_pt, f"{seed}.pt"
            ))
