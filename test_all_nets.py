import os
import torch
import config
import pandas as pd
from more_utils import set_seed
from gcam_utils import gcam_all_imgs
from preprocessing import Preprocess
from net_utils import make_models, train_net, get_cm_result


if __name__ == "__main__":
    ALL_NETS = config.DNNs + [dnn + "(pretrained)" for dnn in config.DNNs]
    # ALL_NETS = ["GoogLeNet"]
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

            # net_name_pt = f"{net_name} (pretrained)"
            # net, opt = make_models(net_name, pretrain=True)
            # result = train_net(p,net,opt)
            # result.to_csv(os.path.join(
            #     config.raw_dir, net_name_pt, f"{seed}.csv"
            # ))
            # torch.save(net.state_dict(), os.path.join(
            #     config.model_dir, net_name_pt, f"{seed}.pt"
            # ))
