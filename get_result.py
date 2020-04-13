import config
import pandas as pd
from more_utils import set_seed
from gcam_utils import gcam_all_imgs
from preprocessing import Preprocess
from net_utils import make_models, train_net, get_cm_result






    # cm_result = get_cm_result(p,net)
    # cam_result = gcam_all_imgs()

# def get_result_single(net_name,seed_num,num_epochs=None,batch_size=16,EStop=None):
#     set_seed(seed_num)
#     net,opt = make_models(net_name)
#     p = Preprocess(batch_size=batch_size)
#     train_results = pd.DataFrame(columns=["epoch", "acc", "loss",
#                                     "val_acc", "val_loss"])
#     for epoch in num_epochs:
#         acc,loss,v_acc,v_loss = train_epoch(p,net,opt)
#         r = pd.DataFrame([{
#             "epoch": epoch,
#             "acc": acc,
#             "val_acc": v_acc,
#             "loss": loss,
#             "val_loss": v_loss,
#         }])
#         train_results = train_results.append(r, sort=True)
#