import config
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from analysis_functions import average_results, fit_and_plot, plot_all_loss, get_best

plt.style.use("seaborn-bright")

result_dict = {}
all_scores = pd.DataFrame(columns=["lr","bs","acc","val_acc","loss","val_loss"])


for lr in config.param_dict["lr"]:
    for bs in config.param_dict["bs"]:
        net_name = "lr" + str(lr) + "_bs" + str(bs)
        avg_result, net_scores = average_results(config.raw_dir_2, net_name, 5)
        avg_result.to_csv(os.path.join(config.avg_dir_2, net_name + ".csv"))
        fit_and_plot(avg_result, config.graph_dir_2, net_name.replace("_pt", " Pretrained"))

        # net_scores = net_scores.dropna()
        # print(net_scores)
        score = pd.DataFrame([{
            "name": net_name, "lr": lr, "bs": bs,
            "acc": np.mean(net_scores["acc"]), "acc_std": np.std(avg_result["acc"]),
            "val_acc": np.mean(net_scores["val_acc"]), "val_acc_std": np.std(avg_result["val_acc"]),
            "loss": np.mean(net_scores["val_loss"]), "loss_std": np.std(avg_result["loss"]),
            "val_loss": np.mean(net_scores["val_loss"]), "val_loss_std": np.std(avg_result["val_loss"]),
        }])

        all_scores = all_scores.append(score, sort=False)
        result_dict[net_name] = avg_result


all_scores.to_csv(os.path.join(config.avg_dir_2,"Network Scores.csv"))

plt.clf()
ind_vars = all_scores[["lr","bs"]]
scale_fun = MinMaxScaler()
y = scale_fun.fit_transform(all_scores[["val_acc"]])
reg = RandomForestRegressor()
reg.fit(ind_vars,y.ravel())
pd.Series(reg.feature_importances_, index=["Learning Rate","Batch Size"]).plot.bar(color="skyblue",edgecolor="gray")
plt.ylabel('Regression Coefficient')#,fontsize=20)
plt.xlabel('Hyperparameter')#, fontsize=20)
plt.xticks(rotation="horizontal")#,fontsize=20)
plt.title('Parameter Importances')#, fontsize=20)
plt.savefig(os.path.join(config.graph_dir_2,"Hyperparameter Regression"))

plt.clf()
sns.violinplot(x='lr',y='val_acc',data=all_scores.reset_index(),
               color="skyblue", inner='stick')
plt.title('Validation Accuracy as Function of Learning Rate')
plt.xlabel("Learning Rate")
plt.ylabel("Validation Accuracy")
plt.savefig(os.path.join(config.graph_dir_2,"Learning Rate Plot"))
plt.clf()

sns.violinplot(x='bs',y='val_acc',data=all_scores.reset_index(),
               color="skyblue", inner='stick')
plt.title('Batch Size as Function of Learning Rate')
plt.xlabel("Batch Size")
plt.ylabel("Validation Accuracy")
plt.savefig(os.path.join(config.graph_dir_2,"Batch Size Plot"))
plt.clf()

grid = all_scores.reset_index().groupby(['lr','bs']).val_acc.mean().unstack()
sns.heatmap(grid,cmap=sns.color_palette("coolwarm", 7),annot=True)
plt.title('Learning Rate vs Batch Size')
plt.xlabel("Batch Size")
plt.ylabel("Learning Rate")
plt.savefig(os.path.join(config.graph_dir_2,"Learning Rate vs Batch Size"))


all_scores["legend"] = all_scores["name"].replace({
    "lr": "learning rate = ","_bs": ", batch size = "},
    regex=True)

best_scores = get_best(all_scores, 5)

path_name = os.path.join(config.graph_dir_2, "Best Losses")
plot_all_loss(result_dict,best_scores,path_name)