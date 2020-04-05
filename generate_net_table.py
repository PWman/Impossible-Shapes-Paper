import pandas as pd
df = pd.read_csv(
    r"C:\Users\peter\OneDrive\Documents\Impossible-Shapes-Paper\Expt1 results\Average results\Network Scores.csv")

final_df = pd.DataFrame(columns=["Network", "Accuracy", "Validation Accuracy", "Loss", "Validation Loss"])

for idx, row in df.iterrows():
    l = str(round(row["loss"], 3)) + "+/-" + str(round(row["loss_std"], 3))
    vl = str(round(row["val_loss"], 3)) + "+/-" + str(round(row["val_loss_std"], 3))
    a = str(round(row["acc"], 3)) + "+/-" + str(round(row["acc_std"], 2))
    va = str(round(row["val_acc"], 3)) + " +/- " + str(round(row["val_acc_std"], 3))
    final_df = final_df.append(pd.DataFrame([{"Network": row["name"], "Accuracy": a, "Validation Accuracy": va,
                                              "Loss": l, "Validation Loss": vl}]))

final_df.to_csv("ALLNETSTABLE.csv")