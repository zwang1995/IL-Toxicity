from feature_extraction import Encoding
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import joblib


class GCM(nn.Module):
    def __init__(self, n1, n2, n3):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(n1, n2)
        self.l2 = nn.Linear(n2, n3)
        self.l3 = nn.Linear(n3, 1)

    def forward(self, input):
        hidden1 = F.sigmoid(self.l1(input))
        hidden2 = F.softplus(self.l2(hidden1))
        output = self.l3(hidden2)
        return output


index_group = {0: "[BH0|-]----",
               1: "[BrH0|-]",
               2: "[CH0|-]#",
               3: "[CH0|-]---",
               4: "[CH0|]-#",
               5: "[CH0|]----",
               6: "[CH0|]--=",
               7: "[CH1|]---",
               8: "[CH1|]-=",
               9: "[CH2|]--",
               10: "[CH3|]-",
               11: "[ClH0|-]",
               12: "[CoH0|]",
               13: "[FH0|]-",
               14: "[IH0|-]",
               15: "[NH0|+]----",
               16: "[NH0|+]--=",
               17: "[NH0|-]--",
               18: "[NH0|]#",
               19: "[NH0|]---",
               20: "[NH0|]-=",
               21: "[NH1|+]---",
               22: "[NH1|]--",
               23: "[NH2|+]--",
               24: "[NH2|]-",
               25: "[NH3|+]-",
               26: "[OH0|+]#",
               27: "[OH0|-]-",
               28: "[OH0|]--",
               29: "[OH0|]=",
               30: "[OH1|]-",
               31: "[PH0|+]----",
               32: "[PH0|-]------",
               33: "[PH0|]---=",
               34: "[SH0|+]---",
               35: "[SH0|-]-",
               36: "[SH0|]--==",
               37: "[cH0|]***",
               38: "[cH0|]-**",
               39: "[cH1|]**",
               40: "[nH0|+]-**",
               41: "[nH0|]-**"}

group_index = dict((index_group[i], i) for i in index_group)
feature_num = len(group_index)

# Load SMILES: method 1
# smiles = ["C[P+](C)(C)C.[Br-]"]

# Load SMILES: method 2
df = pd.read_csv("../data/SMILES_list_for_pred.csv", usecols=["smiles"])
smiles = df.smiles.tolist()

n_IL = len(smiles)
print(f"-> Total number of imported SMILES strings: {n_IL}")
feature_matrix = np.zeros((n_IL, feature_num))

n_valid = 0
for (i, smile) in enumerate(smiles):
    try:
        group_list = Encoding(smile)
        for group in group_list:
            try:
                j = group_index[group]
                feature_matrix[i, j] += 1
            except KeyError as e:
                print(f"ERROR: a new functional group {group} found in IL#{i + 1} with the SMILES {smile}")
                feature_matrix[i, :] = 0
                break
        else:
            n_valid += 1
    except Exception as e:
        feature_matrix[i, :] = 0

print(f"-> Features are successfully extracted for {n_valid} out of {n_IL} ILs")

if "FNN" == "FNN":
    print("-> FNN Model Activated")
    FNN_value = []
    model = torch.load("../output/FNN_v1/FNN_model.pkl")
    mean = 3.1156479399943664
    std = 1.0639953988691089
    for x in feature_matrix:
        if np.any(x):
            f = model(Variable(torch.Tensor(np.array(x))))
            f = f.detach().numpy().astype(float)[0] * std + mean
            FNN_value.append(f)
        else:
            FNN_value.append(None)
    print("-> FNN Prediction Completed")

if "SVM" == "SVM":
    print("-> SVM Model Activated")
    SVM_value = []
    model = joblib.load("../output/SVM_v1/SVM_model.pkl")
    for x in feature_matrix:
        if np.any(x):
            y = model.predict(np.array(x).reshape(1, -1))
            SVM_value.append(y[0])
        else:
            SVM_value.append(None)
    print("-> SVM Prediction Completed")

data = [(smile, FNN_v, SVM_v) for (smile, FNN_v, SVM_v) in zip(smiles, FNN_value, SVM_value)]
output_df = pd.DataFrame(data, columns=["smiles", "FNN_value", "SVM_value"])
output_df.to_csv("../output/pred_for_SMILES_list.csv")
print(output_df)
