# hyper-parameter optimization for FNN models


import os
import csv
import numpy as np
import pandas as pd
from collections import Counter
from rdkit import Chem
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import KFold
from sklearn import metrics
from early_stopping import EarlyStopping


class Molecule_reader():
    def __init__(self, no, name, smiles, property, set):
        self.no = no
        self.name = name
        self.smiles = smiles
        self.property = property
        self.set = set


class FNN(nn.Module):
    def __init__(self, n1, n2, n3):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(n1, n2)
        self.l2 = nn.Linear(n2, n3)
        self.l3 = nn.Linear(n3, 1)

    def forward(self, input):
        hidden1 = torch.sigmoid(self.l1(input))
        hidden2 = F.softplus(self.l2(hidden1))
        output = self.l3(hidden2)
        return output


def RMSELoss(f, y):
    return torch.sqrt(torch.mean((f - y) ** 2))


def create_floder(path):
    try:
        os.makedirs(path)
    except:
        pass


def Modeling(version, EPOCH):
    # Create floder
    floder_path = "../output/FNN_" + str(version) + "/"

    create_floder(floder_path)

    # Load data
    StringPath = open("../data/IPC81.csv", "r")
    PropertyName = "logEC50"
    StringData = csv.DictReader(StringPath)

    # Load structures and properties
    items = []
    lists = []
    for row in StringData:
        lists.append(float(row[PropertyName]))
        item = Molecule_reader(row["no"], row["name"], row["smiles"],
                               float(row[PropertyName]), row["set"])
        items.append(item)

    mean = np.mean(lists)
    std = np.std(lists, ddof=0)

    for item in items:
        item.nor_prop = (item.property - mean) / std

    # Encode feature
    all_list, comb_list = [], []
    CompNum = len(items)
    for i in range(CompNum):
        m = Chem.MolFromSmiles(items[i].smiles)
        single_list = []
        for atom in m.GetAtoms():
            atom_iso = ""
            try:
                cipcode = atom.GetProp("_CIPCode")
                if cipcode == "R":
                    atom_iso = "@"
                elif cipcode == "S":
                    atom_iso = "@@"
            except:
                pass
            symbol = atom.GetSymbol()
            if atom.GetIsAromatic():
                symbol = symbol.lower()
            charge = atom.GetFormalCharge()
            H_num = atom.GetTotalNumHs()

            idx1 = atom.GetIdx()
            bond_num = np.zeros(4)
            bond_symbol = ["-", "=", "#", "*"]
            for x in atom.GetNeighbors():
                idx2 = x.GetIdx()
                bond = m.GetBondBetweenAtoms(idx1, idx2)
                Type = bond.GetBondType()
                Aro = bond.GetIsAromatic()
                if Aro:
                    bond_num[3] += 1
                else:
                    if str(Type) == "SINGLE":
                        bond_num[0] += 1
                    elif str(Type) == "DOUBLE":
                        bond_num[1] += 1
                    elif str(Type) == "TRIPLE":
                        bond_num[2] += 1
            bonds = "".join([bond_symbol[i] * int(bond_num[i]) for i in range(len(bond_num))])
            descriptor = "".join(
                ["[", symbol, "H", str(H_num), "|", atom_iso, "+" * charge, "-" * -charge, "]", bonds])
            all_list.append(descriptor)
            single_list.append(descriptor)
        comb_list.append(single_list)

    unique_list = Counter(all_list)
    des_dict = dict(sorted(unique_list.items(), key=lambda x: x))
    FragNum = len(des_dict)
    AtomVec = dict(zip(des_dict.keys(), range(FragNum)))
    AtomFrame = pd.DataFrame(list(des_dict.items()))
    AtomFrame.to_csv(floder_path + "structural_descriptor_list.csv")

    VecNum = []
    for j in range(len(comb_list)):
        zero = [0] * FragNum
        SubList = comb_list[j]
        for subeach in SubList:
            zero[AtomVec[subeach]] += 1
        setattr(items[j], "vector", zero)
        VecNum.append(zero)
    RegFrame = pd.DataFrame(VecNum)
    RegFrame.to_csv(floder_path + "descriptor_vector.csv")

    # Create training and test sets
    X_train = [item for item in items if item.set == "Training"]
    Y_train = [item.nor_prop for item in items if item.set == "Training"]

    # Create record file
    id = open(floder_path + "cross_validation_record.csv", "w", newline="")
    writer = csv.writer(id)
    writer.writerow(["n1", "n2", "Scores", "Scores_ave", "RMSE-Train", "RMSE-Test"])
    id.close()

    # Train model with the five-fold cross validation & record
    n1 = range(1, 17)
    n2 = range(1, 17)

    ffcv = np.Inf
    for n11 in n1:
        for n22 in n2:
            print("n1 = %d, n2 = %d" % (n11, n22))
            kf = KFold(n_splits=5)
            scores = []
            kF = 0
            for train_index, test_index in kf.split(X_train):
                kF += 1
                x_train_cv, y_train_cv, x_test_cv, y_test_cv = [], [], [], []
                for ti in train_index:
                    x_train_cv.append(X_train[ti])
                    y_train_cv.append(Y_train[ti])
                for te in test_index:
                    x_test_cv.append(X_train[te])
                    y_test_cv.append(Y_train[te])

                x_train_vector = np.array([item.vector for item in x_train_cv])
                x_test_vector = np.array([item.vector for item in x_test_cv])

                x_train = Variable(torch.Tensor(x_train_vector))
                x_test = Variable(torch.Tensor(x_test_vector))
                y = Variable(torch.Tensor(y_train_cv).unsqueeze(1))
                y2 = Variable(torch.Tensor(y_test_cv).unsqueeze(1))

                early_stopping = EarlyStopping()
                model = FNN(FragNum, n11, n22)

                criterion = RMSELoss
                optimizer = torch.optim.Adam(model.parameters())

                for i in range(EPOCH):
                    optimizer.param_groups[0]["lr"] = 0.01
                    f_train = model(x_train)
                    loss = criterion(f_train, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (i + 1) % 1 == 0:
                        f_train = model(x_train)
                        f_train = np.array([item[0] * std + mean for item in f_train.detach().numpy().astype(float)])
                        y_train = np.array([item[0] * std + mean for item in y.detach().numpy().astype(float)])

                        f_test = model(x_test)
                        f_test = np.array([item[0] * std + mean for item in f_test.detach().numpy().astype(float)])
                        y_test = np.array([item[0] * std + mean for item in y2.detach().numpy().astype(float)])

                        RMSE_1 = np.sqrt(metrics.mean_squared_error(y_train, f_train))
                        RMSE_2 = np.sqrt(metrics.mean_squared_error(y_test, f_test))

                        early_stopping(RMSE_1)

                    if early_stopping.early_stop:
                        scores.append(early_stopping.best_score)
                        break

            id = open(floder_path + "cross_validation_record.csv", "a", newline="")
            writer = csv.writer(id)
            writer.writerow([n11, n22, scores, np.average(scores), RMSE_1, RMSE_2])
            id.close()

            if np.average(scores) <= ffcv:
                ffcv = np.average(scores)
                n1_opt, n2_opt = n11, n22

    print(f"Cross validation completed")
    print(f"The hidden layer 1 and 2 have {n1_opt} and {n2_opt} neurons respectively")


if __name__ == "__main__":
    version = "v1"
    EPOCH = 100000  # define a large enough epoch number
    Modeling(version, EPOCH)
