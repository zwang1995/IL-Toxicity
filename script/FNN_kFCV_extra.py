# FNN training using the optimal hyper-parameters obtained from FNN_kFCV.py

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


def Modeling(version, n1, n2, EPOCH):
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
    X_test = [item for item in items if item.set == "Test"]
    Y_train = [item.nor_prop for item in items if item.set == "Training"]
    Y_test = [item.nor_prop for item in items if item.set == "Test"]

    x_train_vector = np.array([item.vector for item in X_train])
    x_test_vector = np.array([item.vector for item in X_test])

    x_train = Variable(torch.Tensor(x_train_vector))
    x_test = Variable(torch.Tensor(x_test_vector))
    y = Variable(torch.Tensor(Y_train).unsqueeze(1))
    y2 = Variable(torch.Tensor(Y_test).unsqueeze(1))

    # Create record file
    id = open(floder_path + "training_record.csv", "w", newline="")
    writer = csv.writer(id)
    writer.writerow(["n1", "n2", "Epoch", "RMSE-Train", "AAE-Train", "R2-Train", "RMSE-Test", "AAE-Test", "R2-Test"])
    id.close()

    # Train model with the five-fold cross validation & record
    early_stopping = EarlyStopping()
    model = FNN(FragNum, n1, n2)

    criterion = RMSELoss
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.param_groups[0]["lr"] = 0.01

    for i in range(EPOCH):
        optimizer.zero_grad()
        f_train = model(x_train)
        loss = criterion(f_train, y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 1 == 0:
            f_train = model(x_train)
            f_train = np.array([item[0] * std + mean for item in f_train.detach().numpy().astype(float)])
            y_train = np.array([item[0] * std + mean for item in y.detach().numpy().astype(float)])

            f_test = model(x_test)
            f_test = np.array([item[0] * std + mean for item in f_test.detach().numpy().astype(float)])
            y_test = np.array([item[0] * std + mean for item in y2.detach().numpy().astype(float)])

            for item in X_train:
                item.prediction = model(Variable(torch.Tensor(item.vector))).detach().numpy().astype(float)[
                                      0] * std + mean
            for item in X_test:
                item.prediction = model(Variable(torch.Tensor(item.vector))).detach().numpy().astype(float)[
                                      0] * std + mean

            RMSE_1 = np.sqrt(metrics.mean_squared_error(y_train, f_train))
            AAE_1 = metrics.mean_absolute_error(y_train, f_train)
            R2_1 = metrics.r2_score(y_train, f_train)

            RMSE_2 = np.sqrt(metrics.mean_squared_error(y_test, f_test))
            AAE_2 = metrics.mean_absolute_error(y_test, f_test)
            R2_2 = metrics.r2_score(y_test, f_test)

            early_stopping(RMSE_1)

            id = open(floder_path + "training_record.csv", "a", newline="")
            writer = csv.writer(id)
            writer.writerow([n1, n2, i + 1, RMSE_1, AAE_1, R2_1, RMSE_2, AAE_2, R2_2])
            id.close()

            if early_stopping.update:
                torch.save(model, floder_path + "FNN_model.pkl")

                ff = open(floder_path + "FNN_results.csv", "w", newline="")
                writerf = csv.writer(ff)
                writerf.writerow(["no", "name", "smiles", "property", "prediction", "dataset"])
                [writerf.writerow([item.no, item.name, item.smiles, item.property, item.prediction, "Training"]) for
                 item in X_train]
                [writerf.writerow([item.no, item.name, item.smiles, item.property, item.prediction, "Test"]) for item in
                 X_test]
                ff.close()

        if early_stopping.early_stop:
            break

    model.mean, model.std = mean, std
    print("Model training completed")


if __name__ == "__main__":
    version = "v1"
    # n1 and n2 are determined by FNN.kFCV.py file
    n1 = 10  # number of neurons in hidden layer 1
    n2 = 6  # number of neurons in hidden layer 2
    EPOCH = 100000  # define a large enough epoch number
    Modeling(version, n1, n2, EPOCH)
