# Used to obtain the optimal SVM model by simultaneous cross-validation and model training

import os
import csv
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn import metrics


class Molecule_reader():
    def __init__(self, no, name, smiles, property, set):
        self.no = no
        self.name = name
        self.smiles = smiles
        self.property = property
        self.set = set


def create_floder(path):
    try:
        os.makedirs(path)
    except:
        pass


def modeling(version):
    # Create floder
    floder_path = "../output/SVM_" + str(version) + "/"

    create_floder(floder_path)

    # Load data
    StringPath = open("../data/IPC81.csv", "r")
    PropertyName = "logEC50"
    StringData = csv.DictReader(StringPath)

    # Load structures and properties
    items = []
    for row in StringData:
        item = Molecule_reader(row["no"], row["name"], row["smiles"], float(row[PropertyName]), row["set"])
        items.append(item)

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
            descriptor = "".join(["[", symbol, "H", str(H_num), "|", atom_iso, "+" * charge, "-" * -charge, "]", bonds])
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

    x_train = np.array([item.vector for item in X_train])
    y_train = np.array([item.property for item in X_train])
    x_test = np.array([item.vector for item in X_test])
    y_test = np.array([item.property for item in X_test])

    # Create record file
    id = open(floder_path + "cross_validation_record.csv", "w", newline="")
    writer = csv.writer(id)
    writer.writerow(
        ["C", "Epsilon", "Scores", "Scores_average", "RMSE-Train", "AAE-Train", "R2-Train", "RMSE-Test", "AAE-Test",
         "R2-Test"])
    id.close()

    # Train model with the five-fold cross validation & record
    c = np.arange(50, 0, -1)
    eps = np.arange(0.01, 0.51, 0.01)
    initial = np.Inf
    for ci in c:
        for epsi in eps:
            model = SVR(C=ci, epsilon=epsi)
            print("\nC value = %d,\tepsilon value = %.2f" % (ci, epsi))
            scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_root_mean_squared_error') * -1

            model = model.fit(x_train, y_train)

            for item in X_train:
                item.prediction = model.predict([item.vector])[0]
            for item in X_test:
                item.prediction = model.predict([item.vector])[0]

            f_train = model.predict(x_train)

            RMSE_1 = np.sqrt(metrics.mean_squared_error(y_train, f_train))
            AAE_1 = metrics.mean_absolute_error(y_train, f_train)
            R2_1 = metrics.r2_score(y_train, f_train)
            print('Train:\tRMSE = %.4f,\tAAE = %.4F,\tR2 = %.4f' % (RMSE_1, AAE_1, R2_1))

            f_test = model.predict(x_test)
            RMSE_2 = np.sqrt(metrics.mean_squared_error(y_test, f_test))
            AAE_2 = metrics.mean_absolute_error(y_test, f_test)
            R2_2 = metrics.r2_score(y_test, f_test)
            print('Test:\tRMSE = %.4f,\tAAE = %.4F,\tR2 = %.4f' % (RMSE_2, AAE_2, R2_2))

            id = open(floder_path + "cross_validation_record.csv", "a", newline="")
            writer = csv.writer(id)
            writer.writerow([ci, epsi, scores, np.average(scores), RMSE_1, AAE_1, R2_1, RMSE_2, AAE_2, R2_2])
            id.close()

            cv_score = np.average(scores)
            if np.average(scores) <= initial:
                initial = cv_score
                joblib.dump(model, floder_path + "SVM_model.pkl")

                ff = open(floder_path + "SVM_Results.csv", "w", newline="")
                writerf = csv.writer(ff)
                writerf.writerow(["no", "name", "smiles", "property", "prediction", "dataset"])
                [writerf.writerow([item.no, item.name, item.smiles, item.property, item.prediction, "Training"])
                 for item in X_train]
                [writerf.writerow([item.no, item.name, item.smiles, item.property, item.prediction, "Test"])
                 for item in X_test]
                ff.close()

    print("Model training completed")


if __name__ == "__main__":
    version = "v1"
    modeling(version)
