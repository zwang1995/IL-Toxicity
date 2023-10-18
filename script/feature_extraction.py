from collections import defaultdict
from rdkit import Chem
import numpy as np
import csv
import json


class Molecule_reader():
    def __init__(self, no, name, smiles, property):
        self.no = no
        self.name = name
        self.smiles = smiles
        self.property = property


def dict_save(path, dict):
    if isinstance(dict, str):
        dict = eval(dict)
    with open(path, 'w', encoding='utf-8') as f:
        str_ = json.dumps(dict, ensure_ascii=False)
        # print(type(str_), str_)
        f.write(str_)
    print("-> Feature dictionary has been saved.")


def iso_identify(atom):
    try:
        RS = atom.GetProp("_CIPCode")
        if RS == "R":
            return ("@")
        elif RS == "S":
            return ("@@")
    except:
        return ("")


def bond_identify(atom, mol):
    idx1 = atom.GetIdx()
    bond_num = np.zeros(4)  # 0- 1= 2# 3* 4` 5`` 6```
    bond_symbol = ["-", "=", "#", "*"]

    for x in atom.GetNeighbors():
        idx2 = x.GetIdx()
        bond = mol.GetBondBetweenAtoms(idx1, idx2)
        type = bond.GetBondType()
        ring = bond.IsInRing()
        aro = bond.GetIsAromatic()
        if aro:
            bond_num[3] += 1
        else:
            if str(type) == "SINGLE":
                bond_num[0] += 1
            elif str(type) == "DOUBLE":
                bond_num[1] += 1
            elif str(type) == "TRIPLE":
                bond_num[2] += 1
    return "".join([bond_symbol[i] * int(bond_num[i]) for i in range(len(bond_num))])


def Encoding(smiles):
    m = Chem.MolFromSmiles(smiles)

    atoms = [a.GetSymbol() for a in m.GetAtoms()]
    for (i, atom) in enumerate(m.GetAtoms()):
        symbol = atom.GetSymbol()
        if atom.GetIsAromatic():
            symbol = symbol.lower()
        H_num = atom.GetTotalNumHs()
        charge = atom.GetFormalCharge()
        atom_iso = iso_identify(atom)
        bonds = bond_identify(atom, m)

        feature = "".join(["[", symbol, "H", str(H_num), "|", atom_iso, "+" * charge, "-" * -charge, "]", bonds])
        atoms[i] = feature
    return atoms


if __name__ == "__main__":
    csv_file = open("Props.csv", "r")
    prop_name = "prop"
    data = csv.DictReader(csv_file)
    values, items = [], []
    for row in data:
        values.append(float(row[prop_name]))
        item = Molecule_reader(row["Index"], row["name"], row["smiles"],
                               float(row[prop_name]))
        items.append(item)

    mean = np.mean(values)
    std = np.std(values, ddof=0)

    for item in items:
        item.nor_prop = (item.property - mean) / std

    feature_dict = defaultdict(lambda: len(feature_dict))
    for item in items:
        one_hot = Encoding(item.smiles, feature_dict)
        frequency = [one_hot.count(i) for i in range(len(feature_dict))]
        item.vector = frequency

    dict_save('Features.txt', feature_dict)
    print("Number of features: ", len(feature_dict))
