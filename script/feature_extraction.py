from rdkit import Chem
import numpy as np


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
