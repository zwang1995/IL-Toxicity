`structural_descriptor_list.csv` presents the list of extracted structural descriptors

- column `0` is the text-form representation of the descriptor
- column `1` is the frequency of the descriptor appeared in all IL structures


`descriptor_vector.csv` presents the descriptor vectors of all ILs

- the vertical index indicates the index of IL in the dataset (start from 0)
- the horizontal index indicates the index of the structural descriptor, corresponding to the index in `structural_descriptor_list.csv`


`cross_validation_record.csv` and `results.csv` present the cross-validation results and predicted values

`model.pkl` is the developed FNN/SVM model


`pred_for_SMILES_list.csv` stores the SMILES strings of ILs to be predicted
