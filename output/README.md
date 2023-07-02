`structural_descriptor_list.csv` presents the list of structural descriptors

- the vertical index indicates the assigned number of the descriptor in the list
- the horizontal index `0` is the text-form presentation of the descriptor
- the horizontal index `1` is the frequency of the descriptor that appeared in all IL structures

`descriptor_vector.csv` presents the descriptor vectors of all ILs

- the vertical index indicates the index of IL in the entire dataset (start from 0)
- the horizontal index indicates the index of the structural descriptor, corresponding to the index in `structural_descriptor_list.csv`


`cross_validation_record.csv` presents the results of cross-validation

`model.pkl` and `results.csv` provide the trained FNN/SVM models and predicted values
