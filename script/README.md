**Python files:**
- `FNN_kFCV.py`: determine the hyper-parameters of the FNN model using the five-fold cross-validation
- `FNN_kFCV_extra.py`: train FNN model using the optimal hyper-parameters identified from `FNN_kFCV.py`
- `SVM_kFCV.py`: determine the hyper-parameters and train the SVM model using the five-fold cross-validation
- `early_stopping.py`: a tool used to control the termination of FNN training
- `feature_extraction.py`: functions used in `prediction.py` to extract features based on the SMILES string
- `prediction.py`: make predictions using the developed FNN and SVM models

**Training:**
1. To train FNN models, run `FNN_kFCV.py` first and then run `FNN_kFCV_extra.py` with the optimal FNN structure obtained from `FNN_kFCV.py`
2. To train SVM models, run `SVM_kFCV.py`

**Prediction:**
1. Store SMILES strings in `../data/SMILES_list_for_pred.csv`
2. Run `prediction.py` and the prediction results can be found in `../output/pred_for_SMILES_list.csv`
