- `FNN-kFCV.py`: determine the hyper-parameters of the FNN model using the five-fold cross-validation
- `FNN-kFCV-Further.py`: train FNN model based on the hyper-parameters determined by `FNN-kFCV.py`
- `EarlyStopping.py`: determine the time to terminate the training of FNN models
- `SVM-kFCV.py`: determine the hyper-parameters and the SVM model using the five-fold cross-validation

Note: to train FNN models, run `FNN-kFCV.py` first and then run `FNN-kFCV-Further.py` with the optimal FNN structures; to train SVM models, run `SVM-kFCV.py` directly
