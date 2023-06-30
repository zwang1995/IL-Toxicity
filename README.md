# Machine Learning for Ionic Liquid Toxicity Prediction

This repository contains the data and scripts involved in the publication:

**[Machine Learning for Ionic Liquid Toxicity Prediction](https://doi.org/10.3390/pr9010065)**

## Requirements 
### Library
* [RDKit](https://www.rdkit.org/): cheminformatics
* [PyTorch](https://pytorch.org/) & [scikit-learn](https://scikit-learn.org/stable/): development of ML models

## Scripts 
- `FNN-kFCV.py`: determine the hyper-parameters of the FNN model using the five-fold cross-validation
- `FNN-kFCV-Further.py`: train FNN model based on the hyper-parameters determined by `FNN-kFCV.py`
- `EarlyStopping.py`: determine the time to stop the training of FNN models
- `SVM-kFCV.py`: determine the hyper-parameters and the SVM model using the five-fold cross-validation

## Authors
* Zihao Wang: zwang@mpi-magdeburg.mpg.de
* [Prof. Dr. Zhen Song](https://hgxy.ecust.edu.cn/2021/0906/c1270a132681/page.htm)
* [Prof. Dr.-Ing. Teng Zhou](https://facultyprofiles.hkust-gz.edu.cn/faculty-personal-page/ZHOU-Teng/tengzhou)
