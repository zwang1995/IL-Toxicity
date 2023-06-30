import numpy as np


# This EarlyStopping.py file is used to determine the time for terminating the training of FNN models.
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.update = None

    def __call__(self, val_loss, model=None, path=None, data=None, path2=None):
        score = val_loss
        if self.best_score is None:
            self.update = True
            self.best_score = score
        elif score > self.best_score - self.delta:
            self.update = False
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.update = True
            self.best_score = score
            self.counter = 0

