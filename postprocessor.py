from typing import Any
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score

class Postprocessor:
    def __init__(self, true_labels, pred_labels) -> None:
        self.true = true_labels
        self.true_binary = None
        self.pred = pred_labels
        self.true_binary = None

    def binarize(self) -> None:
        self.true_binary = [0 if label < 95 else 1 for label in self.true]
        self.pred_binary = [0 if label < 95 else 1 for label in self.pred]

    def compute_cls_metrics(self) -> list:
        mcc = matthews_corrcoef(self.true_binary, self.pred_binary)
        bacc = balanced_accuracy_score(self.true_binary, self.pred_binary)

        return [mcc, bacc]

    def __call__(self) -> list:
        self.binarize()
        return self.compute_cls_metrics()
