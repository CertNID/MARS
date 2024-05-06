import torch
from sklearn.metrics import f1_score,precision_score,recall_score, confusion_matrix
import numpy as np
class Tracker(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk):

    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res
def calc_metrics_classifieracid(label: np.ndarray, pred: np.ndarray):
    assert label.shape == pred.shape
    acc = np.array((label == pred), dtype=float).mean()
    m = confusion_matrix(y_true=label, y_pred=pred)

    return acc, m
def calc_metrics_classifier_class(label: np.ndarray, pred: np.ndarray):
    assert label.shape == pred.shape
    acc = np.array((label == pred), dtype=float).mean()
    m = confusion_matrix(y_true=label, y_pred=pred)

    return acc, m

def calc_metrics_classifiercade(label: np.ndarray, pred: np.ndarray) -> (float, float, float, float):
    assert label.shape == pred.shape
    acc = np.array((label == pred), dtype=float).mean()
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    f1 = f1_score(label, pred)

    return acc, p, r, f1

def calc_metrics_certify(label: np.ndarray, pred: np.ndarray, radius_feat: np.ndarray) -> np.ndarray:
    assert label.shape == pred.shape
    assert pred.shape == radius_feat.shape
    
    return np.mean(np.array((label == pred), dtype=float) * radius_feat)


def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()