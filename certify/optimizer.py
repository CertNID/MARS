import os
import numpy as np
from typing import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class Sigma_Adaptation(nn.Module):
    def __init__(self, d: int):
        """
        d: the number of feature vector dimensions 
        """
        super(Sigma_Adaptation, self).__init__()

        self.d = d
        self.weight_shared = nn.Parameter(torch.zeros((1), requires_grad=True))

        self.weight_init()

    def weight_init(self):
        self.weight_shared.data.normal_(0, 0.01)

    def forward(self, z: torch.tensor) -> torch.tensor:
        assert self.d == z.size(1)
        x = torch.zeros_like(z, device=z.device)
        x += z * self.weight_shared.abs().to(device=z.device)

        return x

    def get_weight(self) -> torch.tensor:
        return self.weight_shared.abs().view(-1)
    
class Loss_CADE(nn.Module):
    def __init__(self, lambda_: float, mad_threshold: float):
        super(Loss_CADE, self).__init__()
        """
        lambda_: the weight of the regularizer
        mad_threshold: MAD threshold for detecting drift
        """
        self.lambda_ = lambda_
        self.mad_threshold = mad_threshold

        self.mu = 1 / 3

    def forward(self, score: torch.tensor, weight: torch.tensor):
        idx_wrong = torch.where(score > self.mad_threshold)[0]
        loss_wrong = (1 / score.size(0)) * torch.sum(score[idx_wrong] - self.mad_threshold) if idx_wrong.size(0) > 0 else torch.tensor(0., requires_grad=True)
        idx_correct = torch.where(score <= self.mad_threshold)[0]
        loss_correct = (1 / score.size(0)) * torch.sum(self.mad_threshold - score[idx_correct]) if idx_correct.size(0) > 0 else torch.tensor(0., requires_grad=True)

        regularizer = torch.log(1 + torch.exp(-weight.abs())).sum()

        return self.mu * (loss_wrong + loss_correct) + self.lambda_ * regularizer

class Loss_ACID(nn.Module):
    def __init__(self, lambda_: float, certify_class: int):
        super(Loss_ACID, self).__init__()
        """
        lambda_: the weight of the regularizer
        certify_class: the certified class as training target
        """
        self.criterion = nn.MultiLabelSoftMarginLoss(reduction='sum')
        self.lambda_ = lambda_
        self.certify_class = certify_class

        self.mu = 1 / 3

    def forward(self, score: torch.tensor, weight: torch.tensor):
        idx_wrong = torch.where(score.argmax(1) != self.certify_class)[0]
        idx_correct = torch.where(score.argmax(1) == self.certify_class)[0]
        target = torch.ones_like(score, device=score.device)
        target[idx_wrong, :] = 0
        target[idx_wrong, self.certify_class] = 1
        target[idx_correct, :] = 1
        target[idx_correct, self.certify_class] = 0
        loss_wrong = (1 / score.size(0)) * self.criterion(score[idx_wrong, :], target[idx_wrong, :]) if idx_wrong.size(0) > 0 else torch.tensor(0., requires_grad=True)
        loss_correct = (1 / score.size(0)) * self.criterion(score[idx_correct, :], target[idx_correct, :]) if idx_correct.size(0) > 0 else torch.tensor(0., requires_grad=True)

        regularizer = torch.log(1 + torch.exp(-weight.abs())).sum()

        return self.mu * (loss_wrong + loss_correct) + self.lambda_ * regularizer
loss_functions = {
    "cade": Loss_CADE,
    "acid": Loss_ACID
}
