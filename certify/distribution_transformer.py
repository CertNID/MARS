import math

import torch
import torch.nn as nn
import torch.distributions as dist

class Gaussian_CDF(nn.Module):
    def __init__(self):
        super(Gaussian_CDF, self).__init__()

        self.w = 1.6638520956039429

    def forward(self, x: torch.tensor) -> torch.tensor:
        return 1 / (1 + torch.exp(- self.w * x))
    
class Laplace_Inv(nn.Module):
    def __init__(self):
        super(Laplace_Inv, self).__init__()

        self.b = 1

    def forward(self, p: torch.Tensor) -> torch.tensor:
        
        inv = 0 - self.b * torch.sign(p - 0.5) * torch.log(1 - 2 * torch.abs(p - 0.5))

        return inv

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1 / (b - a)
    else:
        return 0.0

def uniform_cdf(x, a, b):
    if x < a:
        return 0.0
    elif a <= x <= b:
        return (x - a) / (b - a)
    else:
        return 1.0

def generate_clipped_gaussian_noise(mean, std_dev, epsilon, shape):
   
    gaussian_dist = dist.normal.Normal(loc=mean, scale=std_dev)
    
    # Sample noise from the distribution
    noise = gaussian_dist.sample(shape)
    
    # Clip the noise to the range [-epsilon, epsilon]
    clipped_noise = torch.clamp(noise, min=-epsilon, max=epsilon)
    
    return clipped_noise



class Distribution_Transformer_Laplace(nn.Module):
    def __init__(self, d: int):
        """
        d: the number of feature vector dimensions
        b: the scale parameter for the Laplace distribution
        """
        super(Distribution_Transformer_Laplace, self).__init__()

        self.d = d
        self.cdf_ppf = nn.ModuleList([
            nn.Sequential(Gaussian_CDF(), Laplace_Inv()),
        ])
        self.weight_shared = nn.Parameter(torch.zeros((1), requires_grad=True))
        self.weight_independent = nn.Parameter(torch.zeros((1, self.d), requires_grad=True))

        self.weight_init()

    def weight_init(self):
        self.weight_shared.data.normal_(0, 0.01)
        self.weight_independent.data.normal_(0, 0.01)

    def forward(self, z: torch.tensor) -> torch.tensor:
        assert self.d == z.size(1)
        x = torch.zeros_like(z, device=z.device)
        for i in range(len(self.cdf_ppf)):
            x += self.cdf_ppf[i](z) * self.weight_shared[i].abs()
            x += self.cdf_ppf[i](z) * self.weight_independent[i, :].abs()

        return x


    def get_weight(self) -> torch.tensor:
        return torch.cat([self.weight_shared.abs().view(-1), self.weight_independent.abs().view(-1)], 0)

class Distribution_Transformer_Gaussian(nn.Module):
    def __init__(self, d: int):
        """
        d: the number of feature vector dimensions 
        """
        super(Distribution_Transformer_Gaussian, self).__init__()

        self.d = d
        self.weight_shared = nn.Parameter(torch.zeros((1), requires_grad=True))
        self.weight_independent = nn.Parameter(torch.zeros((1, self.d), requires_grad=True))
        self.weight_init()

    def weight_init(self):
        self.weight_shared.data.normal_(0, 0.01)
        self.weight_independent.data.normal_(0, 0.01)
    def weight_replace(self, sub):
        self.weight_independent.data = sub
    def forward(self, z: torch.tensor,ok) -> torch.tensor:
        assert self.d == z.size(1)
        x = torch.zeros_like(z, device=z.device)
        x += z * self.weight_shared.abs().to(z.device)
        x += z * self.weight_independent.abs().to(z.device)
        if ok == True:
            x = torch.zeros_like(z, device=z.device)
            x += z * self.weight_shared.abs().to(z.device)
            x *= self.weight_independent.abs().to(z.device)
        return x

    def get_weight(self) -> torch.tensor:
        return torch.cat([self.weight_shared.abs().view(-1), self.weight_independent.abs().view(-1)], 0)



# Different feature noise distributions 
distribution_transformers = {
    "gaussian": Distribution_Transformer_Gaussian,
}


class Distribution_Transformer_Loss_CADE(nn.Module):
    def __init__(self, lambda_: float, mad_threshold: float):
        super(Distribution_Transformer_Loss_CADE, self).__init__()
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

class Distribution_Transformer_Loss_ACID(nn.Module):
    def __init__(self, lambda_: float, certify_class: int):
        super(Distribution_Transformer_Loss_ACID, self).__init__()
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
    "cade": Distribution_Transformer_Loss_CADE,
    "acid": Distribution_Transformer_Loss_ACID
}
