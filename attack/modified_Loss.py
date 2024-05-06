import torch.nn as nn
import torch

class Loss_AdaptiveClusteringv2(nn.Module):
    def __init__(self, num_classes: int):
        super(Loss_AdaptiveClusteringv2, self).__init__()

        self.num_classes = num_classes
        self.margin = 1

        self.criterion = nn.MSELoss()

    def forward(self,o: torch.tensor, y: torch.tensor):
        y=y.to(o.device)
        loss_classification = self.criterion(o.float(), y.float())
        
       

        return loss_classification 