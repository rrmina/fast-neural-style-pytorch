import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive):
        distance = nn.PairwiseDistance()(anchor, positive)
        loss = torch.mean(torch.clamp(self.margin - distance, min=0.0) ** 2)
        return loss
