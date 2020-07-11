import numpy as np
import torch
import torch.nn.functional as F

# Classification
def nll_loss(output, target):
    return F.nll_loss(output, target)
# size_average (bool, optional) – 默认情况下，是mini-batchloss的平均值

def cross_entropy_loss(output, target, reduction='sum'):
    return F.cross_entropy(output, target, reduction=reduction)
#  size_average (bool, optional) – 默认情况下，是mini-batchloss的平均值，