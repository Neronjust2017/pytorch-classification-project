import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


output = torch.rand((100, 6, 100))
histo = torch.zeros((output.shape[0], output.shape[1]))

for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        histo[i, j] = np.percentile(output[i, j, :], output.shape[2]//2)
        print(2)
print(2)