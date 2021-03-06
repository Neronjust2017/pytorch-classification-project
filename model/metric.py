import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Normal\MC_dropout\BBB(BBB_LR)\VD\...
def get_pred(output, type=None):
    with torch.no_grad():
        if type in ['MC_dropout','BBB', 'VD']:
            output_softmax = F.softmax(output, dim=1)
            histo = torch.zeros((output_softmax.shape[0], output_softmax.shape[1]))

            for i in range(output_softmax.shape[0]):
                for j in range(output_softmax.shape[1]):
                    # histo[i, j] = np.percentile(output_softmax[i, j, :].cpu(), output_softmax.shape[2]//2)
                    histo[i, j] = np.percentile(output_softmax[i, j, :].cpu(), 50)

            pred = torch.argmax(histo, dim=1).to('cuda')

        elif type in ['Normal']:
            pred = torch.argmax(output, dim=1)
        else:
            print("type error!")
            exit(1)
        return  pred

def accuracy(output, target, type=None):
    pred = get_pred(output, type)
    assert pred.shape[0] == len(target)
    correct = 0
    correct += torch.sum(pred == target).item()

    return correct / len(target)

# Normal\MC_dropout\BBB(BBB_LR)\VD\...
def precision(output, target, type=None):
    pred = get_pred(output, type)
    assert pred.shape[0] == len(target)
    precision = precision_score(target.cpu(), pred.cpu(), average='macro')
    return precision

# Normal\MC_dropout\BBB(BBB_LR)\VD\...
def recall(output, target, type=None):
    pred = get_pred(output, type)
    assert pred.shape[0] == len(target)
    recall = recall_score(target.cpu(), pred.cpu(), average='macro')
    return recall

# Normal\MC_dropout\BBB(BBB_LR)\VD\...
def f1(output, target, type=None):
    pred = get_pred(output, type)
    assert  pred.shape[0] == len(target)
    f1 = f1_score(target.cpu(), pred.cpu(), average='macro')
    return f1

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

