import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from base import BaseDataLoader, BaseDataLoader_2
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from .utils import readmts_uci_har, transform_labels

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class HumanActivityRecognitionDataLoader(BaseDataLoader):
    """
        HumanActivityRecognition data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, test_split=0.0, num_workers=1,
                 training=True):

        x_train, y_train, x_test, y_test = readmts_uci_har(data_dir)

        x_train = x_train.swapaxes(1, 2)
        x_test = x_test.swapaxes(1, 2)

        y_train, y_test = transform_labels(y_train, y_test)

        X = np.concatenate((x_train, x_test))
        Y = np.concatenate((y_train, y_test))

        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y)

        dataset = TensorDataset(X, Y)
        super().__init__(dataset, batch_size, shuffle, validation_split, test_split, num_workers, normalization=True)

