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

class HumanActivityRecognitionDataLoader2(BaseDataLoader):
    """
        HumanActivityRecognition data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, test_split=0.0, num_workers=1,
                 training=True):

        x_train, y_train, x_test, y_test = readmts_uci_har(data_dir)

        x_train = x_train.swapaxes(1, 2)
        x_test = x_test.swapaxes(1, 2)

        y_train, y_test = transform_labels(y_train, y_test)

        for i in range(len(x_train)):
            for j in range(len(x_test)):
                c = (x_train[i] == x_test[j])
                d = c.all()
                if d:
                    break

        X = np.concatenate((x_train, x_test))
        Y = np.concatenate((y_train, y_test))

        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y)

        dataset = TensorDataset(X, Y)
        super().__init__(dataset, batch_size, shuffle, validation_split, test_split, num_workers, normalization=True)


class HumanActivityRecognitionDataLoader(BaseDataLoader):
    """
        HumanActivityRecognition data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, test_split=0.0, num_workers=1,
                 training=True):

        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        # Output classes to learn how to classify
        LABELS = [
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING"
        ]

        DATASET_PATH = data_dir

        TRAIN = "train/"
        TEST = "test/"

        # Load "X" (the neural network's training and testing inputs)

        def load_X(X_signals_paths):
            """
            Given attribute (train or test) of feature, read all 9 features into an
            np ndarray of shape [sample_sequence_idx, time_step, feature_num]
                argument:   X_signals_paths str attribute of feature: 'train' or 'test'
                return:     np ndarray, tensor of features
            """
            X_signals = []

            for signal_type_path in X_signals_paths:
                file = open(signal_type_path, 'rb')
                # Read dataset from disk, dealing with text files' syntax
                X_signals.append(
                    [np.array(serie, dtype=np.float32) for serie in [
                        row.replace(b'  ', b' ').strip().split(b' ') for row in file
                    ]]
                )
                file.close()

            return np.transpose(np.array(X_signals), (1, 2, 0))

        X_train_signals_paths = [
            DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
        ]
        X_test_signals_paths = [
            DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
        ]

        x_train = load_X(X_train_signals_paths)
        x_test = load_X(X_test_signals_paths)

        x_train = x_train.swapaxes(1, 2)
        x_test = x_test.swapaxes(1, 2)

        # Load "y" (the neural network's training and testing outputs)

        def load_y(y_path):
            """
            Read Y file of values to be predicted
                argument: y_path str attibute of Y: 'train' or 'test'
                return: Y ndarray / tensor of the 6 one_hot labels of each sample
            """
            file = open(y_path, 'rb')
            # Read dataset from disk, dealing with text file's syntax
            y_ = np.array(
                [elem for elem in [
                    row.replace(b'  ', b' ').strip().split(b' ') for row in file
                ]],
                dtype=np.int32
            )
            file.close()

            # Substract 1 to each output class for friendly 0-based indexing
            # return one_hot(y_ - 1)
            return y_ - 1

        y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
        y_test_path = DATASET_PATH + TEST + "y_test.txt"

        y_train = load_y(y_train_path)
        y_test = load_y(y_test_path)

        X = np.concatenate((x_train, x_test))
        Y = np.concatenate((y_train, y_test))

        Y = Y.reshape((len(Y), ))

        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long()

        dataset = TensorDataset(X, Y)
        super().__init__(dataset, batch_size, shuffle, validation_split, test_split, num_workers, normalization=True)

