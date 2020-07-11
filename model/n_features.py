from model import CNN
import torch
model = CNN(input_dim=9, num_classes=1, filters=[256], kernels=[3], activation='ReLU')
x = torch.randn(100, 9, 128)
model(x)