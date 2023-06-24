import os
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchsummary import summary
import urllib.request
import zipfile



data_file = "data" #google colab:/content/WBC-Differential-Learning-Tool/data

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

train_percentage = 0.6
val_percentage = 0.15
test_percentage = 0.25

batch_size = 8
num_workers = 2

from function import create_datasets, create_dataloaders, train_model, test_model
#Prepare dataset
train_dataset, val_dataset, test_dataset, class_names, num_classes = create_datasets(data_file, train_percentage, val_percentage)
dataloaders, dataset_sizes = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers)

# Instantiate the Resnet34 model
net = torchvision.models.resnet34(weights=True)
# Cross entropy loss combines softmax and nn.NLLLoss() in one single class.
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.SGD(net.parameters(), lr=0.1)
n_epochs= 50
n_batches = 20
scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                        max_lr=0.5,
                                        base_momentum = 0.6,
                                        steps_per_epoch=n_batches,
                                        epochs=n_epochs)
lr_steps = []
mom_steps = []
for epoch in range(n_epochs):
    for batch in range(n_batches):
        optimizer.step()
        lr_steps.append(optimizer.param_groups[0]['lr'])
        mom_steps.append(optimizer.param_groups[0]['momentum'])
        scheduler.step()

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cost_path = train_model(net, criterion, optimizer, dataloaders, device, num_epochs=50)

# Calculate the test set accuracy and recall for each class
acc,recall_vals = test_model(net,test_loader,device)
print('Test set accuracy is {:.3f}'.format(acc))
for i in range(7):
    print('For class {}, recall is {}'.format(class_names[i],recall_vals[i]))

