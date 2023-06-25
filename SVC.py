import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from PIL import Image
import torchvision
from torchvision import datasets, transforms

data_file = "data" #google colab:/content/WBC-Differential-Learning-Tool/data

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

train_percentage = 0.6
val_percentage = 0.15
test_percentage = 0.25

batch_size = 8
num_workers = 2

input_size = 150528
learning_rate = 0.01
momentum = 1
num_epochs=20

from function import create_datasets, create_dataloaders, train_SVC_model, test_SVC_model
train_dataset, val_dataset, test_dataset, class_names, num_classes = create_datasets(data_file, train_percentage, val_percentage)
dataloaders, dataset_sizes = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers)

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

class SVM_Loss(torch.nn.modules.Module):
    """
    SVM Loss function
    """    
    def __init__(self):
        """
        Initialize the SVM Loss function
        """
        super(SVM_Loss,self).__init__()

    def forward(self, outputs, labels, batch_size):
        """
        Forward pass of the SVM Loss function
        """
        return torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))/batch_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
svm_model = nn.Linear(input_size,num_classes)

  ## Loss and optimizer
svm_loss_criteria = SVM_Loss()
svm_optimizer = torch.optim.SGD(svm_model.parameters(), lr=learning_rate, momentum=momentum)
total_step = len(dataloaders["train"])

## Train model
model = train_SVC_model(svm_model, input_size, svm_loss_criteria, svm_optimizer, dataloaders, batch_size, device, num_epochs)

model_dir = 'models'
filename = 'SVM_SGD.pt'

# Save the entire model
torch.save(model, os.path.join(model_dir,filename))

#Test the model
model = torch.load("/content/WBC-Differential-Learning-Tool/models/SVM_SGD.pt")
test_SVC_model(model, dataloaders["test"], device, input_size)

