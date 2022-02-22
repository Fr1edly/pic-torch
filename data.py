from asyncore import read
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms

import csv, shutil

num_epochs = 20
lr =.001

epochs = 2
batch_size=5
lr=0.003
train_data_path = 'assets/images/train'
test_data_path = 'assets/images/test'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
test_data= torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

classes = [str(i) for i in range(0,67)]

"""
csvfile = open('assets/images/train.csv')
reader = csv.reader(csvfile)
for row in reader:
    print(row[0])
    if row[0] == "filename":
        pass
    else:
        shutil.copy('assets/images/train/train/'+row[0], 'assets/images/train/'+row[1]+'/'+row[0])
    
"""