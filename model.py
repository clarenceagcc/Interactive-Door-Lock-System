import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Define the Siamese Network Model using BCE Loss
class SiameseNetworkBCE(nn.Module):
    def __init__(self):
        super(SiameseNetworkBCE, self).__init__()
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1280, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.l2_norm = F.normalize
        self.fc2 = nn.Linear(256, 1)

    def forward_once(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.l2_norm(x, dim=-1)
        return x

    def forward(self, input1, input2):
        embedding1 = self.forward_once(input1)
        embedding2 = self.forward_once(input2)
        diff = torch.abs(embedding1 - embedding2)
        out = torch.sigmoid(self.fc2(diff))
        return out