'''
utils.py contains the necesary functions for common model training processes 
'''

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
import pandas as pd
import sklearn as sk
import os
import time 
from datetime import datetime
import pytz

def predict_video(model, video_frames):
    prediction = model(video_frames)
    return prediction

def validation_a(model, data_loader, criterion, device):
    test_loss = 0

    for f,a,v,v_name in data_loader:
        f,a,v = f.to(device), a.to(device), v.to(device)

        output = model.forward(f)
        test_loss += criterion(output, a).item()

    return test_loss

def validation_v(model, data_loader, criterion, device):
    test_loss = 0

    for f,a,v,v_name in data_loader:
        f,a,v = f.to(device), a.to(device), v.to(device)

        output = model.forward(f)
        test_loss += criterion(output, v).item()

    return test_loss

def plot_curves(model_name,epoch,train_loss,val_loss):
    e = [i for i in range(1, epoch+1)]       
    plt.plot(e,train_loss, label='Training Loss')
    plt.plot(e,val_loss, label='Validation Loss')
    plt.xticks(np.arange(min(e), max(e)+1, 1.0))
    plt.legend()
    plt.title(f'{model_name} loss',color='black')
    plt.xlabel('epoch',color='black')
    plt.ylim(ymin=0)
    plt.ylabel('loss',color='black')
    plt.tick_params(colors='black')
    loss_graph = f'./validation_results/{model_name}'
    plt.savefig(loss_graph,dpi=100,bbox_inches = 'tight')
    plt.show()
    
def load_model(input_model, path):
    cp = torch.load(path)
    model = input_model
    model.load_state_dict(cp)
    return model
    
def save_model(model, path):
    """
    save function to save model for later use
    """
    torch.save(model.state_dict(), path)