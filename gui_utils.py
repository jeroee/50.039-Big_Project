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
import cv2
import math
from mtcnn.src import detect_faces, show_bboxes
from PIL import Image

class CnnGru(nn.Module):
    def __init__(self,features,num_layers,hidden_size,batch_size):
        super(CnnGru,self).__init__()
        self.features = features
        self.num_layers = num_layers # the sequence 
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = 'cuda'
        
        # CNN
        self.conv2d_1 = nn.Conv2d(3, 16, 3, 1, 1) # 224 x224
        self.conv2d_2 = nn.Conv2d(16, 32, 3, 1, 1) # 112 x112
        self.conv2d_3 = nn.Conv2d(32, 64, 3, 1, 1) # 

        self.maxPool2d_1 = nn.MaxPool2d((2, 2))
        self.maxPool2d_2 = nn.MaxPool2d((2, 2))
        self.maxPool2d_3 = nn.MaxPool2d((2, 2))

        self.flatten_1 = nn.Flatten()
        self.fc1 = nn.Linear(50176,features)

        # RNN
        self.grucell = nn.GRUCell(input_size=features, hidden_size=hidden_size) #, num_layers = num_layers, batch_first = True)
        self.fc2 = nn.Linear(hidden_size, 1)  
        
    def forward_cnn(self, x: torch.Tensor):
        x = F.relu(self.conv2d_1(x))
        x = self.maxPool2d_1(x)
        x = F.relu(self.conv2d_2(x))
        x = self.maxPool2d_2(x)
        x = F.relu(self.conv2d_3(x))
        x = self.maxPool2d_3(x)
        x = self.flatten_1(x)
        x = self.fc1(x) 
        x = F.relu(x)
        return x

    def forward_rnn(self, x: torch.Tensor):
        output = []
        h0 = torch.zeros(x.size(0),self.hidden_size).to(self.device)
        
        for frame in x.split(1,dim=1):
            if frame.shape[0]==1:
                frame = torch.squeeze(frame,1)
            elif frame.shape[0]>1:
                frame = torch.squeeze(frame)

            h0 = self.grucell(frame,h0)
            out =  self.fc2(h0)
            output += [out]
        output = torch.cat(output, dim=1)
        return output

    def forward(self, x: torch.Tensor):
        batch_size, timesteps, C, H, W = x.size()
        gru_in = torch.zeros(batch_size, timesteps, self.features).to(self.device)
        
        for i in range(self.num_layers):
            temp = x[:,i,:,:,:]
            temp = self.forward_cnn(temp)
            gru_in[:,i,:] = temp
            
        output = self.forward_rnn(gru_in)
        return output

def load_model(input_model, path):
    """
    loader function to load model for use.
    input_model to indicate base model used and path will indicate the path where the model will be saved
    """
    cp = torch.load(path)
    model = input_model
    return model

def read_video(video_file):

    cap = cv2.VideoCapture(video_file)
    frames = torch.FloatTensor(10, 3 , 224, 224)

    frame_idx = [i-1 for i in range(40, 401) if i%40 ==0] 

    for i in range(len(frame_idx)):

        # get the specific frame
        cap.set(1,frame_idx[i])
        ret, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame)
        frame = frame.permute(2, 0, 1)
        frame_PIL = transforms.ToPILImage()(frame).convert("RGB")
        face = detect_faces(frame_PIL)
        face_tensor = torch.from_numpy(np.array(face)/255) 
        frames[i:] = face_tensor.permute(2, 0, 1)
    
    frames = frames.unsqueeze(0)

    return frames


def get_emotion(a,v):
    angle = math.degrees(math.atan2(a, v))
    #q1:
    if angle > 0:
        if angle < 45:
            return 'pleasure'
        elif angle < 90:
            return 'excitement'
        elif angle < 135:
            return 'arousal'
        elif angle <= 180:
            return 'distress'
    elif angle < 0:
        if angle > -45:
            return 'relaxation'
        elif angle > -90:
            return 'sleepiness'
        elif angle > -135:
            return 'depression'
        elif angle > -180:
            return 'displease'