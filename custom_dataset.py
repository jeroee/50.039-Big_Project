'''
custom_dataset.py creates custom datasets for training, validation, testing.
Make the respective changes to the paths based on directory
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
import cv2

from mtcnn.src import detect_faces, show_bboxes
from PIL import Image

class Train_Dataset():
    def __init__(self):

        self.groups = 'train'

        # Path to images for different parts of the dataset
        self.dataset_paths = {'video': './data/video/train_trim/',\
                              'valence': './data/label/valence/train_trim/',\
                              'arousal': './data/label/arousal/train_trim/'} 
        
        #self.metadata = pd.read_csv('meta.csv', header=None)
        self.vids = os.listdir(self.dataset_paths['video'])
        self.valences = os.listdir(self.dataset_paths['valence'])
        self.arousals = os.listdir(self.dataset_paths['arousal'])
    
    def __len__(self):
        return len(self.vids)
        
    def read_video(self, video_file, attribute_file):

        cap = cv2.VideoCapture(video_file)
        frames = torch.FloatTensor(10, 3 , 224, 224)
        
        arousal_path = self.dataset_paths['arousal'] + attribute_file
        valence_path = self.dataset_paths['valence'] + attribute_file
        
        arousal_df = pd.read_csv(arousal_path, header=None)
        valence_df = pd.read_csv(valence_path, header=None)
        
        arousal = []
        valence = []
        
        frame_idx = [i-1 for i in range(40, 401) if i%40 ==0] # frames we will be looking at
        
        for i in range(len(frame_idx)):
            
            # get the specific frame
            cap.set(1,frame_idx[i])
            ret, frame = cap.read()
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            frame_PIL = transforms.ToPILImage()(frame).convert("RGB")
            face = detect_faces(frame_PIL)
            #face_tensor = torch.from_numpy(np.array(face))
            face_tensor = torch.from_numpy(np.array(face)/255) # normalize?
            frames[i:] = face_tensor.permute(2, 0, 1)
        
            arousal.append(arousal_df[0][i])
            valence.append(valence_df[0][i])
            
        arousal = torch.FloatTensor(arousal)
        valence = torch.FloatTensor(valence)
        return frames, arousal, valence

    def __getitem__(self, index):
        path = self.dataset_paths['video']
        vid_filename = self.vids[index]
        vid_path = path + vid_filename
        
        attribute_file = self.valences[index]
        
        frames, arousal, valence = self.read_video(vid_path, attribute_file)
        video_name = self.vids[index]
        return frames, arousal, valence, video_name
    
class Valid_Dataset():
    def __init__(self):

        self.groups = 'train'

        # Path to images for different parts of the dataset
        self.dataset_paths = {'video': './data/video/valid_trim/',\
                              'valence': './data/label/valence/valid_trim/',\
                              'arousal': './data/label/arousal/valid_trim/'} 
        
        #self.metadata = pd.read_csv('meta.csv', header=None)
        self.vids = os.listdir(self.dataset_paths['video'])
        self.valences = os.listdir(self.dataset_paths['valence'])
        self.arousals = os.listdir(self.dataset_paths['arousal'])
    
    def __len__(self):
        return len(self.vids)
        
    def read_video(self, video_file, attribute_file):

        cap = cv2.VideoCapture(video_file)
        frames = torch.FloatTensor(10, 3 , 224, 224)
        
        arousal_path = self.dataset_paths['arousal'] + attribute_file
        valence_path = self.dataset_paths['valence'] + attribute_file
        
        arousal_df = pd.read_csv(arousal_path, header=None)
        valence_df = pd.read_csv(valence_path, header=None)
        
        arousal = []
        valence = []
        
        frame_idx = [i-1 for i in range(40, 401) if i%40 ==0] # frames we will be looking at
        
        for i in range(len(frame_idx)):
            
            # get the specific frame
            cap.set(1,frame_idx[i])
            ret, frame = cap.read()
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            frame_PIL = transforms.ToPILImage()(frame).convert("RGB")
            face = detect_faces(frame_PIL)
            #face_tensor = torch.from_numpy(np.array(face))
            face_tensor = torch.from_numpy(np.array(face)/255) # normalize?
            frames[i:] = face_tensor.permute(2, 0, 1)
        
            arousal.append(arousal_df[0][i])
            valence.append(valence_df[0][i])
            
        arousal = torch.FloatTensor(arousal)
        valence = torch.FloatTensor(valence)
        return frames, arousal, valence

    def __getitem__(self, index):
        path = self.dataset_paths['video']
        vid_filename = self.vids[index]
        vid_path = path + vid_filename
        
        attribute_file = self.valences[index]
        
        frames, arousal, valence = self.read_video(vid_path, attribute_file)
        video_name = self.vids[index]
        return frames, arousal, valence, video_name
    
class Test_Dataset():
    def __init__(self):

        self.groups = 'train'

        # Path to images for different parts of the dataset
        self.dataset_paths = {'video': './data/video/test_trim/',\
                              'valence': './data/label/valence/test_trim/',\
                              'arousal': './data/label/arousal/test_trim/'} 
        
        #self.metadata = pd.read_csv('meta.csv', header=None)
        self.vids = os.listdir(self.dataset_paths['video'])
        self.valences = os.listdir(self.dataset_paths['valence'])
        self.arousals = os.listdir(self.dataset_paths['arousal'])
    
    def __len__(self):
        return len(self.vids)
        
    def read_video(self, video_file, attribute_file):

        cap = cv2.VideoCapture(video_file)
        frames = torch.FloatTensor(10, 3 , 224, 224)
        
        arousal_path = self.dataset_paths['arousal'] + attribute_file
        valence_path = self.dataset_paths['valence'] + attribute_file
        
        arousal_df = pd.read_csv(arousal_path, header=None)
        valence_df = pd.read_csv(valence_path, header=None)
        
        arousal = []
        valence = []
        
        frame_idx = [i-1 for i in range(40, 401) if i%40 ==0] # frames we will be looking at
        
        for i in range(len(frame_idx)):
            
            # get the specific frame
            cap.set(1,frame_idx[i])
            ret, frame = cap.read()
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            frame_PIL = transforms.ToPILImage()(frame).convert("RGB")
            face = detect_faces(frame_PIL)
            #face_tensor = torch.from_numpy(np.array(face))
            face_tensor = torch.from_numpy(np.array(face)/255) # normalize?
            frames[i:] = face_tensor.permute(2, 0, 1)
        
            arousal.append(arousal_df[0][i])
            valence.append(valence_df[0][i])
            
        arousal = torch.FloatTensor(arousal)
        valence = torch.FloatTensor(valence)
        return frames, arousal, valence

    def __getitem__(self, index):
        path = self.dataset_paths['video']
        vid_filename = self.vids[index]
        vid_path = path + vid_filename
        attribute_file = self.valences[index]
        
        frames, arousal, valence = self.read_video(vid_path, attribute_file)
        video_name = self.vids[index]
        return frames, arousal, valence, video_name