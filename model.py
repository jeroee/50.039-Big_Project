'''
model.py creates custom model architectures for predicting arousal and valence emotion values
'''


import torch , torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from collections import OrderedDict
from torchvision import models
import pytz
from datetime import datetime

from custom_dataset import Train_Dataset, Valid_Dataset, Test_Dataset
from utils import predict_video, plot_curves, save_model,load_model, validation_a, validation_v

class CnnGru(nn.Module):
    def __init__(self,features,num_layers,hidden_size,batch_size):
        '''
        Initialising parameters used in the cnn-gru model
        '''
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
        '''
        building custom cnn (inspired from lenet)
        '''
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
        '''
        building custom rnn (gru)
        '''
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
    
def train_arousal(model,train_loader,val_loader, lr=0.001, n_epochs=5, save_path='./'):
    print(f'start training arousal model.....') 
    print(f'total epochs: {n_epochs}')
    device = 'cuda'
    start = time.time()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    print_every = len(train_loader)
    steps = 0 
    train_loss = 0
    val_loss_lst=[]
    train_loss_lst=[]
    Singapore = pytz.timezone('Asia/Singapore')
    
    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        batch = 1
        for frames, arousal, valence, video_name in train_loader:
            print(f'epoch:{epoch} batch:{batch}')
            frames, arousal, valence = frames.to(device),arousal.to(device),valence.to(device)
            steps+=1
            
            optimizer.zero_grad()
            output = model.forward(frames)
            loss = criterion(output, arousal)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # At the end of every epoch ... (print_every is the length of train_loader)
            if steps % print_every == 0:
                # Eval mode
                model.eval()
                # Turn off gradients for validation
                with torch.no_grad():
                    val_loss = validation_a(model, val_loader, criterion, device)
                    #print(val_loss)
                print("Epoch: {}/{} - ".format(epoch, n_epochs),
                      "Time: {} ".format(datetime.now(Singapore)),
                      "Training Loss: {:.3f} - ".format(train_loss/len(train_loader)),
                      "Validation Loss: {:.3f} - ".format(val_loss/len(val_loader)))               
                                                                
                val_loss_lst.append(val_loss/len(val_loader))
                train_loss_lst.append(train_loss/len(train_loader))
                train_loss = 0
            batch+=1
    
    print('model: arousal - epochs:', n_epochs)
    print(f"Run time: {(time.time() - start)/60:.3f} min")
    # saving model
    save_model(model,save_path)
    
    return model, train_loss_lst, val_loss_lst


def train_valence(model,train_loader,val_loader, lr=0.001, n_epochs=5, save_path='./'):
    print(f'start training valence model.....') 
    print(f'total epochs: {n_epochs}')
    device = 'cuda'
    start = time.time()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    print_every = len(train_loader)
    steps = 0 
    train_loss = 0
    val_loss_lst=[]
    train_loss_lst=[]
    Singapore = pytz.timezone('Asia/Singapore')
    
    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        batch = 1
        for frames, arousal, valence, video_name in train_loader:
            print(f'epoch:{epoch} batch:{batch}')
            frames, arousal, valence = frames.to(device),arousal.to(device),valence.to(device)
            steps+=1
            
            optimizer.zero_grad()
            output = model.forward(frames)
            loss = criterion(output, valence)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # At the end of every epoch ... (print_every is the length of train_loader)
            if steps % print_every == 0:
                # Eval mode
                model.eval()
                # Turn off gradients for validation
                with torch.no_grad():
                    val_loss = validation_v(model, val_loader, criterion, device)
                    #print(val_loss)
                print("Epoch: {}/{} - ".format(epoch, n_epochs),
                      "Time: {} ".format(datetime.now(Singapore)),
                      "Training Loss: {:.3f} - ".format(train_loss/len(train_loader)),
                      "Validation Loss: {:.3f} - ".format(val_loss/len(val_loader)))               
                                                                
                val_loss_lst.append(val_loss/len(val_loader))
                train_loss_lst.append(train_loss/len(train_loader))
                train_loss = 0
            batch+=1
    
    print('model: valence - epochs:', n_epochs)
    print(f"Run time: {(time.time() - start)/60:.3f} min")
    # saving model
    save_model(model,save_path)
    
    return model, train_loss_lst, val_loss_lst