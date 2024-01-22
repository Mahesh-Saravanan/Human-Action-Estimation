import cv2
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from torchsummary import summary as sm
import dataload
import importlib
importlib.reload(dataload)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__() 
      
        self.Conv = nn.Conv1d(54, 48, kernel_size=2, stride=1 , padding=1)
        self.relu=nn.ReLU()
        self.pool = nn.MaxPool1d(3,1) 

        self.Conv2 = nn.Conv1d(48, 32, kernel_size=2, stride=1 , padding=1)
        self.relu=nn.ReLU()
        self.pool2 = nn.MaxPool1d(3,1)

        self.dropout = nn.Dropout(p=0.5) 

        self.Conv3 = nn.Conv1d(32, 1, kernel_size=2, stride=1 , padding=1)
        self.relu=nn.ReLU()
        self.pool3 = nn.MaxPool1d(3,1)

        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(27 ,64, bias = True)
        self.relu=nn.ReLU()

        self.fc2 = nn.Linear(64,32)
        self.relu = nn.ReLU()        

        self.fc3 = nn.Linear(32,5)
    
    def forward(self, input): 

        out = self.Conv(input)
        out = self.relu(out)
        out = self.pool(out)

        out = self.Conv2(out)
        out = self.relu(out)
        out = self.pool2(out)

        out = self.dropout(out)

        out = self.Conv3(out)
        out = self.relu(out)
        out = self.pool3(out)

        out = self.dropout(out)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)

        return out

class LoadData(Dataset):
    def __init__(self,features,labels):
        self.features = features
        self.labels = labels
        self.samples_number = features.shape[0]
    def __getitem__(self,index):
        return self.features[index], self.labels[index]
    def __len__(self):
        return self.samples_number

def fit(model,training_features,training_labels,epochs,hp,cuda,bs = 64,learning_rate = 0.0001):
    if(cuda == True):
        model = model.cuda()
        #training_features = training_features.cuda()
        #training_labels = training_labels.cuda()
    dataset = LoadData(training_features,training_labels)
    #dataloader = DataLoader(dataset=dataset , batch_size=batch_size , shuffle=False)
    predicted_class = None
    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),learning_rate)
    loss_buffer= []
    thirty_preds = []
    preds_buffer = []
  
    for epoch in range(epochs):
        for i in range (len(dataset)):
            index = np.random.randint(0, len(dataset)-1)

            input = torch.tensor(dataset[i][0])
            label = torch.tensor(dataset[i][1])
            if cuda:
                input = torch.tensor(dataset[i][0]).cuda()
                label = torch.tensor(dataset[i][1]).cuda()
            input = torch.reshape(input.T,(1 , 54 , hp))

            preds = model.forward(input.float()) 


            # the target(label) i give for the loss function is a list(torch) with size 30
            target = torch.zeros_like(preds)
            target[0][0][label] = 1

            loss = Loss(preds[0], target[0])

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            loss_buffer.append(loss.item())

            _, predicted_class = torch.max(preds[0],1)
            preds_buffer.append(predicted_class.item())

            #print("\r",f"Predicted Class {predicted_class.item()} -- True Class {label}",end="")

        #av_loss = np.sum(loss_buffer) / len(loss_buffer)
        #print(f"\nLoss on epoch {epoch} is {av_loss}\n")


    return model

def evaluate(model,training_features,training_labels,hp,gpu):
    dataset = LoadData(training_features,training_labels)
    if gpu:
        model=model.cuda()
    acc=0
    cm=np.zeros((5,5))
    for i in range(len(dataset)):
        index = np.random.randint(0, len(dataset)-1)

        input = torch.tensor(dataset[i][0])
        if gpu:
            input = input.cuda()


        input = torch.reshape(input.T,(1 , 54 , hp))

        preds = model.forward(input.float())
        cm[torch.argmax(preds).item() , training_labels[i]] +=1
        if torch.argmax(preds).item() == training_labels[i]:
            acc+=1
    
    
    return round((acc/len(dataset)*100),2),cm

def predict(model1d,testdata,hp,gpu):
    final1 = []
    for i in range(305):
        temp = np.array(testdata[i])
        templist = []
        for j in range(temp.shape[0]):
            image = torch.tensor(temp[j])
            temp1 = torch.reshape(image.T,(1 , 54 , hp))
            if gpu:
                temp1=temp1.cuda()
            predic = model1d.forward(temp1.float())
            templist.append(torch.argmax(predic).item())
        tap=np.argmax(np.bincount(templist))

        final1.append(tap) 
    return final1
def summary(model,in_size):
    model = model.cpu()
    print(sm(model,in_size,device="cpu"))
    
    
