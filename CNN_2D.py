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
        
        self.cov = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()       
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.cov1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()       
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.cov2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()       
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flat = nn.Flatten()
        self.do = nn.Dropout(p=0.3)
        self.fc = nn.Linear(160,64)
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(64,32)
        self.relu1 = nn.ReLU()        
        
        self.fc2 = nn.Linear(32,5)        
        
    def forward(self, x):
        out = self.cov(x)
        out = self.relu(out)
        out = self.pool(out)
        
        out = self.cov1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        
        out = self.cov2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        
        out = self.flat(out)
        out = self.do(out)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)        
        out = self.fc2(out)

        return out

def fit(model,xtrainN,ytrainN,epochs,hp,gpu,bs=64,lra=0.0001):
    
    transform = transforms.ToTensor()
    ytrain=torch.tensor(ytrainN)
    
    xtrain = torch.zeros((xtrainN.shape[0],1,hp,54))

    for i in range(xtrain.shape[0]):
        xtrain[i] = transform(xtrainN[i])

    if gpu:
        xtrain = xtrain.cuda()
        ytrain= ytrain.cuda()
        model = model.cuda()

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lra)

    N = xtrain.shape[0]

    tb = time.time()
    
    for epoch in range(epochs):

        model=model.train()

        shuf = np.random.permutation(N)

        for i in range(int(N/bs) -bs):

            pred = model(xtrain[[shuf[i * bs : (i*bs) +bs]]].float())

            l = loss(pred,ytrain[[shuf[i * bs : (i*bs) +bs]]].long())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    return model

def evaluate(model,xtestN,ytestN,hp,gpu):
    transform = transforms.ToTensor()
    ytest=torch.tensor(ytestN)
    
    xtest = torch.zeros((xtestN.shape[0],1,hp,54))

    for i in range(xtest.shape[0]):
        xtest[i] = transform(xtestN[i])

    if gpu:
        xtest = xtest.cuda()
        ytest= ytest.cuda()
        model = model.cuda()
    pred = model(xtest.float())
    
    acc = 0
    cm = np.zeros((5,5))
    

    for i in range(xtest.shape[0]):
        cm[torch.argmax(pred[i]).item(),ytest[i].item()] += 1
        if torch.argmax(pred[i]).item()==ytest[i].item():
            acc+=1
    
    return round((acc/len(xtest)*100),2),cm

def predict_(model,inputt,hp,gpu):
    transform = transforms.ToTensor()
    test = torch.zeros((inputt.shape[0],1,hp,54))
    for i in range(inputt.shape[0]):
        img = transform(inputt[i])

    
        test[i] = img
    
    if gpu:
        test = test.cuda()
        model = model.cuda()
    
    pred = model(test)
 
    return pred

def predict(model,func,test_data,hp,gpu):
    final = []
    for i in range(305):
        pred = func(model.eval(),np.array(test_data[i]),hp,gpu)
        predN=torch.argmax(pred,axis=1).cpu().numpy()
        tap=np.argmax(np.bincount(predN))
        final.append(tap)
    return final

def summary(model,in_size):
    model = model.cpu()
    print(sm(model,in_size,device='cpu'))
    
  
