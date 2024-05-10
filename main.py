#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 10 10:54:18 2024

@author: ajnas
"""
"""
Setups

pip install segmentation-models-pytorch
pip install -U git+https://github.com/albumentations-team/albumentations
pip install --upgrade opencv-contrib-python

Download datasets

git clone https://github.com/parth1620/Human-Segmentation-Dataset-master.git

"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(),"Human-Segmentation-Dataset-master"))

import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import helper_1
#setup configuration
CSV_File = os.path.join(os.getcwd(), 'Human-Segmentation-Dataset-master/train.csv')
DATA_DIR = os.getcwd()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHs = 30
LR = 0.002
IMAGE_SIZE = 320
BATCH_SIZE = 16

ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'

#Read images
df = pd.read_csv(CSV_File)
df.head()

row = df.iloc[0]

image_path = row.images
mask_path = row.masks

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/255.0

#visualization

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

ax1.set_title('IMAGE')
ax1.imshow(image)

ax2.set_title('MASK')
ax2.imshow(mask)

#Split the dataset into train and validation

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)


#Data augmentation
#using  albamentation 
import albumentations as A

def get_train_augs():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
        ], is_check_shapes=False)

def get_valid_augs():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        ], is_check_shapes=False)


#Create custome dataset
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, df, augmentations):
        self.df = df
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_path = row.images
        mask_path = row.masks
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)
        
        if self.augmentations:
            data=self.augmentations(image = image, mask= mask)
            image = data['image']
            mask = data['mask']
            
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        mask = np.transpose(mask, (2,0,1)).astype(np.float32)
        
        image = torch.Tensor(image)/255.0
        mask = torch.round(torch.Tensor(mask)/255.0)
        
        return image, mask
        

trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_valid_augs())


idx = 0
image, mask = trainset[idx]
print(image.shape)
print(mask.shape)
helper_1.show_image(image, mask)

# Load data into dataloader
from torch.utils.data import DataLoader

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE)

#check the shape

for image, mask in trainloader:
    break # just see one batch
    
print(f"One batch shape: {mask.shape}")


#create a segmentation model
#Using Unet
# using Diceloss

from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        
        self.arc = smp.Unet(
            encoder_name= ENCODER,
            encoder_weights= WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None
            )
        
    def forward(self, images, masks = None):
        logits = self.arc(images)
        
        if masks != None:
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1, loss2
        
        return logits
            

model = SegmentationModel()
model.to(DEVICE)

#Create train and validation loop

def train_fn(data_loader, model, optimizer):
    model.train()
    total_loss = 0.0
    
    for images, masks in tqdm(data_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        optimizer.zero_grad()
        logits, loss = model(images, masks)
        optimizer.step()
        
        total_loss+=loss.item()
    return total_loss/len(data_loader)

def eval_fn(data_loader, model):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
        
            logits, loss = model(images, masks)
        
            total_loss+=loss.item()
    return total_loss/len(data_loader)
        

#training loop (Train Model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_valid_loss = np.Inf

for i in range(EPOCHs):
    train_loss = train_fn(trainloader, model, optimizer)
    valid_loss = eval_fn(validloader, model)
    
    if valid_loss<best_valid_loss:
        torch.save(model.state_dict(), 'best_model.pt')
        print("saved model")
        best_valid_loss = valid_loss
        
    print(f"Epoch:{i+1} Train loss:{train_loss} Valid loss:{valid_loss}")


#inference

    