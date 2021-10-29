#!/bin/bash
from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np 

# @misc{mulitdigitmnist,
#   author = {Sun, Shao-Hua},
#   title = {Multi-digit MNIST for Few-shot Learning},
#   year = {2019},
#   journal = {GitHub repository},
#   url = {https://github.com/shaohua0116/MultiDigitMNIST},
# }


class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # Step 1:
        # 1. conv2d(Depth = depth of input feature volume, 
        # 2. output channels = output should have depth of this value,  
        # 3. Shape of your kernel
        
        # 1
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 10)
        # 2, 3, 4
        self.conv1 = nn.Conv2d(1, 40, 5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(40, 80, 5, padding=2)
        
        # 2 max pool layers --> Half 28 twice
        self.newfc = nn.Linear(80*7*7, 100)
        self.newfc2 = nn.Linear(100, 10)
        
        # 5
        self.conv1 = nn.Conv2d(1, 40, 5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(40, 80, 5, padding=2)
        self.newfc = nn.Linear(80*7*7, 1000)
        self.newfc2 = nn.Linear(1000, 10)


        # Activations
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)


        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)

    def numFlatFeatures(self, x):
      size = x.size()[1:]
      num_feat = 1
      for s in size:
        num_feat *= s
      return num_feat


    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # print("MODEL 1")
        # print(X.shape)
        # X = X.view(-1, 28*28)
        # print(X.shape)

        X = self.fc1(X)
        X = self.sigmoid(X)
        X = self.fc2(X)
        
        fcl = X
        # Uncomment the following return stmt once method implementation is done.
        return  fcl
        # Delete line return NotImple mentedError() once method is implemented.
        # return NotImplementedError()

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # print("SHAPE", X.shape)
        # print("0 shape:", X.shape[0])
        print("MODEL 2")
        X = self.sigmoid(self.conv1(X))
        X = self.pool(X)
        X = self.sigmoid(self.conv2(X))
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)

        X = self.newfc(X)

        fcl = X
        # Uncomment the following return stmt once method implementation is done.
        return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # print("MODEL 3")
        X = self.relu(self.conv1(X))
        X = self.pool(X)
        X = self.relu(self.conv2(X))
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)

        X = self.newfc(X)

        fcl = X

        # Uncomment the following return stmt once method implementation is done.
        return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # print("MODEL 4")
        X = self.relu(self.conv1(X))
        X = self.pool(X)
        X = self.relu(self.conv2(X))
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)
        X = self.newfc(X)
        X = self.newfc2(X)

        fcl = X

        # Uncomment the following return stmt once method implementation is done.
        return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # print("MODEL 5")
        X = self.relu(self.conv1(X))
        X = self.pool(X)
        X = self.relu(self.conv2(X))
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)
        X = self.newfc(X)
        X = self.dropout(X)
        X = self.newfc2(X)
        X = self.dropout(X)

        fcl = X
        # Uncomment the following return stmt once method implementation is done.
        return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()
    
    






