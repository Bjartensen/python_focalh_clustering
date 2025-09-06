import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu
from torch.nn.functional import interpolate
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


#https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3/
class UNet(nn.Module):
    def __init__(self):
        super().__init__()


        # TO-DO:
        # Take start feature input. For instance 64.

        # Encoder

        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        # Pipe
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)


        # Gaussian decoder branch

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, 1, kernel_size=1) # Output

        # Counting branch
        #self.count_fc1 = nn.Linear(256, 128)
        #self.count_fc2 = nn.Linear(128, 1)


    def forward(self, x):
        """
        Forward function propagating an input through the network.
        """
        # Encoder path
        print(x.shape)
        xe11 = relu(self.e11(x))
        print(xe11.shape)
        xe12 = relu(self.e12(xe11))
        print(xe12.shape)
        xp1 = self.pool1(xe12)
        print(xp1.shape)

        xe21 = relu(self.e21(xp1))
        print(xe21.shape)
        xe22 = relu(self.e22(xe21))
        print(xe22.shape)
        xp2 = self.pool2(xe22)
        print(xp2.shape)


        # Pipe
        pipe1 = relu(self.e31(xp2))
        print(pipe1.shape)
        pipe2 = relu(self.e32(pipe1))
        print(pipe2.shape)


        # Gaussian decoder path
        xu1 = self.upconv1(pipe2)
        print(xu1.shape)

        xu11 = torch.cat([xu1, xe22], dim=1)
        print(xu11.shape)
        xd11 = relu(self.d11(xu11))
        print(xd11.shape)
        xd12 = relu(self.d12(xd11))
        print(xd12.shape)

        xu2 = self.upconv2(xd12)
        print(xu2.shape)
        # TO-DO resize should not be hardcoded
        xu2 = interpolate(xu2, size=(21, 21), mode='bilinear', align_corners=True) # Resize
        print(xu2.shape)
        xu22 = torch.cat([xu2, xe12], dim=1)
        print(xu22.shape)
        xd21 = relu(self.d21(xu22))
        print(xd21.shape)
        xd22 = relu(self.d22(xd21))
        print(xd22.shape)

        out = self.outconv(xd22) # Output
        print(out.shape)

        print("DONE COMPUTING")



        # Counting
        #count_blobs = F.adaptive_avg_pool2d(pipe2, (1, 1)).view(pipe2.size(0), -1)
        #count = F.relu(self.count_fc1(count_blobs))
        #count = self.count_fc2(count)

        return out

    def print_dims(self, T):
        """
        Something to print the dimensions at each layer.
        """
        print("Input:", T.shape)

        print("After encode 1:")
        xe11 = relu(self.e11(T))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)
        print(xp1.shape)


        print("After encode 2:")
        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)
        print(xp2.shape)


        print("After pipe:")
        pipe1 = relu(self.e31(xp2))
        pipe2 = relu(self.e32(pipe1))
        print(pipe2.shape)


        print("After decode 1:")
        xu1 = self.upconv1(pipe2)
        xu11 = torch.cat([xu1, xe22], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))
        print(xd12.shape)


        print("After decode 2:")
        xu2 = self.upconv2(xd12)
        xu2 = interpolate(xu2, size=(21, 21), mode='bilinear', align_corners=True) # Resize
        xu22 = torch.cat([xu2, xe12], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))
        print(xd22.shape)

        print("Final:")
        out = self.outconv(xd22)
        print(out.shape)


