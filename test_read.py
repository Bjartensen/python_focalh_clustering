import ROOT # Segmentation error when imported after some of the others... Insanity.
import lib.base_nn as BNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time


print("Initializing data loader")

dataloader = BNN.Data()


print("Data loader initialized")

tfile = ROOT.TFile("/home/bjartur/workspace/focalh_data_transformer/data/converted/200_1000_self_mix_3p_ma_800_100.root")
ttree = tfile.Get("EventsTree")

print("Interpolating...")
ret,coms = dataloader.read_ttree_event(ttree, 3011)
print("Done")


print(coms)
gaus_img = dataloader.gaussian_class_activation_map(coms, 21, 21, 3)
fig, ax = plt.subplots()
dataloader.plot_tensor_image(gaus_img, ax=ax)
fig.savefig("gauss_image.png", bbox_inches="tight")

fig, ax = plt.subplots()
dataloader.plot_tensor_physical(ret[0][0], ax=ax)
fig.savefig("tensor_physical.png", bbox_inches="tight")

fig, ax = plt.subplots()
dataloader.plot_tensor_image(ret[0][0], ax=ax)
fig.savefig("tensor_image.png", bbox_inches="tight")

sobel_x = torch.tensor([[1,0,-1],
                        [2,0,-2],
                        [1,0,-1]], dtype=torch.float32)
sobel_x = sobel_x.view(1,1,3,3)

sobel_y = torch.tensor([[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]], dtype=torch.float32)
sobel_y = sobel_y.view(1,1,3,3)

edge_x = F.conv2d(ret, sobel_x, padding=1)
edge_y = F.conv2d(ret, sobel_y, padding=1)
print("orig:", ret.shape)
print("convolved:", edge_x.shape)

gradient_magnitude = torch.sqrt(edge_x**2 + edge_y**2)

# If you want to normalize the result to be in the range [0, 1]
gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())


fig, ax = plt.subplots()
dataloader.plot_tensor_image(gradient_magnitude[0][0], ax=ax)
fig.savefig("edges.png", bbox_inches="tight")

