import ROOT # Segmentation error when imported after some of the others... Insanity.
import lib.base_nn as BNN
import lib.unet_nn as UNet
from lib.train import Train
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time


def test():
    print("Initializing data loader")
    dataloader = BNN.Data()
    u = UNet.UNet()
    print("Data loader initialized")
    tfile = ROOT.TFile("/home/bjartur/workspace/focalh_data_transformer/data/converted/200_1000_self_mix_3p_ma_800_100.root")
    ttree = tfile.Get("EventsTree")
    print("Interpolating...")
    ret,coms = dataloader.read_ttree_event(ttree, 3011)
    print("Done")

    #u.print_dims(ret)
    print(ret.shape)
    x = u.forward(ret)
    print(x.shape)
    x = u(ret)
    print(x.shape)
    fig, ax = plt.subplots()
    dataloader.plot_tensor_image(x[0][0].detach().numpy(), ax)
    fig.savefig("forward.png", bbox_inches="tight")


def test_train():
    dataloader = BNN.Data()
    u = UNet.UNet()

    image_criterion = nn.MSELoss()
    count_criterion = nn.SmoothL1Loss()
    lr = 0.21
    momentum = 0.98
    epochs = 1000

    trainer = Train(model=u, image_crit=image_criterion, count_crit=count_criterion, learning_rate=lr, momentum=momentum)

    dir = "/home/bjartur/workspace/focalh_data_transformer/data/converted/"
    files = [
    "350_1000_uniform_noped_generic.root"
    , "350_1000_self_mix_2p.root"
    , "350_1000_self_mix_3p.root"
    , "350_1000_self_mix_5p.root"
    , "350_1000_self_mix_8p.root"
    , "350_1000_self_mix_10p.root"
    ]
    ttrees = [None]*len(files)
    tfiles = [None]*len(files)
    for i in range(len(files)):
        tfiles[i] = ROOT.TFile(dir+files[i])
    for i in range(len(files)):
        ttrees[i] = tfiles[i].Get("EventsTree")


    tfile = ROOT.TFile("/home/bjartur/workspace/focalh_data_transformer/data/converted/350_1000_self_mix_3p.root")
    ttree = tfile.Get("EventsTree")

    tfile2p = ROOT.TFile("/home/bjartur/workspace/focalh_data_transformer/data/converted/350_1000_self_mix_2p.root")
    ttree2p = tfile2p.Get("EventsTree")

    # Make dataset
    N = 10
    image_list = []
    target_list = []
    #target = []
    count_list = []
    for t in ttrees:
        for i in range(N):
            try:
                ret, coms = dataloader.read_ttree_event(t, i)
                target = dataloader.gaussian_class_activation_map(coms, 21, 21, 3)
                count = torch.tensor(len(coms), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            except RuntimeError:
                continue
            count_list.append(count)
            image_list.append(ret)
            target_list.append(target)


    image_tensor = torch.cat(image_list, dim=0)
    target_tensor = torch.cat(target_list, dim=0)
    count_tensor = torch.cat(count_list, dim=0)


    print(image_tensor.shape)
    fig, ax = plt.subplots()
    dataloader.plot_tensor_image(image_tensor[-1][0], ax=ax)
    fig.savefig("train_test_event.png", bbox_inches="tight")

    print(target_tensor.shape)
    fig, ax = plt.subplots()
    dataloader.plot_tensor_image(target_tensor[-1][0], ax=ax)
    fig.savefig("train_test_gaus.png", bbox_inches="tight")

    trainer.run(epochs, image_tensor, target_tensor, count_tensor)




    rows=len(ttrees)
    cols=3
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*3,rows*3))
    for i in range(1):
        ax[i,0].set_title("Input")
        ax[i,1].set_title("Target")
        ax[i,2].set_title("Prediction")



    for i,t in enumerate(ttrees):
        eret, ecoms = dataloader.read_ttree_event(t, N+10)
        etarget = dataloader.gaussian_class_activation_map(ecoms, 21, 21, 3)
        x,c = u(eret)
        print(c)
        dataloader.plot_tensor_image(eret[-1][0], ax=ax[i][0])
        dataloader.plot_tensor_image(etarget[-1][0], ax=ax[i][1])
        dataloader.plot_tensor_image(x[-1][0].detach().numpy(), ax=ax[i][2])

    fig.savefig("eval_pred.png", bbox_inches="tight")


test_train()
