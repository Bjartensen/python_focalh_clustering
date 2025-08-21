import ROOT # Segmentation error when imported after some of the others... Insanity.
import lib.base_nn as BNN
import torch
import matplotlib.pyplot as plt
import numpy as np




def main():

    tfile = ROOT.TFile("/home/bjartur/workspace/focalh_data_transformer/data/converted/E150_P1_N1000.root")
    ttree = tfile.Get("EventsTree")

    dataloader = BNN.Data()
    print("Interpolating...")
    ret1,coms1,dlab1,map1 = dataloader.read_ttree_event(ttree, 0)
    ret2,coms2,dlab2,map2 = dataloader.read_ttree_event(ttree, 1)
    print("Done")


    print("vals:",ret1[0][0].ravel())
    print("labels:",dlab1)
    print("map:",map1)
    print("pixel shape:",ret1[0][0].ravel().shape)
    print("label shape:",dlab1.shape)
    print("map shape:",map1.shape)

    fig, ax = plt.subplots()
    dataloader.plot_tensor_image(ret1[0][0]+ret2[0][0], ax=ax)
    fig.savefig("inverse_original.png", bbox_inches="tight")

main()
