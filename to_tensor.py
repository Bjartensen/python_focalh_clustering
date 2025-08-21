"""
Transform root file data to tensor data and save to file.
"""

import ROOT # Segmentation error when imported after some of the others... Insanity.
import lib.base_nn as BNN
import argparse
import torch
import os

def load_file(filename):
    tfile = ROOT.TFile(filename)
    ttree = tfile.Get("EventsTree")
    return tfile, ttree


def to_tensor(ttree):
    dataloader = BNN.Data()
    data = dataloader.to_training_tensor(ttree)
    return data


def convert(filename):
    print("Converting...")
    tfile, ttree = load_file(filename)
    data = to_tensor(ttree)
    print(f"Sucessfully converted {len(data["event"])} events")

    savename = os.path.basename(filename)
    savename = savename.rstrip(".root") + "_IMG.pt"

    print(f"Saving as " + savename)
    torch.save(data, savename)

def main():
    parser = argparse.ArgumentParser(description="Transform events to tensor images")
    parser.add_argument("--file", type=str, required=True, help="Root file")

    args = parser.parse_args()
    print("Transforming", args.file)

    convert(args.file)


if __name__ == "__main__":
    main()
