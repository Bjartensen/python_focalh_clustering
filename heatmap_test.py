from lib.focal import FocalH
from lib.modified_aggregation import ModifiedAggregation
import ROOT
import numpy as np
import lib.base_nn as BNN
import matplotlib.pyplot as plt




def main():
    filename = "/home/bjartur/workspace/python_focalh_clustering/data/E150_P3_N1000.root"
    tfile = ROOT.TFile(filename, "READ")
    ttree = tfile.Get("EventsTree")
    Nentries = ttree.GetEntries()
    ttree.GetEntry(0)

    npx = np.array(ttree.x, dtype=np.float32)
    npy = np.array(ttree.y, dtype=np.float32)
    npval = np.array(ttree.value, dtype=np.float32)
    nplab = np.array(ttree.labels, dtype=np.int32)
    npfracs = np.array(ttree.fractions, dtype=np.float32)

    dataloader = BNN.Data()
    l = dataloader.get_major_labels(nplab, npfracs, 3)

    adj = np.load("p2_adj.npy")
    idx = np.load("p2_cell_idx.npy")
    iadj = np.load("p2_sim_adj_map.npy")

    print(idx)
    print(iadj)

    ma = ModifiedAggregation(800, 100)
    cl,_ = ma.run(adj, npval[iadj])

    fig, ax = plt.subplots()

    foc = FocalH()
    foc.heatmap(npval, l, ax=ax)


    fig.savefig("heatmap_test.png", bbox_inches="tight")



    print("testing heatmap")

main()
