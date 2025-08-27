import ROOT
from lib.focal import FocalH
from lib.modified_aggregation import ModifiedAggregation
import numpy as np
import lib.base_nn as BNN
import matplotlib.pyplot as plt


def centroid(vertices):
    xmin = vertices[:,0].min()
    xmax = vertices[:,0].max()
    ymin = vertices[:,1].min()
    ymax = vertices[:,1].max()
    return ((xmax+xmin)/2).astype(np.float32), ((ymax+ymin)/2).astype(np.float32)

def main():
    filename = "data/E150_P3_N100.root"
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
    labels = dataloader.get_major_labels(nplab, npfracs, 3)

    adj = np.load("p2_adj.npy")
    idx = np.load("p2_cell_idx.npy")
    iadj = np.load("p2_sim_adj_map2.npy")
    #inv = np.argsort(iadj)
    foc = FocalH()


    # Use foc.search()
    correct_order = np.zeros(249,dtype=np.int32)
    for i in range(249):
        correct_order[i] = foc.search(npx[i], npy[i])

    print(f"orig max: {npval.argmax()}, mapped max: {npval[iadj].argmax()}")
    print(f"orig pos: {npx[npval.argmax()]}, {npy[npval.argmax()]}")

    for i,e in enumerate(iadj):
        # something with idx[iadj][i]
        print(f"map {i} to {e}. Value: {npval[i]}. Mapped value: {npval[iadj][i]}. Somehow: {iadj[idx][i]}")

    ma = ModifiedAggregation(800, 100)
    cl,_ = ma.run(adj, npval[iadj])

    fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(20,10))

    #foc.heatmap(npval[correct_order], labels[correct_order], ax=ax[0])
    #foc.heatmap(npval[iadj][iadj][iadj], cl[iadj][iadj], npx, npy, ax=ax[0][0])
    #foc.heatmap(npval[inv], labels[inv], npx, npy, ax=ax[0][0])
    foc.heatmap(npval[iadj], cl, ax=ax[0][0])

    # THREE TIMES TO TRANSFORM CORRECTLY

    for l in set(labels):
        mask = labels == l
        ax[0][1].scatter(npx[mask],npy[mask],s=npval[mask]/10)

    for i in range(249):
        w = foc.detector_width
        c = centroid(foc.paths[i].vertices)
        ax[1][0].text(npx[i]/w + 0.5,npy[i]/w + 0.5,str(i),fontsize=6,horizontalalignment='center',verticalalignment='center')
        ax[1][1].text(c[0]/w + 0.5,c[1]/w + 0.5,str(i),fontsize=6,horizontalalignment='center',verticalalignment='center')
        #ax[1][1].text(npx[i]/w + 0.5,npy[i]/w + 0.5,str(foc.search(npx[i],npy[i])),fontsize=6,horizontalalignment='center',verticalalignment='center')
        ax[1][2].text(npx[iadj][i]/w + 0.5,npy[iadj][i]/w + 0.5,str(i),fontsize=6,horizontalalignment='center',verticalalignment='center')


    fig.savefig("heatmap_test.png", bbox_inches="tight")



    print("testing heatmap")

main()
