"""
Class that acts as the handler for Modified Aggregation.
Initialize an instance, either from parameters or hyperparameter optimization object.
Prepare data, and cluster. This will replace in part code in optimize.py
"""

from lib.modified_aggregation import ModifiedAggregation
import Optuna
import ROOT



class ModifiedAggregationClusterer:
    def __init__(self):
        # Reference 
        pass

    @classmethod
    def from_study(cls, study):
        """
        Read Optuna study object and use parameters
        """
        pass


    def data(self, config):
        """
        Prepare data by reading the generic root files, read
        adjacency matrix and inverse index transform.
        Returning adj matrix and numpy array?
        Many tfiles? Using yaml file?
        """

        adj = np.load("p2_adj.npy")
        iadj = np.load("p2_sim_adj_map2.npy")

        files = config["files"]
        length = len(files)
        l_npy = []
        l_npx = []
        l_npval = []
        l_npdlab = []

        for file in files:
            npx,npy,npval,npdlab = self.read_tfile(file)
            l_npx.append(npx)
            l_npy.append(npy)
            l_npval.append(npval)
            l_npdlab.append(npdlab)

        arr_npx = np.concatenate(l_npx)
        arr_npy = np.concatenate(l_npy)
        arr_npval = np.concatenate(l_npval)
        arr_npdlab = np.concatenate(l_npdlab)

        # Return adjacency matrix, mapped values and mapped labels?
        pass

    def read_tfile(self, tfile):
        """
        Read tfile and return values
        """

        tfile = ROOT.TFile(file["path"], "READ")
        ttree = tfile.Get("EventsTree")
        Nentries = ttree.GetEntries()

        dataloader = BNN.Data()

        # Hardcoded for prototype 2 (and CAEN/focalsim for saturation)
        # Should be yaml (lol)
        FOCAL2_CELLS = 249
        FOCAL2_SAT = 4096

        npx = np.zeros(Nentries*FOCAL2_CELLS, dtype=np.float32).reshape(Nentries, FOCAL2_CELLS)
        npy = np.zeros(Nentries*FOCAL2_CELLS, dtype=np.float32).reshape(Nentries, FOCAL2_CELLS)
        npval = np.zeros(Nentries*FOCAL2_CELLS, dtype=np.float32).reshape(Nentries, FOCAL2_CELLS)
        npdlab = np.zeros(Nentries*FOCAL2_CELLS, dtype=np.int32).reshape(Nentries, FOCAL2_CELLS)
        num_particles = int(file["particles"])

        for i in range(Nentries):
            ttree.GetEntry(i)
            npx[i] = np.array(ttree.x)
            npy[i] = np.array(ttree.y)
            npval[i] = np.array(ttree.value).clip(max=FOCAL2_SAT)
            l = np.array(ttree.labels)
            f = np.array(ttree.fractions)
            npdlab[i] = dataloader.get_major_labels(l,f,num_particles)
        tfile.Close()

        return npx,npy,npval,npdlab

    def event_data(self, ttree, event):
        """
        Prepare data for single event.
        """
        pass


    def cluster(self, A, v, labels):
        """
        The optimize can then first call data and then divide as necessary
        and train on one and evaluate on the other (same for this though).
        Return values, cl and labels.
        """
        pass


