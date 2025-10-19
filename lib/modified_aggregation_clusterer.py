"""
Class that acts as the handler for Modified Aggregation.
Initialize an instance, either from parameters or hyperparameter optimization object.
Prepare data, and cluster. This will replace in part code in optimize.py
"""

from lib.modified_aggregation import ModifiedAggregation
import ROOT
import numpy as np
import lib.base_nn as BNN


class ModifiedAggregationClusterer:
    def __init__(self):
        # Reference 
        pass


    def data(self, config):
        """
        Prepare data by reading the generic root files, read
        adjacency matrix and inverse index transform.
        Returning adj matrix and numpy array?
        Many tfiles? Using yaml file?
        """

        dataloader = BNN.Data()
        adj = np.load("p2_adj.npy")
        iadj = np.load("p2_sim_adj_map2.npy")
        #_,_,arr_npval,arr_npdlab,l_energy = dataloader.generic_data(config)
        d = dataloader.generic_data(config)
        d["adj"] = adj
        d["iadj"] = iadj
        # Return adjacency and mapped values and labels
        #return adj, arr_npval[:, iadj], arr_npdlab[:, iadj], l_energy
        return d


    def event_data(self, ttree, event):
        """
        Prepare data for single event.
        """
        pass

    def cluster_debug(self, tfile, ttree, entry, method, params):
        """
        Cluster a single event and add necessary debug information.
        """
        dataloader = BNN.Data()
        d = dataloader.generic_event(tfile, ttree, entry)
        adj = np.load("p2_adj.npy")
        iadj = np.load("p2_sim_adj_map2.npy")

        ma = ModifiedAggregation(params["seed"], params["agg"])
        tags,_ = ma.run(adj, d["values"][iadj])

        d["tags"] = tags[iadj][iadj][iadj] # ...
        #d["values"] = d["values"][iadj]
        d["values"] = d["values"]

        return d


    def cluster(self, seed, agg, A, values):
        """
        The optimize can then first call data and then divide as necessary
        and train on one and evaluate on the other (same for this though).
        Return values, cl and labels.
        Should take in data and parameters. Then in optimize.py the data
        can be produced once and ran with different parameters.
        """
        ma = ModifiedAggregation(seed,agg)
        # iadj??
        tags = np.zeros_like(values, dtype=np.int32)
        for i in range(len(values)):
            tags[i],_ = ma.run(A, values[i])

        return tags


