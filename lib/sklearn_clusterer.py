"""
Object to cluster with different sklearn types.
Will need a switch function (from optimize.py probably)
and also the data transforms. Everything will probably be moved here.
"""


from sklearn import cluster, mixture
import lib.base_nn as BNN
import numpy as np

class SklearnClusterer:
    def __init__(self):
        # Reference 
        pass




    def data(self, config):
        """
        Prepare generic data.
        """

        dataloader = BNN.Data()
        iadj = np.load("p2_sim_adj_map2.npy")

        data = dataloader.generic_data(config)
        data["adj"] = iadj

        # Return adjacency and mapped values and labels
        return data


    def transformation(self, x, y, z, trans):
        """
        Transform the data according to some transformation method.
        Why is this in this class?
        """
        parameters = trans["parameters"]
        pars_unpacked = [value for key,value in parameters.items()]
        dataloader = BNN.Data()
        match trans["name"]:
            case "multiply":
                x, y = dataloader.transform_multiply(x, y, z, **parameters)
                return np.column_stack([x,y])
            case "3d":
                pass
            case _:
                raise ValueError(f"Unknown transformation: {trans}")
        return x,y,z


    def event_data(self, ttree, event):
        """
        Prepare data for single event.
        """
        pass


    def handle_method(self, method_name, pars):
        model = None
        match method_name:
            case "dbscan":
                model = cluster.DBSCAN(**pars)
            case "hdbscan":
                model = cluster.HDBSCAN(**pars)
            case "baygauss":
                model = mixture.BayesianGaussianMixture(**pars)
            case "kmeans":
                model = cluster.KMeans(**pars)
            case _:
                raise ValueError(f"Unknown method: {method_name}")
        return model

    #def cluster(self, seed, agg, A, values):
    # Just pass in a named dictionary
    def cluster(self, data, trans, method, method_pars):
        """
        Transform and cluster
        """

        # Unpack trans and method pars?

        x = data["x"]
        y = data["y"]
        z = data["values"]
        dataloader = BNN.Data()

        Nevents = len(data["values"])
        tags = [None]*Nevents
        X = [None]*Nevents # Transformed coordinates
        Y = [None]*Nevents # Clustered labels of transformed coordinates

        # transformation(x, y, z, trans):
        for i in range(Nevents):
            X[i] = self.transformation(x[i], y[i], z[i], trans)

        model = self.handle_method(method["name"], method_pars)

        for i in range(Nevents):
            if method["labels"] == True:
                model.fit(X[i])
                Y[i] = model.labels_.astype(int)
                Y[i] += 1 # Not clustered is 0 in my clustering methods
                tags[i] = dataloader.kdtree_map(X[i], np.column_stack([x[i], y[i]]), Y[i])
            else:
                model.fit(X[i])
                Y[i] = model.predict(X[i])
                Y[i] += 1 # Not clustered is 0 in my clustering methods
                tags[i] = dataloader.kdtree_map(X[i], np.column_stack([x[i], y[i]]), Y[i])


        return tags


