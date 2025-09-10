"""
Object to cluster with different sklearn types.
Will need a switch function (from optimize.py probably)
and also the data transforms. Everything will probably be moved here.
"""


from sklearn import cluster, mixture
import lib.base_nn as BNN
import numpy as np
from lib.metrics import count_labels

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


    def handle_method(self, method_name, pars, n_clusters=None):
        model = None
        match method_name:
            # Non-parametric
            case "dbscan":
                model = cluster.DBSCAN(**pars)
            case "hdbscan":
                model = cluster.HDBSCAN(**pars)
            case "optics":
                model = cluster.OPTICS(**pars)

            # Parametric
            case "baygauss":
                model = mixture.BayesianGaussianMixture(n_clusters, **pars)
            case "kmeans":
                model = cluster.KMeans(n_clusters, **pars)
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


        # if not parametric, define here
        model = self.handle_method(method["name"], method_pars)

        # if method is parametric do oracle version
        for i in range(Nevents):
            # if parametric: Move into loop in case of parametric
            # model = self.handle_method(method["name"], n_clusters, method_pars)
            try:

                if method["parametric"]:
                    # use metrics.count()?
                    model = self.handle_method(method["name"], method_pars, count_labels(data["labels"][i]))
                model.fit(X[i])

                # Slightly different interface
                if method["labels"] == True:
                    Y[i] = model.labels_.astype(int)
                else:
                    Y[i] = model.predict(X[i])

                Y[i] += 1 # Not clustered is 0 in my clustering methods
                tags[i] = dataloader.kdtree_map(X[i], np.column_stack([x[i], y[i]]), Y[i])
            except ValueError:
                Y[i] = np.zeros(len(X[i]))
                tags[i] = dataloader.kdtree_map(X[i], np.column_stack([x[i], y[i]]), Y[i])
                print(f"ValueError")

        return tags


