"""
Script that does:
    Run set of clustering methods.
    On set of data defined in yaml.
    Run hyperparameter optimzation with Optuna.
"""

import pickle
from datetime import datetime
import sys
import copy
import yaml
import argparse
from typing import Any
import numpy as np
from sklearn.utils import shuffle
from sklearn import cluster, mixture
import ROOT
import optuna
from lib.modified_aggregation import ModifiedAggregation
from lib.modified_aggregation_clusterer import ModifiedAggregationClusterer
import lib.base_nn as BNN
import lib.unet_nn as UNet
from lib.unet_clusterer import UNetClusterer
from lib.train import Train
from lib.focal import FocalH
from lib import efficiency, coverage, vmeas, compute_score, average_energy,              count_clusters,count_labels
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA = "analysis/data.yaml"
METHODS = "analysis/methods.yaml"
TRANSFORMATIONS = "analysis/transformations.yaml"

def load_data(type):
    with open(DATA, "r") as file:
        config = yaml.safe_load(file)
    return config[type]

def load_method(type):
    with open(METHODS, "r") as file:
        config = yaml.safe_load(file)
    return config[type]

def load_transformation():
    with open(TRANSFORMATIONS, "r") as file:
        config = yaml.safe_load(file)
    return config

# Need data function
# Need Optuna objective function -- generalized?
# DIFFERENT FILE?

# Early stopping
# https://github.com/optuna/optuna/issues/1001#issuecomment-596478792


def run(data, method, its):
    analysis_type = data["name"]
    print(method)
    print(f"Running analysis {analysis_type}")
    print(f"\tOn {method['name']}")
    print(f"\tFor {its} iterations.")

    handle_method(data, method, its)

    # Maybe it should be data.yaml and methods.yaml


def save_study(study, data, its, method, model=None):
    bundle = dict()
    bundle["method"] = method
    bundle["study"] = study
    bundle["data"] = data
    bundle["its"] = its

    now = datetime.now()
    timestamp = now.strftime("%d%m%Y_%H%M%S")

    filename = "study_"+method["name"]+"_"+timestamp+".pkl"
    filename_model = "model_"+method["name"]+"_"+timestamp+".pt"
    dir = "studies/"

    if model != None:
        print("Saving model")
        # Save model in case of CNN
        bundle["model_file"] = filename_model
        torch.save(model, dir+filename_model)

    # Save study and extra parameters
    with open(dir+filename, "wb") as f:
        pickle.dump(bundle, f)


def handle_method(data: Any, method: str, its: int):
    """
    Function to handle type of method.
    I decide here what the interface should be for all the methods.
    Hardcode for now but this could also be define in files.
    """
    method_name = method["name"]
    match method_name.lower():
        case "ma":
            print(f"Optimizing {method_name}")
            study = ma_optimize(data, method, its)
            print("Study done. Saving to file.")
            save_study(study, data, its, method)
            print("Saved.")

        case "cnn":
            print(f"Optimizing {method_name}")
            study, model = cnn_optimize(data, method, its)
            print("Study done. Saving to file.")
            save_study(study, data, its, method, model)
            print("Saved.")

        case "dbscan":
            print(f"Optimizing {method_name}")
            study = sklearn_optimize(data, method, its)
            print("Study done. Saving to file.")
            save_study(study, data, its, method)
            print("Saved.")

        case "hdbscan":
            print(f"Optimizing {method_name}")
            study = sklearn_optimize(data, method, its)
            print("Study done. Saving to file.")
            save_study(study, data, its, method)
            print("Saved.")

        case "baygauss":
            print(f"Optimizing {method_name}")
            study = sklearn_optimize(data, method, its)
            print("Study done. Saving to file.")
            save_study(study, data, its, method)
            print("Saved.")

        case _:
            raise ValueError(f"Unknown method: {method_name}")


def p2_data_img(data: Any):
    """
    Converting generic events to images for the CNN.
    """
    files = data["files"]
    Nfiles = len(files)
    event_list = []
    target_list = []
    count_list = []
    mapping_list = []
    dlabel_list = []
    energy_list = []
    dataloader = BNN.Data()
    for file in files:
        tfile = ROOT.TFile(file["path"], "READ")
        ttree = tfile.Get("EventsTree")
        data = dataloader.to_training_tensor(ttree)
        event_list.append(data["event"])
        target_list.append(data["target"])
        count_list.append(data["count"])
        mapping_list.append(data["mapping"])
        dlabel_list.append(data["dlabels"])
        for e in data["energy"]:
            energy_list.append(e)

    events = torch.cat(event_list)
    targets = torch.cat(target_list)
    counts = torch.cat(count_list)
    mapping = torch.cat(mapping_list)
    dlabels = torch.cat(dlabel_list)

    return events, targets, counts, mapping, dlabels, energy_list


# TO-DO: Move to separate file
# ma_opt.py e.g.
# then ma_exec for using a trained/optimized model?
def ma_optimize(data: Any, method: Any, its: int):
    """
    Optimizing modified aggregation.
    """
    print(f"Optimizing modified aggregation for {its} iterations.")

    ma_cluster = ModifiedAggregationClusterer()
    adj, values, labels, _ = ma_cluster.data(data)
    values, labels = shuffle(values, labels)
    clusters = np.zeros_like(labels)

    def objective(trial):
        pars = dict()
        unpack_parameters(pars, trial, method) # Also does trial.suggest
        if pars["agg"]>=pars["seed"]: return float("inf")

        # Metric should be an input parameter defined in yaml
        score = np.zeros(len(values), dtype=np.float32)
        tags = ma_cluster.cluster(pars["seed"], pars["agg"], adj, values)
        score = compute_score(tags, labels, values, "average_intensity_ratio")
        return (score.mean()-1)**2

    study = optuna.create_study()
    study.optimize(objective, n_trials=its)
    return study



def cnn_optimize(data: Any, method: Any, its: int):
    """
    Optimizing CNN.
    """

    # Get data
    #events, targets, counts, mapping, dlabels, energy = p2_data_img(data)
    #adj = np.load("p2_image_adj_21x21.npy")

    unet_cluster = UNetClusterer()

    events, targets, counts, mapping, dlabels, values, energy, adj = unet_cluster.data(data)
    event_train, event_eval, \
    target_train, target_eval, \
    count_train, count_eval, \
    mapping_train, mapping_eval, \
    dlabels_train, dlabels_eval, \
    values_train, values_eval, \
    energy_train, energy_eval \
    = train_test_split(events, targets, counts, mapping, dlabels, values, energy, test_size=0.4)

    dataloader = BNN.Data()

    # Store models
    models = []

    def objective(trial):
        pars = dict()
        unpack_parameters(pars, trial, method) # Also does trial.suggest
        image_criterion = nn.MSELoss() # Could also be hyperparameter
        u = UNet.UNet()

        if pars["agg"]>=pars["seed"]:
            models.append(copy.deepcopy(u))
            return float("inf")


        # Train
        trainer = Train(model=u, image_crit=image_criterion, learning_rate=pars["lr"], momentum=pars["momentum"])
        trainer.run(pars["epochs"], event_train, target_train)
        models.append(copy.deepcopy(u))


        tags = unet_cluster.cluster(event_eval, u, pars["seed"], pars["agg"], adj, dlabels_eval, mapping_eval)
        labels_sq = dlabels_eval.squeeze().detach().numpy()
        values_sq = values_eval.squeeze().detach().numpy()
        score = compute_score(tags, labels_sq, values_sq, "efficiency")

        return (score.mean()-1)**2


    study = optuna.create_study()
    study.optimize(objective, n_trials=its)

    total_memory = sum([sys.getsizeof(model) + get_model_memory_usage(model) for model in models])
    print(f"Total memory usage: {total_memory / (1024 ** 2):.2f} MB for {len(models)} models")

    best_model = None
    try:
        best_model = models[study.best_trial.number]
    except IndexError:
        print("No models stored...")

    return study, best_model


# Should be removed. Mostly temporary.
def get_model_memory_usage(model):
    # Calculate memory usage in bytes
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_size + buffer_size  # in bytes


# Could be a yaml_util file
def unpack_parameters(par_keys, trial, config):
    """
    Unpack from yaml file the parameters and suggest in one go.
    """
    print(config)
    for i,par in enumerate(config["parameters"]):
        print("")
        if par["type"] == "float":
            par_keys[par['name']] = trial.suggest_float(par['name'], float(par['min']), float(par['max']))
        elif par["type"] == "int":
            par_keys[par['name']] = trial.suggest_int(par['name'], int(par['min']), int(par['max']))
        elif par["type"] == "bool":
            par_keys[par['name']] = trial.suggest_categorical(par['name'], [True, False])
        elif par["type"] == "string":
            par_keys[par['name']] = trial.suggest_categorical(par['name'], par['list'])



def sklearn_optimize(data, method, its):
    """
    I probably still want them separate, but they will be almost identical.
    This function can be used to direct the flow.
    """
    trans = load_transformation()["basic"]
    dataloader = BNN.Data()
    x,y,z,dlab = dataloader.generic_data(data)
    method_name = method["name"]

    N = len(x)

    def objective(trial):
        """
        Data transformation
        """
        X = [None]*N # Transformed coordinates
        Y = [None]*N # Clustered labels of transformed coordinates
        cl = [None]*N # Could be array of known size

        data_pars = []
        transformation_types = [name for name, config in trans.items()]
        transformation_choice = trial.suggest_categorical("trans", transformation_types)

        for i in range(N):
            X[i] = transformation(x[i], y[i], z[i], trans[transformation_choice])

        model = None
        """
        Method initialization
        """
        match method_name:
            case "dbscan":
                pars = dict()
                unpack_parameters(pars, trial, method) # Also does trial.suggest
                model = cluster.DBSCAN(**pars)
            case "hdbscan":
                pars = dict()
                unpack_parameters(pars, trial, method) # Also does trial.suggest
                model = cluster.HDBSCAN(**pars)
            case "baygauss":
                pars = dict()
                unpack_parameters(pars, trial, method) # Also does trial.suggest
                model = mixture.BayesianGaussianMixture(**pars)
            case "kmeans":
                pars = dict()
                unpack_parameters(pars, trial, method) # Also does trial.suggest
                model = cluster.KMeans(**pars)
            case _:
                raise ValueError(f"Unknown method: {method_name}")

        """
        Cluster and evaluate
        """
        if method["labels"] == True:
            for i in range(len(X)):
                model.fit(X[i])
                Y[i] = model.labels_.astype(int)
                Y[i] += 1 # Not clustered is 0 in my clustering methods
                cl[i] = dataloader.kdtree_map(X[i], np.column_stack([x[i], y[i]]), Y[i])
        else:
            for i in range(len(X)):
                model.fit(X[i])
                Y[i] = model.predict(X[i])
                Y[i] += 1 # Not clustered is 0 in my clustering methods
                cl[i] = dataloader.kdtree_map(X[i], np.column_stack([x[i], y[i]]), Y[i])

        score_type = "average_energy"
        score = compute_score(cl, dlab, z, score_type)

        fig,ax=plt.subplots(nrows=2, ncols=2, figsize=(10,10))
        fig.suptitle(f"Score [{score_type}]: {score.mean()} (lower=better)")
        for a in ax.flatten():
            a.set_xlim(-10,10)
            a.set_ylim(-10,10)
        test_idx = 55

        iadj = np.load("p2_sim_adj_map2.npy")
        SAT = 4096
        foc = FocalH()
        foc.heatmap(z[test_idx][iadj],dlab[test_idx][iadj],ax[0][0],SAT)
        #ax[0][0].scatter(x[test_idx], y[test_idx], s=50*z[test_idx]/z[test_idx].max())
        ax[0][1].scatter(X[test_idx][:,0], X[test_idx][:,1], marker=".")

        for l in set(Y[test_idx]):
            if l == 0:
                continue
            mask = Y[test_idx] == l
            ax[1][0].scatter(X[test_idx][mask][:,0], X[test_idx][mask][:,1], marker=".")

        foc.heatmap(z[test_idx][iadj],cl[test_idx][iadj],ax[1][1],SAT)

        ax[0][0].set_title(f"Original")
        ax[0][1].set_title(f"Multiplied")
        ax[1][0].set_title(f"{method_name} clustering")
        ax[1][1].set_title(f"Mapped back")

        fig.savefig(f"dump/trans_test_{trial.number}.png", bbox_inches="tight")
        plt.clf()

        return (score.mean() - 1)**2


    study = optuna.create_study()
    study.optimize(objective, n_trials=its)

    return study


def transformation(x, y, z, trans):
    """
    Transform the data according to some transformation method.
    """
    parameters = trans["parameters"]
    pars_unpacked = [par["constant"] for par in parameters]
    dataloader = BNN.Data()
    match trans["name"]:
        case "multiply":
            x, y = dataloader.transform_multiply(x, y, z, *pars_unpacked)
            return np.column_stack([x,y])
        case "3d":
            pass
        case _:
            raise ValueError(f"Unknown transformation: {trans}")
    return x,y,z


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument("--data", type=str, required=True, help="Dataset (define in yaml)")
    parser.add_argument("--method", type=str, required=True, help="Clustering method")
    parser.add_argument("--its", type=int, required=True, help="Number of iterations")

    # Should have: data, method, iterations

    args = parser.parse_args()
    data = load_data(args.data)
    method = load_method(args.method)

    run(data, method, args.its)


if __name__ == "__main__":
    main()
