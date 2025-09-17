"""
Script that does:
    Run set of clustering methods.
    On set of data defined in yaml.
    Run hyperparameter optimzation with Optuna.
"""

import pickle
import threading
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
from lib.sklearn_clusterer import SklearnClusterer
from lib.train import Train
from lib.focal import FocalH
from lib import metrics# efficiency, coverage, vmeas, compute_score, average_energy,              count_clusters,count_labels
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

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


def run(data, method, its, timestamps, jobs):
    analysis_type = data["name"]
    print(f"Running analysis {analysis_type}")
    print(f"\tOn {method['name']}")
    print(f"\tFor {its} iterations.")

    handle_method(data, method, its, timestamps, jobs)

    # Maybe it should be data.yaml and methods.yaml


def save_study(study, data, its, method, timestamps, model=None):
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

    timestamps["t_optimize_end"] = time.time()
    bundle["timestamps"] = timestamps

    with open(dir+filename, "wb") as f:
        pickle.dump(bundle, f)


def handle_method(data: Any, method: str, its: int, timestamps, jobs: int):
    """
    Function to handle type of method.
    I decide here what the interface should be for all the methods.
    Hardcode for now but this could also be define in files.
    """
    method_name = method["name"]
    match method_name.lower():
        case "ma":
            print(f"Optimizing {method_name}")
            study = ma_optimize(data, method, its, timestamps, jobs)
            print(f"Study done. Best params: {study.best_params}")
            save_study(study, data, its, method, timestamps)
            print("Saved.")

        case "cnn":
            print(f"Optimizing {method_name}")
            study, model = cnn_optimize(data, method, its, timestamps)
            print(f"Study done. Best params: {study.best_params}")
            save_study(study, data, its, method, timestamps, model)
            print("Saved.")

        case "dbscan":
            print(f"Optimizing {method_name}")
            study = sklearn_optimize(data, method, its, timestamps)
            print(f"Study done. Best params: {study.best_params}")
            save_study(study, data, its, method, timestamps)
            print("Saved.")

        case "hdbscan":
            print(f"Optimizing {method_name}")
            study = sklearn_optimize(data, method, its, timestamps)
            print(f"Study done. Best params: {study.best_params}")
            save_study(study, data, its, method, timestamps)
            print("Saved.")

        case "optics":
            print(f"Optimizing {method_name}")
            study = sklearn_optimize(data, method, its, timestamps)
            print(f"Study done. Best params: {study.best_params}")
            save_study(study, data, its, method, timestamps)
            print("Saved.")

        case "baygauss":
            print(f"Optimizing {method_name}")
            study = sklearn_optimize(data, method, its, timestamps)
            print(f"Study done. Best params: {study.best_params}")
            save_study(study, data, its, method, timestamps)
            print("Saved.")

        case "gauss":
            print(f"Optimizing {method_name}")
            study = sklearn_optimize(data, method, its, timestamps)
            print(f"Study done. Best params: {study.best_params}")
            save_study(study, data, its, method, timestamps)
            print("Saved.")

        case "kmeans":
            print(f"Optimizing {method_name}")
            study = sklearn_optimize(data, method, its, timestamps)
            print(f"Study done. Best params: {study.best_params}")
            save_study(study, data, its, method, timestamps)
            print("Saved.")

        case _:
            raise ValueError(f"Unknown method: {method_name}")



# TO-DO: Move to separate file
# ma_opt.py e.g.
# then ma_exec for using a trained/optimized model?
def ma_optimize(data: Any, method: Any, its: int, timestamps, jobs: int):
    """
    Optimizing modified aggregation.
    """
    print(f"Optimizing modified aggregation for {its} iterations.")

    ma_cluster = ModifiedAggregationClusterer()
    d = ma_cluster.data(data)
    values = d["values"]
    labels = d["labels"]
    #values, labels = shuffle(values, labels)

    timestamps["t_data_loaded"] = time.time()

    def objective(trial):

        pars = dict()
        unpack_parameters(pars, trial, method) # Also does trial.suggest
        if pars["agg"]>=pars["seed"]: return float("inf")

        # Metric should be an input parameter defined in yaml
        score = np.zeros(len(values), dtype=np.float32)
        tags = ma_cluster.cluster(pars["seed"], pars["agg"], d["adj"], values)

        #score = compute_score(tags, labels, values, "efficiency")
        score = metrics.separation_efficiency_opt(tags, labels, values, d["energy"])

        return (score.mean()-1)**2

    study = optuna.create_study()
    study.optimize(objective, n_trials=its, n_jobs=jobs)

    timestamps["t_study_finished"] = time.time()

    return study



def cnn_optimize(data: Any, method: Any, its: int, timestamps):
    """
    Optimizing CNN.
    """

    # Get data
    #events, targets, counts, mapping, dlabels, energy = p2_data_img(data)
    #adj = np.load("p2_image_adj_21x21.npy")

    unet_cluster = UNetClusterer()

    d = unet_cluster.data(data)

    timestamps["t_data_loaded"] = time.time()

    events = d["events"]
    targets = d["targets"]
    counts = d["counts"]
    mapping = d["mapping"]
    dlabels = d["labels"]
    values = d["values"]
    energy = d["energy"]

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


        print("Clustering...")
        tags = unet_cluster.cluster(event_eval, u, pars["seed"], pars["agg"], d["adj"], dlabels_eval, mapping_eval)
        labels_sq = dlabels_eval.squeeze().detach().numpy()
        values_sq = values_eval.squeeze().detach().numpy()
        print("Computing score...")
        score = compute_score(tags, labels_sq, values_sq, "efficiency")

        return (score.mean()-1)**2


    study = optuna.create_study()
    study.optimize(objective, n_trials=its)

    timestamps["t_study_finished"] = time.time()

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
def unpack_parameters(par_keys, trial, config, prefix=""):
    """
    Unpack from yaml file the parameters and suggest in one go.
    """
    for i,par in enumerate(config["parameters"]):
        if par["type"] == "float":
            par_keys[par['name']] = trial.suggest_float(prefix+par['name'], float(par['min']), float(par['max']))
        elif par["type"] == "int":
            par_keys[par['name']] = trial.suggest_int(prefix+par['name'], int(par['min']), int(par['max']))
        elif par["type"] == "bool":
            par_keys[par['name']] = trial.suggest_categorical(prefix+par['name'], [True, False])
        elif par["type"] == "string":
            par_keys[par['name']] = trial.suggest_categorical(prefix+par['name'], par['list'])



def sklearn_optimize(data, method, its, timestamps):
    """
    I probably still want them separate, but they will be almost identical.
    This function can be used to direct the flow.
    """
    trans = load_transformation()["basic"]
    dataloader = BNN.Data()

    sk_cluster = SklearnClusterer()
    d = sk_cluster.data(data)

    timestamps["t_data_loaded"] = time.time()

    def objective(trial):
        trans_pars = dict()
        transformation_types = [name for name, config in trans.items()]
        transformation_choice = trial.suggest_categorical("trans::type", transformation_types)
        trans_pars["name"] = transformation_choice
        trans_pars_dict = dict()
        unpack_parameters(trans_pars_dict, trial, trans[transformation_choice], prefix="trans::") # Also does trial.suggest
        trans_pars["parameters"] = trans_pars_dict

        """
        Cluster and evaluate
        """
        # Before this you need to check if oracle
        # use sk_cluster
        # sk_cluster(X,)
        method_pars = dict()
        unpack_parameters(method_pars, trial, method, prefix="method::") # Also does trial.suggest

        tags = sk_cluster.cluster(d, trans_pars, method, method_pars)

        score_type = "separation"
        #score = compute_score(tags, d["labels"], d["values"], score_type)
        score = metrics.separation_efficiency_opt(tags, d["labels"], d["values"], d["energy"])


        test_idx = 55
        fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        fig.suptitle(f"Score [{score_type}]: {score[test_idx]:.3f}, mean: {score.mean():.3f} (1 is best)")
        for a in ax.flatten():
            a.set_xlim(-10,10)
            a.set_ylim(-10,10)

        iadj = np.load("p2_sim_adj_map2.npy")
        SAT = 4096
        foc = FocalH()
        foc.heatmap(d["values"][test_idx][iadj],d["labels"][test_idx][iadj],ax[0],SAT)

        foc.heatmap(d["values"][test_idx][iadj],tags[test_idx][iadj],ax[1],SAT)

        ax[0].set_title(f"Original")
        ax[1].set_title(f"Mapped back")

        fig.savefig(f"dump/trans_test_{trial.number}.png", bbox_inches="tight")
        plt.clf()

        return (score.mean() - 1)**2

    study = optuna.create_study()
    study.optimize(objective, n_trials=its)

    timestamps["t_study_finished"] = time.time()

    return study


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument("--data", type=str, required=True, help="Dataset (define in yaml)")
    parser.add_argument("--method", type=str, required=True, help="Clustering method")
    parser.add_argument("--its", type=int, required=True, help="Number of iterations")
    parser.add_argument("--jobs", type=int, required=False, default=1, help="Number of iterations")


    # Should have: data, method, iterations

    args = parser.parse_args()

    #print(f"--data: {args.data}, --method: {args.method}, --its: {args.its}, --jobs: {args.jobs}")
    data = load_data(args.data)
    method = load_method(args.method)


    timestamps = {"t_optimize_start": time.time()}

    run(data, method, args.its, timestamps, args.jobs)


if __name__ == "__main__":
    main()
