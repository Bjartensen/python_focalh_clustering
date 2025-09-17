import ROOT
import pickle
from datetime import datetime
import sys
import copy
import yaml
import argparse
from typing import Any
import numpy as np
import optuna
from lib.modified_aggregation_clusterer import ModifiedAggregationClusterer
from lib.unet_clusterer import UNetClusterer
from lib.sklearn_clusterer import SklearnClusterer
from lib.focal import FocalH
import lib.misc_util
from lib import metrics #efficiency, coverage, vmeas, compute_score, average_energy,              count_clusters,count_labels
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA = "analysis/data.yaml"
METHODS = "analysis/methods.yaml"
TRANSFORMATIONS = "analysis/transformations.yaml"
EVALUATION_DIRECTORY = "evaluation/"

def load_data(type):
    with open(DATA, "r") as file:
        config = yaml.safe_load(file)
    return config[type]


def run(data: Any, study: Any):
    print("Running evaluation")
    #tags, labels, values, energy = handle_method(data, study)
    result = handle_method(data, study)
    tags = result["tags"]

    #result = dict()

    print("Computing metrics...")

    # Compute different things
    # Efficiency
    eff = metrics.compute_score(result["tags"], result["labels"], result["values"], "efficiency")
    sep_eff = metrics.separation_efficiency(result["tags"], result["labels"], result["values"], result["energy"], linearity_yaml="test", energy_resolution_yaml="test")
    vmeas = metrics.compute_score(result["tags"], result["labels"], result["values"], "vmeasure")


    #vmeas_weighted = compute_score(tags, labels, values, "vmeasure_weighted")
    coverage = metrics.compute_score(result["tags"], result["labels"], result["values"], "coverage")
    particles = metrics.compute_score(result["tags"], result["labels"], result["values"], "count_labels")
    avg_energy = metrics.average_energy(result["energy"])

    """
    result["data"] = data
    result["study"] = study
    result["tags"] = d["tags"]
    result["labels"] = labels
    result["values"] = values
    """
    result["data"] = data
    result["efficiency"] = eff
    result["separation"] = sep_eff
    result["vmeasure"] = vmeas
    #result["vmeasure_weighted"] = vmeas_weighted
    result["coverage"] = coverage
    result["particles"] = particles
    result["avg_energy"] = avg_energy

    study["eval"] = result

    print("Done.")
    print("Saving...")

    now = datetime.now()
    timestamp = now.strftime("%d%m%Y_%H%M%S")
    filename = "eval_"+study["method"]["name"]+"_"+timestamp+".pkl"
    dir = EVALUATION_DIRECTORY

    with open(dir+filename, "wb") as f:
        pickle.dump(study, f)

    print("Saved.")


def handle_method(data: Any, study: Any):
    name = study["method"]["name"]
    print(f"Clustering with {name}")
    if name == "ma":
        pars = study["study"].best_params
        cluster = ModifiedAggregationClusterer()
        d = cluster.data(data)
        d["tags"] = cluster.cluster(pars["seed"], pars["agg"], d["adj"], d["values"])
        return d
    elif name == "cnn":
        pars = study["study"].best_params
        cluster = UNetClusterer()
        #events, targets, counts, mapping, labels, values, energy, adj = cluster.data(data)
        d = cluster.data(data)
        p = Path(study["load_path"])
        u = torch.load(str(p.parent)+"/"+study["model_file"], weights_only=False)
        d["tags"] = cluster.cluster(d["events"], u, pars["seed"], pars["agg"], d["adj"], d["labels"], d["mapping"])

        # Weird that they are even tensors...
        d["labels"] = d["labels"].squeeze().detach().numpy()
        d["values"] = d["values"].squeeze().detach().numpy()
        return d
    elif name in ["hdbscan", "dbscan"]:
        pars = study["study"].best_params
        trans_pars, method_pars = misc_util.split_trans_method(pars)
        cluster = SklearnClusterer()
        d = cluster.data(data)
        #tags = cluster.cluster(d, trans_pars, method, method_pars)
        d["tags"] = cluster.cluster(d, trans_pars, study["method"], method_pars)
        return d
    else:
        return


def main():
    parser = argparse.ArgumentParser(description="Evaluate clustering method")
    parser.add_argument("--data", type=str, required=True, help="Dataset")
    parser.add_argument("--study", type=str, required=True, help="Clustering method")

    args = parser.parse_args()
    data = load_data(args.data)
    study = misc_util.open_bundle(args.study)

    run(data, study)


if __name__ == "__main__":
    main()
