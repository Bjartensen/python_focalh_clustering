"""
Script that does:
    Run set of clustering methods.
    On set of data defined in yaml.
    Run hyperparameter optimzation with Optuna.
"""

import yaml
import argparse
from typing import Any
import numpy as np
from sklearn.utils import shuffle
import ROOT
import optuna
from lib.modified_aggregation import ModifiedAggregation
import lib.base_nn as BNN
import lib.unet_nn as UNet
from lib.train import Train
from lib import efficiency, coverage, vmeas,               count_clusters,count_labels
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Need data function
# Need Optuna objective function -- generalized?
# DIFFERENT FILE?

# Early stopping
# https://github.com/optuna/optuna/issues/1001#issuecomment-596478792

CONFIG = "../analysis/optimize.yaml"

def run(config, method, its):
    analysis_type = config["analysis"]["type"]
    print(f"Running analysis {analysis_type}")
    print(f"\tOn {method}")
    print(f"\tFor {its} iterations.")

    handle_method(config, method, its)

    # Maybe it should be data.yaml and methods.yaml

def handle_method(config: Any, method_name: str, its: int):
    """
    Function to handle type of method.
    I decide here what the interface should be for all the methods.
    Hardcode for now but this could also be define in files.
    """
    match method_name.lower():
        case "ma":
            print(f"Optimizing {method_name}")
            study = ma_optimize(config, its)
            print("Best parameters:", study.best_trial.params)
            # Save study
        case "cnn":
            print(f"Optimizing {method_name}")
            study = cnn_optimize(config, its)
            print("Best parameters:", study.best_trial.params)
            # Save study
        case _:
            raise ValueError(f"Unknown method: {method_name}")


# Any clustering method need only return labels
# Rest can be deduced from this. (???)
# Then helper functions for labels and clusters to compute efficiency etc

def p2_data(config: Any):
    # Return labels as well?
    files = config["analysis"]["files"]
    length = len(files)
    l_npval = []
    l_npmajorlab = []

    for file in files:
        npval,npmajorlab = read_generic_tfile(file)
        l_npval.append(npval)
        l_npmajorlab.append(npmajorlab)

    l_npval = np.concatenate(l_npval)
    l_npmajorlab = np.concatenate(l_npmajorlab)

    return l_npval, l_npmajorlab

def read_generic_tfile(file):
    tfile = ROOT.TFile(file["path"], "READ")
    ttree = tfile.Get("EventsTree")
    Nentries = ttree.GetEntries()

    dataloader = BNN.Data()

    # Hardcoded for prototype 2 (and CAEN for saturation)
    FOCAL2_CELLS = 249
    FOCAL2_SAT = 4096

    npval = np.zeros(Nentries*FOCAL2_CELLS, dtype=np.float32).reshape(Nentries, FOCAL2_CELLS)
    npmajorlab = np.zeros(Nentries*FOCAL2_CELLS, dtype=np.int32).reshape(Nentries, FOCAL2_CELLS)

    num_particles = int(file["particles"])

    for i in range(Nentries):
        ttree.GetEntry(i)
        npval[i] = np.array(ttree.value).clip(max=FOCAL2_SAT)

        l = np.array(ttree.labels)
        f = np.array(ttree.fractions)
        npmajorlab[i] = dataloader.get_major_labels(l,f,num_particles)

    tfile.Close()
    return npval,npmajorlab


def p2_data_img(config: Any):
    """
    Converting generic events to images for the CNN.
    """
    files = config["analysis"]["files"]
    Nfiles = len(files)
    event_list = []
    target_list = []
    count_list = []
    mapping_list = []
    dlabel_list = []
    dataloader = BNN.Data()
    for file in files:
        tfile = ROOT.TFile(file["path"], "READ")
        ttree = tfile.Get("EventsTree")
        print(file)
        data = dataloader.to_training_tensor(ttree)
        event_list.append(data["event"])
        target_list.append(data["target"])
        count_list.append(data["count"])
        mapping_list.append(data["mapping"])
        dlabel_list.append(data["dlabels"])


    events = torch.cat(event_list)
    targets = torch.cat(target_list)
    counts = torch.cat(count_list)
    mapping = torch.cat(mapping_list)
    dlabels = torch.cat(dlabel_list)

    return events, targets, counts, mapping, dlabels


# TO-DO: Move to separate file
def ma_optimize(config: Any, its: int):
    """
    Optimizing modified aggregation.
    """
    print(f"Optimizing modified aggregation for {its} iterations.")

    # Could be part of yaml
    adj = np.load("p2_adj.npy")
    idx = np.load("p2_cell_idx.npy")
    iadj = np.load("p2_sim_adj_map.npy")

    # Get data
    values, labels = p2_data(config)
    # Maybe a randomize
    values, labels = shuffle(values, labels)

    clusters = np.zeros_like(labels)

    def objective(trial):
        seed = trial.suggest_float("seed", 0, 4096)
        agg = trial.suggest_float("agg", 0, 4096)
        if agg>=seed: return float("inf")

        # Different function?
        ma = ModifiedAggregation(seed=seed, agg=agg)

        # Metric should be an input parameter
        eff = np.zeros(len(values), dtype=np.float32)
        cov = np.zeros(len(values), dtype=np.float32)
        vm = np.zeros(len(values), dtype=np.float32)
        for i in range(len(values)):
            clusters[i],_ = ma.run(adj, values[i][iadj])
            eff[i] = efficiency(clusters[i], labels[i])
            cov[i] = coverage(clusters[i], labels[i], values[i])
            vm[i] = vmeas(clusters[i], labels[i])

        return (vm.mean()-1)**2

    study = optuna.create_study()
    study.optimize(objective, n_trials=its)
    return study


def cnn_optimize(config: Any, its: int):
    """
    Optimizing CNN.
    """
    print(f"Optimizing cnn for {its} iterations.")

    # Get data
    events, targets, counts, mapping, dlabels = p2_data_img(config)
    event_train, event_eval, \
    target_train, target_eval, \
    count_train, count_eval, \
    mapping_train, mapping_eval, \
    dlabels_train, dlabels_eval \
    = train_test_split(events, targets, counts, mapping, dlabels, test_size=0.4)

    print("eval:",event_eval.shape)
    print("target:",target_eval.shape)
    print("count:",count_eval.shape)
    print("mapping:",mapping_eval.shape)
    print("dlabels:",dlabels_eval.shape)

    adj = np.load("p2_image_adj_21x21.npy")

    dataloader = BNN.Data()

    def objective(trial):
        u = UNet.UNet()
        image_criterion = nn.MSELoss() # Could also be hyperparameter
        seed = trial.suggest_float("seed", 0.01, 1)
        agg = trial.suggest_float("agg", 0, 1)
        if agg>=seed: return float("inf")

        lr = trial.suggest_float("lr", 0.01,1) #0.21
        #lr = trial.suggest_float("lr", 0.21,0.21) #0.21
        momentum = trial.suggest_float("momentum", 0.01,1) #0.98
        #momentum = trial.suggest_float("momentum", 0.98,0.98) #0.98
        epochs = trial.suggest_int("epochs", 10, 50) #1000

        trainer = Train(model=u, image_crit=image_criterion, learning_rate=lr, momentum=momentum)
        trainer.run(epochs, event_train, target_train)

        # Evaluate
        x = u(event_eval)
        torch.save(event_eval, "eval_"+str(trial.number)+".pt")
        torch.save(x, "x_"+str(trial.number)+".pt")
        torch.save(count_eval, "count_"+str(trial.number)+".pt")

        # ModifiedAggregation
        # Do the max stuff?
        ma = ModifiedAggregation(seed=seed, agg=agg)
        eff = np.zeros(len(x), dtype=np.float32)
        for i in range(len(x)):
            vals = x[i][0].flatten().detach().numpy()
            clusters,_ = ma.run(adj, vals)
            lab = dataloader.invert_labels(clusters, mapping_eval[i][0].detach().numpy(), vals, dlabels_eval[i][0].shape[0])
            # Should be able to map new clusters to old labels now
            #eff[i] = float(1) / float(count_eval[i])
            true_labels = dlabels_eval[i][0].detach().numpy()
            #print(count_clusters(clusters), count_labels(dlabels_eval[i][0]))
            eff[i] = efficiency(clusters, true_labels)
        print("mu eff:",eff.mean())
        return (eff.mean()-1)**2


    study = optuna.create_study()
    study.optimize(objective, n_trials=its)

    return study



def sklearn_optimize():
    """
    I probably still want them separate, but they will be almost identical.
    """
    pass



def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument("--config", type=str, required=True, help="YAML file path")
    parser.add_argument("--method", type=str, required=True, help="Clustering method")
    parser.add_argument("--its", type=int, required=True, help="Number of iterations")

    # Should have: data, method, iterations

    args = parser.parse_args()
    config = load_config(args.config)
    run(config, args.method, args.its)


if __name__ == "__main__":
    main()
