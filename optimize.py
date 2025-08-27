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
import lib.base_nn as BNN
import lib.unet_nn as UNet
from lib.train import Train
from lib import efficiency, coverage, vmeas,               count_clusters,count_labels
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

    # Save study and extra parameters
    with open(dir+filename, "wb") as f:
        pickle.dump(bundle, f)

    if model != None:
        print("Saving model")
        # Save model in case of CNN
        bundle["model_file"] = filename_model
        torch.save(model, dir+filename_model)


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
            save_study(study, data, its, method_name, model)
            print("Saved.")

        case "dbscan":
            print(f"Optimizing {method_name}")
            sklearn_optimize(data, method, its)

        case "hdbscan":
            print(f"Optimizing {method_name}")
            sklearn_optimize(data, method, its)

        case "baygauss":
            print(f"Optimizing {method_name}")
            sklearn_optimize(data, method, its)

        case _:
            raise ValueError(f"Unknown method: {method_name}")


# Any clustering method need only return labels
# Rest can be deduced from this. (???)
# Then helper functions for labels and clusters to compute efficiency etc

def p2_data(data: Any):
    # Return labels as well?
    files = data["files"]
    length = len(files)
    l_npy = []
    l_npx = []
    l_npval = []
    l_npmajorlab = []

    for file in files:
        npx,npy,npval,npmajorlab = read_generic_tfile(file)
        l_npx.append(npx)
        l_npy.append(npy)
        l_npval.append(npval)
        l_npmajorlab.append(npmajorlab)

    arr_npx = np.concatenate(l_npx)
    arr_npy = np.concatenate(l_npy)
    arr_npval = np.concatenate(l_npval)
    arr_npmajorlab = np.concatenate(l_npmajorlab)

    return arr_npx, arr_npy, arr_npval, arr_npmajorlab

def read_generic_tfile(file):
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
    npmajorlab = np.zeros(Nentries*FOCAL2_CELLS, dtype=np.int32).reshape(Nentries, FOCAL2_CELLS)

    num_particles = int(file["particles"])

    for i in range(Nentries):
        ttree.GetEntry(i)
        npx[i] = np.array(ttree.x)
        npy[i] = np.array(ttree.y)
        npval[i] = np.array(ttree.value).clip(max=FOCAL2_SAT)

        l = np.array(ttree.labels)
        f = np.array(ttree.fractions)
        npmajorlab[i] = dataloader.get_major_labels(l,f,num_particles)

    tfile.Close()
    return npx,npy,npval,npmajorlab


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
# ma_opt.py e.g.
# then ma_exec for using a trained/optimized model?
def ma_optimize(data: Any, method: Any, its: int):
    """
    Optimizing modified aggregation.
    """
    print(f"Optimizing modified aggregation for {its} iterations.")

    # Could be part of yaml
    adj = np.load("p2_adj.npy")
    idx = np.load("p2_cell_idx.npy")
    #iadj = np.load("p2_sim_adj_map.npy")
    iadj = np.load("p2_sim_adj_map2.npy")

    # Get data
    _,_,values, labels = p2_data(data)
    #_,_,values, labels = p2_data(data)
    # Maybe a randomize
    values, labels = shuffle(values, labels)

    clusters = np.zeros_like(labels)


    def objective(trial):
        pars = []
        for i,par in enumerate(method["parameters"]):
            if par["type"] == "float":
                pars.append(trial.suggest_float(par['name'], float(par['min']), float(par['max'])))
            # No other types


        if pars[1]>=pars[0]: return float("inf")

        # Different function?
        ma = ModifiedAggregation(*pars)

        # Metric should be an input parameter defined in yaml
        # Use "score" as generalized name
        eff = np.zeros(len(values), dtype=np.float32)
        cov = np.zeros(len(values), dtype=np.float32)
        vm = np.zeros(len(values), dtype=np.float32)
        for i in range(len(values)):
            # Technically, all the labels need to be mapped as well
            clusters[i],_ = ma.run(adj, values[i][iadj])
            eff[i] = efficiency(clusters[i], labels[i][iadj])
            cov[i] = coverage(clusters[i][iadj], labels[i], values[i][iadj]) # wait this should be mapped??
            vm[i] = vmeas(clusters[i], labels[i][iadj])

        return (eff.mean()-1)**2

    study = optuna.create_study()
    study.optimize(objective, n_trials=its)
    return study


def cnn_optimize(data: Any, its: int):
    """
    Optimizing CNN.
    """

    # Get data
    events, targets, counts, mapping, dlabels = p2_data_img(data)
    event_train, event_eval, \
    target_train, target_eval, \
    count_train, count_eval, \
    mapping_train, mapping_eval, \
    dlabels_train, dlabels_eval \
    = train_test_split(events, targets, counts, mapping, dlabels, test_size=0.4)

    adj = np.load("p2_image_adj_21x21.npy")

    dataloader = BNN.Data()

    # Store models
    models = []

    def objective(trial):

        image_criterion = nn.MSELoss() # Could also be hyperparameter

        # Hardcode instead. And then maybe always rescale?
        seed = trial.suggest_float("seed", 0.5, 0.5)
        agg = trial.suggest_float("agg", 0.0, 0.0)
        u = UNet.UNet()

        if agg>=seed:
            models.append(copy.deepcopy(u))
            return float("inf")


        lr = trial.suggest_float("lr", 0.1,1.) #0.21
        momentum = trial.suggest_float("momentum", 0.1,1.) #0.98
        epochs = trial.suggest_int("epochs", 10, 100) #1000

        trainer = Train(model=u, image_crit=image_criterion, learning_rate=lr, momentum=momentum)
        trainer.run(epochs, event_train, target_train)

        models.append(copy.deepcopy(u))

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

        eval_metric = (eff.mean()-1)**2
        return eval_metric


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


def get_model_memory_usage(model):
    # Calculate memory usage in bytes
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_size + buffer_size  # in bytes


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
    x,y,z,dlab = p2_data(data)
    method_name = method["name"]

    dataloader = BNN.Data()
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
            case _:
                raise ValueError(f"Unknown method: {method_name}")

        """
        Cluster and evaluate
        """
        print(f"Clustering with {method_name}...")
        if method["labels"] == True:
            for i in range(len(X)):
                model.fit(X[i])
                Y[i] = model.labels_.astype(int)
                Y[i] += Y[i]+1 # Not clustered is 0 in my clustering methods
                cl[i] = dataloader.kdtree_map(X[i], np.column_stack([x[i], y[i]]), Y[i])
        else:
            for i in range(len(X)):
                model.fit(X[i])
                Y[i] = model.predict(X[i])
                Y[i] += Y[i]+1 # Not clustered is 0 in my clustering methods
                cl[i] = dataloader.kdtree_map(X[i], np.column_stack([x[i], y[i]]), Y[i])

        score = np.zeros(N)
        for i in range(N):
            score[i] = vmeas(cl[i], dlab[i])
        print(f"mean of score: {score.mean()}")

        print("Done.")
        print(f"{len(X)}, {len(Y)}")
        fig,ax=plt.subplots(nrows=2, ncols=2, figsize=(10,10))
        fig.suptitle(f"{score.mean()}")
        for a in ax.flatten():
            a.set_xlim(-10,10)
            a.set_ylim(-10,10)
        test_idx = 55

        print(cl[i])

        ax[0][0].scatter(x[test_idx], y[test_idx], s=50*z[test_idx]/z[test_idx].max())
        ax[0][1].scatter(X[test_idx][:,0], X[test_idx][:,1], marker=".")

        print(len([Y[test_idx]]))
        for l in set(Y[test_idx]):
            if l == 0:
                continue
            mask = Y[test_idx] == l
            ax[1][0].scatter(X[test_idx][mask][:,0], X[test_idx][mask][:,1], marker=".")

        for l in set(cl[test_idx]):
            if l == 0:
                continue
            mask = cl[test_idx] == l
            ax[1][1].scatter(x[test_idx][mask], y[test_idx][mask], marker="s")

        fig.savefig(f"trans_test_{trial.number}.png")

        return (score.mean() - 1)**2


    study = optuna.create_study()
    study.optimize(objective, n_trials=its)

    # Same evaluation

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
