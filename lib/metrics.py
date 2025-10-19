import numpy as np
from sklearn.metrics import v_measure_score, silhouette_score
import yaml
from scipy.optimize import linear_sum_assignment
import math as ma



"""
Metrics used when evaluating clustering.
Mostly they will depend on true vs predicted labels (cluster assignments).
And in some cases the adc/energy/intensity value contained in each.
"""

def count(labels):
    """
    Count labels. Ignoring label 0 is probably never gonna come into play.
    """
    num = list(set([lab for lab in labels if lab!=0]))
    return len(num)

# Retire
def count_clusters(clusters):
    """
    Count the number of clusters, ignoring cluster 0 meaning unassigned.
    """
    num = list(set([cl for cl in clusters if cl!=0]))
    return len(num)


# Retire
def count_labels(labels):
    """
    Count labels. Ignoring label 0 is probably never gonna come into play.
    """
    num = list(set([lab for lab in labels if lab!=0]))
    return len(num)


def efficiency(clusters, labels):
    """
    Compute the efficieny (clusters-per-particle)
    """
    eff = float(0)
    try:
        cnt_clusters = float(count(clusters))
        cnt_labels = float(count(labels))
        eff = cnt_clusters / cnt_labels
    except ZeroDivisionError:
        eff = float(0)
    return eff


FITS = "analysis/fits.yaml"

#def separation_efficiency_opt(tags, labels, values, energies):
def separation_efficiency_opt(d):
    d_ret = separation_efficiency(d, linearity_yaml="test", energy_resolution_yaml="test")
    return d_ret["separation_efficiency"][:,0].sum() / d_ret["separation_efficiency"][:,1].sum()

#def separation_efficiency(tags, labels, values, energies, linearity_yaml="test", energy_resolution_yaml="test"):
def separation_efficiency(d, linearity_yaml="test", energy_resolution_yaml="test"):
    """
    Compute separation efficiency (resolved showers) from
    linearity and energy resolution fits.
    """

    tags = d["tags"]
    labels = d["labels"]
    values = d["values"]
    energies = d["energy"]

    def load_fit(fit, source, type):
        with open(FITS, "r") as file:
            config = yaml.safe_load(file)
        return config[fit][source][type]

    linearity = load_fit(linearity_yaml, "mc", "linearity")
    lin_a = linearity["a"]
    lin_b = linearity["b"]

    energy_resolution = load_fit(energy_resolution_yaml, "mc", "energy_resolution")
    eres_a = energy_resolution["a"]
    eres_b = energy_resolution["b"]
    eres_c = energy_resolution["c"]

    Nevents = len(tags)

    efficiency = np.zeros(Nevents*2).reshape(Nevents,2)
    pairs = [None]*Nevents
    tag_coms = [None]*Nevents
    label_coms = [None]*Nevents
    matched_indices = [None]*Nevents
    for i in range(Nevents):
        td = dict()
        td["tags"] = tags[i]
        td["labels"] = labels[i]
        td["values"] = values[i]
        td["energy"] = energies[i]
        td["x"] = d["x"][i]
        td["y"] = d["y"][i]
        #res, max_len, p, tag_c, label_c = resolved(td, lin_a, lin_b, eres_a, eres_b, eres_c)
        res_d = resolved(td, lin_a, lin_b, eres_a, eres_b, eres_c)
        efficiency[i] = (res_d["resolved"], res_d["objects"])
        pairs[i] = res_d["energy_pairs"]
        tag_coms[i] = res_d["tag_coms"]
        label_coms[i] = res_d["label_coms"]
        matched_indices[i] = res_d["matched_idx"]


    d["separation_efficiency"] = efficiency
    d["energy_pairs"] = pairs
    d["tag_coms"] = tag_coms
    d["label_coms"] = label_coms
    d["matched_indices"] = matched_indices


    # Add matched list of reconstructed energies
    #return efficiency, pairs, tag_coms, label_coms
    return d


def match_labels(predicted_sum, true_sum):
    """
    Should be changed to match by center-of-mass distances
    """
    cost_matrix = np.abs(np.subtract.outer(true_sum, predicted_sum))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


def com(dim, values, thresh=0.75):
    mask = values > values.max()*thresh
    return (dim[mask]*values[mask]).sum() / values[mask].sum()

def get_coms(x, y, labels, values):
    coms = []
    #for l in set(labels):
    # TypeError
    for l in np.unique(labels):
        mask = labels == l
        if l == 0:
            continue
        xcom = com(x[mask], values[mask])
        ycom = com(y[mask], values[mask])
        coms.append((xcom,ycom))
    return coms

def match_com(d):
    """
    Match by computing center of masses and finding closest.
    """
    #iadj = np.load("p2_sim_adj_map2.npy")
    x = d["x"]
    y = d["y"]
    tags = d["tags"]
    labels = d["labels"]
    values = d["values"]

    tag_coms = get_coms(x, y, tags, values)
    label_coms = get_coms(x, y, labels, values)

    matched_tag = []
    matched_label = []

    # Very Baroque...
    counter = 0
    while len(matched_label) < len(label_coms):
        stop = False
        counter += 1
        if counter > 1000:
            print(f"Took {counter} iterations, probably something wrong!")
            break
        min_tag_idx = 0
        min_label_idx = 0
        min_dist = np.inf

        for i,l in enumerate(label_coms):
            if len(matched_tag) >= len(tag_coms):
                stop = True
                break
            if i in matched_label:
                continue
            for j,t in enumerate(tag_coms):
                if j in matched_tag:
                    continue
                dist = ma.sqrt((l[0]-t[0])**2 + (l[1]-t[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    min_tag_idx = j
                    min_label_idx = i
        if stop:
            break
        matched_tag.append(min_tag_idx)
        matched_label.append(min_label_idx)

    # Padding
    if len(label_coms) > len(tag_coms):
        for i,e in enumerate(label_coms):
            if i in matched_label:
                continue
            matched_tag.append(-1)
            matched_label.append(i)
    else:
        for j,e in enumerate(tag_coms):
            if j in matched_tag:
                continue
            matched_tag.append(j)
            matched_label.append(-1)


    d = dict()
    d["tag_coms"] = tag_coms
    d["label_coms"] = label_coms
    d["matched_tag"] = matched_tag
    d["matched_label"] = matched_label

    return d


#def resolved(tags, labels, values, E, lin_a, lin_b, eres_a, eres_b, eres_c):
def resolved(d, lin_a, lin_b, eres_a, eres_b, eres_c):
    """
    For some clustered event, compute whether they are resolved
    by looking at the energy confusion and energy resolution.
    This metric "scales" with energy.
    """

    tags = d["tags"]
    labels = d["labels"]
    values = d["values"]
    E = d["energy"]

    #if len(set(labels)) != len(E):
    #    print("Metrics::Warning::resolved suppressed labels")

    sigma_E = energy_resolution(E, eres_a, eres_b, eres_c)


    label_sums = compute_sums(labels,values)
    tag_sums = compute_sums(tags,values)
    E_true = reconstruct_energy(label_sums, lin_a, lin_b)
    E_pred = reconstruct_energy(tag_sums, lin_a, lin_b)
    rows,cols = match_labels(E_pred, E_true)
    #print(f"E: {E}, sigma_E: {sigma_E}, E_true: {E_true}, E_pred: {E_pred}")

    #di = dict()
    matched = match_com(d)

    resolved_num = 0
    max_length = max(len(E_pred), len(E_true))

    pairs = []
    for i,e in enumerate(matched["matched_label"]):
        if e == -1:
            pairs.append((0, E_pred[matched["matched_tag"][i]]))
            continue
        elif matched["matched_tag"][i] == -1:
            pairs.append((E_true[e],0))
            continue
        else:
            E_confusion = abs(E_true[e] - E_pred[matched["matched_tag"][i]])
            pairs.append((E_true[e], E_pred[matched["matched_tag"][i]]))
        if E_confusion < sigma_E[e]:
            resolved_num += 1

    ret_d = dict()
    ret_d["resolved"] = resolved_num
    ret_d["objects"] = len(pairs)
    ret_d["energy_pairs"] = pairs
    ret_d["tag_coms"] = matched["tag_coms"]
    ret_d["label_coms"] = matched["label_coms"]
    ret_d["matched_idx"] = list(zip(matched["matched_label"], matched["matched_tag"]))
    #return resolved, len(pairs), pairs, matched["tag_coms"], matched["label_coms"]
    return ret_d



def reconstruct_energy_from_characteristics(adc, linearity_yaml="test"):
    linearity = load_fit(linearity_yaml, "mc", "linearity")
    lin_a = linearity["a"]
    lin_b = linearity["b"]
    return reconstruct_energy(adc,lin_a,lin_b)


def reconstruct_energy(adc,a,b):
    """
    Reconstruct energy based on linerity fit.
    """
    return (adc-b)/a


def energy_resolution(E,a,b,c):
    """
    Compute energy resolution based on energy and energy resolution fit.
    """
    sigma_E = np.sqrt( (a/np.sqrt(E))**2 + (b/E)**2 + c**2)
    return sigma_E * E


def total(labels, values):
    mask = labels != 0
    return values[mask].sum().astype(float)

def compute_sums(label, values):
    sums = []
    #for l in set(label):
    for l in np.unique(label):
        if l == 0:
            continue
        mask = label == l
        sums.append(values[mask].sum())
    return np.array(sums)

def coverage(clusters, labels, values):
    """
    Compute the ratio of energy in clusters to energy in labels
    """
    cov = 0
    try:
        cov = total(clusters,values) / total(labels,values)
    except ZeroDivisionError:
        cov = 0

    return cov


def average_intensity_ratio(clusters, labels, values):
    try:
        #RuntimeWarning: invalid value encountered in scalar divide
        #cl_avg = total(clusters,values) / count(clusters)

        cl_avg = total(clusters,values) / count(clusters)
        lab_avg = total(labels,values) / count(labels)
        # Check for NaN or inf in the result
        if np.isnan(cl_avg) or np.isnan(lab_avg) or np.isinf(cl_avg) or np.isinf(lab_avg):
            return 0
    except ValueError:
        print("valueerror")
        return 0
    return cl_avg/lab_avg

def vmeas(clusters, labels):
    return v_measure_score(clusters, labels)

def vmeas_weighted(clusters, labels, values):
    """
    Implement the weighted form I developed earlier.
    Could be interesting to see how they correlate.
    """
    norm = values / values.max()
    vals = (100*norm).astype(int)
    cl = np.repeat(clusters, vals)
    lab = np.repeat(labels, vals)
    return v_measure_score(cl, lab)

def silh(clusters, labels):
    # Makes no sense...
    return silhouette_score(clusters, labels)

def average_energy(energy):
    """
    Simple function to convert a variable size list of
    energies to numpy with average.
    """
    en = np.zeros(len(energy))
    for i in range(len(energy)):
        en[i] = np.mean(energy[i])
    return en


def confusion(tags, labels, values):
    """
    Compute the confusion (E_pred - E_true)
    """
    pass

# Could add cross validation at some part of the metric calculation?
def compute_score_mean(d, score):
    if score in ["efficiency"
                 , "coverage"
                 , "average_intensity_ratio"
                 , "vmeasure"
                 , "vmeasure_weighted"
                 , "count_labels"
                 , "count_tags"]:
        return compute_score(d, score).mean()
    elif score == "separation":
        #return separation_efficiency_opt(d["tags"], d["labels"], d["values"], d["energy"])
        return separation_efficiency_opt(d)


def compute_score(d, score):
    """
    Function to handle which score to compute.
    Could also be handled in part by yaml (lol).
    """
    tags = d["tags"]
    labels = d["labels"]
    values = d["values"]
    scores = np.zeros(len(values), dtype=np.float32)
    if score == "efficiency":
        for i in range(len(scores)):
            scores[i] = efficiency(tags[i], labels[i])
        return scores
    elif score == "coverage":
        for i in range(len(scores)):
            scores[i] = coverage(tags[i], labels[i], values[i])
        return scores
    elif score == "average_intensity_ratio":
        for i in range(len(scores)):
            scores[i] = average_intensity_ratio(tags[i], labels[i], values[i])
        return scores
    elif score == "vmeasure":
        for i in range(len(scores)):
            scores[i] = vmeas(tags[i], labels[i])
        return scores
    elif score == "vmeasure_weighted":
        for i in range(len(scores)):
            scores[i] = vmeas_weighted(tags[i], labels[i], values[i])
        return scores
    elif score == "count_labels":
        for i in range(len(scores)):
            scores[i] = count(labels[i])
        return scores
    elif score == "count_tags":
        for i in range(len(scores)):
            scores[i] = count(tags[i])
        return scores
    else:
        raise ValueError(f"Metric {score} not recognized.")







# Working version for backup....
def resolved_backup(tags, labels, values, E, lin_a, lin_b, eres_a, eres_b, eres_c):
    """
    For some clustered event, compute whether they are resolved
    by looking at the energy confusion and energy resolution.
    This metric "scales" with energy.
    """

    if len(set(labels)) != len(E):
        print("Metrics::Warning::resolved suppressed labels")

    sigma_E = energy_resolution(E, eres_a, eres_b, eres_c)
    E_true = reconstruct_energy(compute_sums(labels,values), lin_a, lin_b)
    E_pred = reconstruct_energy(compute_sums(tags,values), lin_a, lin_b)
    rows,cols = match_labels(E_pred, E_true)
    #print(f"E: {E}, sigma_E: {sigma_E}, E_true: {E_true}, E_pred: {E_pred}")

    resolved = 0
    max_length = max(len(E_pred), len(E_true))

    pairs = np.zeros(len(E)*2).reshape(-1,2)

    # I need to keep track of matched particles

    # TO-DO: with arrays
    for i in range(len(rows)):
        pairs[i] = compute_sums(labels,values)[rows[i]], compute_sums(tags,values)[cols[i]]
        #pairs[i] = E_true[rows[i]], E_pred[cols[i]]
        E_confusion = abs(E_true[rows[i]] - E_pred[cols[i]])
        if E_confusion < sigma_E[i]:
            resolved += 1

    return resolved, max_length, pairs
