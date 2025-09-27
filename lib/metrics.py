import numpy as np
from sklearn.metrics import v_measure_score, silhouette_score
import yaml
from scipy.optimize import linear_sum_assignment



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

def separation_efficiency_opt(tags, labels, values, energies):
    # This should do maybe just be np.repeat to get resolved/total
    # 1/2 and 5/10 gives each 0.5 and 0.5, the average of which is 0.5
    # but 6/11 is not 0.5
    # Or rather just do the sum and not element wise

    sep = separation_efficiency(tags, labels, values, energies, linearity_yaml="test", energy_resolution_yaml="test")

    #return sep[:,0] / sep[:,1]
    return sep[:,0].sum() / sep[:,1].sum()

def separation_efficiency(tags, labels, values, energies, linearity_yaml="test", energy_resolution_yaml="test"):
    """
    Compute separation efficiency (resolved showers) from
    linearity and energy resolution fits.

    """

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
    for i in range(Nevents):
        efficiency[i] = resolved(tags[i], labels[i], values[i], energies[i], lin_a, lin_b, eres_a, eres_b, eres_c)

    return efficiency


def match_labels(predicted_sum, true_sum):
    """
    Should be changed to match by center-of-mass distances
    """
    cost_matrix = np.abs(np.subtract.outer(true_sum, predicted_sum))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


# functio


def resolved(tags, labels, values, E, lin_a, lin_b, eres_a, eres_b, eres_c):
    """
    For some clustered event, compute whether they are resolved
    by looking at the energy confusion and energy resolution.
    This metric "scales" with energy.
    """


    sigma_E = energy_resolution(E, eres_a, eres_b, eres_c)
    E_true = reconstruct_energy(compute_sums(labels,values), lin_a, lin_b)
    E_pred = reconstruct_energy(compute_sums(tags,values), lin_a, lin_b)
    rows,cols = match_labels(E_pred, E_true)
    #print(f"E: {E}, sigma_E: {sigma_E}, E_true: {E_true}, E_pred: {E_pred}")

    resolved = 0
    max_length = max(len(E_pred), len(E_true))

    # TO-DO: with arrays
    for i in range(len(rows)):
        E_confusion = abs(E_true[rows[i]] - E_pred[cols[i]])
        if E_confusion < sigma_E[i]:
            resolved += 1
    return resolved, max_length


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


# Retire
def clusters_sum(clusters, values):
    mask = clusters != 0
    return values[mask].sum().astype(float)

# Retire
def labels_sum(labels, values):
    mask = labels != 0
    return values[mask].sum().astype(float)

def total(labels, values):
    mask = labels != 0
    return values[mask].sum().astype(float)

def compute_sums(label, values):
    sums = []
    for l in set(label):
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
        return separation_efficiency_opt(d["tags"], d["labels"], d["values"], d["energy"])


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

