import numpy as np
from sklearn.metrics import v_measure_score, silhouette_score



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
    eff = 0
    try:
        eff = float(count(clusters)) / float(count(labels))
    except ZeroDivisionError:
        eff = 0
    return eff

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
        cl_avg = total(clusters,values) / count(clusters)
        lab_avg = total(labels,values) / count(labels)
        # Check for NaN or inf in the result
        if np.isnan(cl_avg) or np.isnan(lab_avg) or np.isinf(cl_avg) or np.isinf(lab_avg):
            return 0
    except ValueError:
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

def compute_score(tags, labels, values, score):
    """
    Function to handle which score to compute.
    Could also be handled in part by yaml (lol).
    """
    scores = np.zeros(len(values))
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
        return np.ones(len(values))*-1

