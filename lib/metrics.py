import numpy as np
from sklearn.metrics import v_measure_score, silhouette_score



"""
Metrics used when evaluating clustering.
Mostly they will depend on true vs predicted labels (cluster assignments).
And in some cases the adc/energy/intensity value contained in each.
"""

def count_clusters(clusters):
    """
    Count the number of clusters, ignoring cluster 0 meaning unassigned.
    """
    num = list(set([cl for cl in clusters if cl!=0]))
    return len(num)


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
        eff = float(count_clusters(clusters)) / float(count_labels(labels))
    except ZeroDivisionError:
        eff = 0
    return eff



def clusters_sum(clusters, values):
    mask = clusters != 0
    return values[mask].sum().astype(float)

def labels_sum(labels, values):
    mask = labels != 0
    return values[mask].sum().astype(float)

def coverage(clusters, labels, values):
    """
    Compute the ratio of energy in clusters to energy in labels
    """
    cov = 0
    try:
        clusters_sum(clusters,values) / labels_sum(labels,values)
    except ZeroDivisionError:
        cov = 0

    return cov


def vmeas(clusters, labels):
    return v_measure_score(clusters, labels)

def vmeas_weighted(clusters, labels, values):
    """
    Implement the weighted form I developed earlier.
    Could be interesting to see how they correlate.
    """
    pass

def silh(clusters, labels):
    # Makes no sense...
    return silhouette_score(clusters, labels)



