"""
Class that acts as the handler for Modified Aggregation.
Initialize an instance, either from parameters or hyperparameter optimization object.
Prepare data, and cluster. This will replace in part code in optimize.py
"""

from lib.modified_aggregation import ModifiedAggregation
import Optuna



class ModifiedAggregationClusterer:
    def __init__(self):
        pass

    @classmethod
    def from_study(cls, study):
        """
        Read Optuna study object and use parameters
        """
        pass


    def data(self):
        """
        Prepare data by reading the generic root files, read
        adjacency matrix and inverse index transform.
        Returning adj matrix and numpy array?
        Many tfiles? Using yaml file?
        """
        pass


