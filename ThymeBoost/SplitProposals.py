# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 10:54:38 2020

@author: Tyler Blume
"""
import numpy as np
import pandas as pd

class SplitProposals:
    """
    Generate splits to try when fit_type = 'local'.

    Parameters
    ----------
    given_splits : list
        Splits to use when using fit_type='local'.
        
    exclude_splits : list
        exclude these index when considering splits for 
        fit_type='local'.  Must be idx not datetimeindex if 
        using a Pandas Series.
        
    min_sample_pct : float 
        Percentage of samples required to consider a split. 
        Must be 0<min_sample_pct<=1.
    
    n_split_proposals : int
        Number of split proposals based on the gradients.
                        
    approximate_splits : boolean
        Whether to use proposal splits based on gradients
        or exhaustively try splits with at least 
        min_sample_pct samples.
    
    Returns
    -------
    A list of splits to try.

    """
    def __init__(self, 
                 given_splits, 
                 exclude_splits, 
                 min_sample_pct,
                 n_split_proposals,
                 approximate_splits):
        self.given_splits = given_splits
        self.exclude_splits = exclude_splits
        self.min_sample_pct = min_sample_pct
        self.n_split_proposals = n_split_proposals
        self.approximate_splits = approximate_splits       
  
    def get_approximate_split_proposals(self):
        gradient = pd.Series(np.abs(np.gradient(self.time_series, edge_order=1)))
        gradient = gradient.sort_values(ascending = False)[:self.n_split_proposals]
        proposals = list(gradient.index)
        min_split_idx = int(max(3, len(self.time_series) * self.min_sample_pct))
        proposals = [i for i in proposals if i > min_split_idx]
        proposals = [i for i in proposals if i < len(self.time_series) - min_split_idx]
        proposals = [i for i in proposals if i not in self.exclude_splits]
        if len(proposals) < 1:
            proposals == list(range(min_split_idx, len(self.time_series) - min_split_idx))
            
        return proposals
    
    def get_split_proposals(self, time_series):
        self.time_series = time_series
        if self.approximate_splits:
            return self.get_approximate_split_proposals()
        elif self.given_splits:
            return self.given_splits
        else:
            min_idx =int(max(5, len(self.time_series) * self.min_sample_pct))            
            return list(range(min_idx, len(self.time_series) - min_idx))