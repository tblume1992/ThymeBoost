"""
A class to propose splits according to the most and least interesting points 
based on the gradient.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats


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
    #TODO: add other strategies
    def __init__(self,
                 given_splits,
                 exclude_splits,
                 min_sample_pct,
                 n_split_proposals,
                 split_strategy,
                 approximate_splits):
        self.given_splits = given_splits
        self.exclude_splits = exclude_splits
        self.min_sample_pct = min_sample_pct
        self.n_split_proposals = n_split_proposals
        self.approximate_splits = approximate_splits
        self.split_strategy = split_strategy

    def gradient_based_proposal(self):
        gradient = pd.Series(np.abs(np.gradient(self.time_series, edge_order=1)))
        gradient = gradient.sort_values(ascending=False)[:self.n_split_proposals]
        proposals = list(gradient.index)
        return proposals

    def histogram_based_proposal(self):
        n = len(self.time_series)
        iqr = stats.iqr(self.time_series)
        n_bins = (2 * iqr) / (n**(1/3)) + 1
        # self.n_split_proposals = n_bins
        proposals = list(np.arange(int(n / n_bins), n, int(n / n_bins)))
        return proposals

    def set_proposal_method(self):
        if self.split_strategy == 'gradient':
            proposal_obj = self.gradient_based_proposal
        elif self.split_strategy == 'histogram':
            proposal_obj = self.histogram_based_proposal
        return proposal_obj

    def get_approximate_split_proposals(self):
        """
        Propose indices to split the data on.

        Returns
        -------
        proposals : list
            The proposals to try out.

        """
        proposals_method = self.set_proposal_method()
        proposals = proposals_method()
        min_split_idx = int(max(3, len(self.time_series) * self.min_sample_pct))
        proposals = [i for i in proposals if i > min_split_idx]
        proposals = [i for i in proposals if i < len(self.time_series) - min_split_idx]
        proposals = [i for i in proposals if i not in self.exclude_splits]
        if not proposals:
            proposals = list(range(min_split_idx, len(self.time_series) - min_split_idx))
        return proposals

    def get_split_proposals(self, time_series):
        """
        Get proposals based on simple gradient strategy or histograms.

        Parameters
        ----------
        time_series : np.array
            The time series.

        Returns
        -------
        list
            A list of split indices to try.

        """
        self.time_series = time_series
        if self.approximate_splits:
            return self.get_approximate_split_proposals()
        elif self.given_splits:
            return self.given_splits
        else:
            min_idx = int(max(5, len(self.time_series) * self.min_sample_pct))
            return list(range(min_idx, len(self.time_series) - min_idx))



