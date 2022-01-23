# -*- coding: utf-8 -*-
"""
A base class which is inherited by both ensemble and optimize classes. 
Used to clean large parameter lists of illegal combinations
"""
import numpy as np


class ParamIterator:
    """
    The ensemble/optimizer base class
    """

    def __init__(self):
        pass

    def param_check(self, params):
        """
        Given a dict of params, check for illegal combinations

        Parameters
        ----------
        params : dict
            A dictionary of params for one single thymeboost model.

        Returns
        -------
        params : dict
            A dictionary with illegal values nullified.

        """
        v = list(params.values())
        k = list(params.keys())
        exo = False
        if 'exogenous' in k:
            exogenous = params['exogenous']
            params.pop('exogenous', None)
            v = list(params.values())
            k = list(params.keys())
        else:
            exogenous = None
        if 'ewm' not in v and 'ewm_alpha' in k:
            params['ewm_alpha'] = None
        if ('ses' not in v and 'des' not in v and 'damped_des' not in v) and \
           ('alpha' in k):
            params['alpha'] = None
        if ('des' not in v and 'damped_des' not in v) and \
           ('beta' in k):
            params['beta'] = None
        if 'linear' not in v and 'trend_weights' in k:
            params['trend_weights'] = None
        if 'linear' not in v and 'l2' in k:
            params['l2'] = None
        if ('linear' not in v and 'ransac' not in v and 'loess' not in v) and 'poly' in k:
            params['poly'] = None
        if 'loess' not in v and 'window_size' in k:
            params['window_size'] = None
        if 'fourier' not in v and 'fourier_order' in k:
            params['fourier_order'] = None
        if 'arima' not in str(v) and 'arima_order' in k:
            params['arima_order'] = None
        if 'decision_tree' not in v and 'tree_depth' in k:
            params['tree_depth'] = None
        # if 'local' in v and ('loess' in v or 'ewm' in v or 'ses' in v or 'des'
        #                      in v or 'damped_des' in v or 'arima' in v):
        #     params['fit_type'] = 'global'
        params['exogenous'] = exogenous
        return params



    def sanitize_params(self, param_list):
        """
        Iterate through param dicts to sanitize illegal combinations.

        Parameters
        ----------
        param_list : list
            A List of param dicts.

        Returns
        -------
        list
            List of cleaned param dicts.

        """
        cleaned = [self.param_check(i) for i in param_list]
        #drop duplicate settings breaks with arrays from seasonality_weights
        #return [i for n, i in enumerate(cleaned) if i not in cleaned[n + 1:]]
        return cleaned
