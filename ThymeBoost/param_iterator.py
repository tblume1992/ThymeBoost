# -*- coding: utf-8 -*-
"""
A base class which is inherited by both ensemble and optimize classes.
Used to clean large parameter lists of illegal combinations
"""
import numpy as np
import pandas as pd


class ParamIterator:
    """
    The ensemble/optimizer base class
    """

    def __init__(self):
        pass

    def _safe_check_in_list(self, value, value_list):
        """
        Safely check if a value is in a list, handling arrays/DataFrames.

        Parameters
        ----------
        value : str
            The value to search for
        value_list : list
            List of values that may contain arrays or DataFrames

        Returns
        -------
        bool
            True if value is in the list
        """
        for item in value_list:
            try:
                if item == value:
                    return True
            except (ValueError, TypeError):
                # Skip items that can't be compared (arrays, DataFrames, etc.)
                continue
        return False

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
        if not self._safe_check_in_list('ewm', v) and 'ewm_alpha' in k:
            params['ewm_alpha'] = None
        if (not self._safe_check_in_list('ses', v) and not self._safe_check_in_list('des', v) and
            not self._safe_check_in_list('damped_des', v) and not self._safe_check_in_list('croston', v)) and \
           ('alpha' in k):
            params['alpha'] = None
        if (not self._safe_check_in_list('des', v) and not self._safe_check_in_list('damped_des', v)) and \
           ('beta' in k):
            params['beta'] = None
        if not self._safe_check_in_list('linear', v) and 'trend_weights' in k:
            params['trend_weights'] = None
        if not self._safe_check_in_list('linear', v) and 'l2' in k:
            params['l2'] = None
        if (not self._safe_check_in_list('linear', v) and not self._safe_check_in_list('ransac', v) and
            not self._safe_check_in_list('loess', v)) and 'poly' in k:
            params['poly'] = None
        # if 'loess' not in v and 'window_size' in k and 'moving_average' not in v and 'window_size' in k:
        #     params['window_size'] = None
        if not self._safe_check_in_list('fourier', v) and 'fourier_order' in k:
            params['fourier_order'] = None
        if 'arima' not in str(v) and 'arima_order' in k:
            params['arima_order'] = None
        if not self._safe_check_in_list('decision_tree', v) and 'tree_depth' in k:
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
