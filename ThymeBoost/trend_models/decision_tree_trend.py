# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

class DecisionTreeModel(TrendBaseModel):
    model = 'decision_tree'
    
    def __init__(self):
        self.model_params = None
        self.fitted = None
        self._online_steps = 0

    def __str__(self):
        return f'{self.model}()'

    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop for an optimized theta model.

        Parameters
        ----------
        time_series : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.kwargs = kwargs
        bias = kwargs['bias']
        tree_depth = kwargs['tree_depth']
        # y -= bias
        X = np.array(range(len(y))).reshape((-1, 1))
        tree_model = DecisionTreeRegressor(max_depth=tree_depth)
        self.model_obj = tree_model.fit(X, y)
        self.fitted = self.model_obj.predict(X)# + bias
        last_fitted_values = self.fitted[-1]
        self.model_params = last_fitted_values
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        return np.tile(model_params, forecast_horizon)
