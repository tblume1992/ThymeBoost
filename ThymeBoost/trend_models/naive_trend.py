# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
import numpy as np


class NaiveModel(TrendBaseModel):
    model = 'naive'
    
    def __init__(self):
        self.model_params = None
        self.fitted = None

    def __str__(self):
        return f'{self.model}()'

    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop for a naive/random-walk model AKA the last value.

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
        self.model_params = y[-1]
        y = np.append(np.array(y[0]), y)
        y = y[:-1]
        self.fitted = y
        return self.fitted
    
    def predict(self, forecast_horizon, model_params):
        return np.tile(model_params, forecast_horizon)
    

