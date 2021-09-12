# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
import numpy as np


class MeanModel(TrendBaseModel):
    model = 'mean'
    
    def __init__(self):
        self.model_params = None
        self.fitted = None

    def __str__(self):
        return f'{self.model}()'

    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop for a mean model.

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
        mean_est = np.mean(y)
        self.model_params = mean_est
        self.fitted = np.tile(mean_est, len(y)) 
        return self.fitted
    
    def predict(self, forecast_horizon, model_params):
        return np.tile(model_params, forecast_horizon)
    

        