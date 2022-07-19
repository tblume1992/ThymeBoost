# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
import numpy as np


class MedianModel(TrendBaseModel):
    model = 'median'
    
    def __init__(self):
        self.model_params = None
        self.fitted = None
        self._online_steps = 0

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
        median_est = np.median(y)
        self.model_params = median_est
        self.fitted = np.tile(median_est, len(y))
        return self.fitted
    
    def predict(self, forecast_horizon, model_params):
        return np.tile(model_params, forecast_horizon)
    