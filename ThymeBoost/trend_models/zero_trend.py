# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
import numpy as np
import pandas as pd

class ZeroModel(TrendBaseModel):
    """A utility trend method. If you do not want to center or detrend your data before approximating seasonality then it is useful.
    """
    model = 'zero'
    
    def __init__(self):
        self.model_params = None
        self.fitted = None

    def __str__(self):
        return f'{self.model}()'

    def fit(self, y, **kwargs):
        """
        A utility trend that simply returns 0

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
        self.fitted = np.zeros(len(y))
        self.model_params = 0
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        return np.zeros(forecast_horizon)