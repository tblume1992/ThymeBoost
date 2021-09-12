# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
import numpy as np
import pandas as pd

class EwmModel(TrendBaseModel):
    model = 'ewm'
    
    def __init__(self):
        self.model_params = None
        self.fitted = None

    def __str__(self):
        return f'{self.model}({self.kwargs["ewm_alpha"]})'

    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop for a ewm model using alpha.

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
        alpha = kwargs['ewm_alpha']
        bias = kwargs['bias']
        y = pd.Series(y - bias)
        self.fitted = np.array(y.ewm(alpha=alpha).mean()) + bias
        last_fitted_values = self.fitted[-1]
        self.model_params = last_fitted_values
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        return np.tile(model_params, forecast_horizon)
