# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel

class ThetaModel(TrendBaseModel):
    model = 'theta'
    
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
        y -= bias
        theta_model = ThetaModel(y, method="additive", period=1) + bias
        fitted = theta_model.fit()
        self.fitted = theta_model
        last_fitted_values = self.fitted[-1]
        self.model_params = last_fitted_values
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        return np.tile(model_params, forecast_horizon)
