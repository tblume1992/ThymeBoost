# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel


class CrostonModel(TrendBaseModel):
    model = 'theta'
    
    def __init__(self):
        self.model_params = None
        self.fitted = None
        self._online_steps = 0

    def __str__(self):
        return f'{self.model}()'

    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop for the croston model. Stolen from Sktime:
        https://www.sktime.org/en/v0.8.0/api_reference/auto_generated/sktime.forecasting.croston.Croston.html
        Thank you Sktime!

        Parameters
        ----------
        time_series : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        fitted values.

        """
        self.kwargs = kwargs
        bias = kwargs['bias']
        y -= bias
        n_timepoints = len(y)  # Historical period: i.e the input array's length
        smoothing = kwargs['alpha']
        if smoothing is None:
            smoothing = .5

        # Fit the parameters: level(q), periodicity(a) and forecast(f)
        q, a, f = np.full((3, n_timepoints + 1), np.nan)
        p = 1  # periods since last demand observation

        # Initialization:
        first_occurrence = np.argmax(y[:n_timepoints] > 0)
        q[0] = y[first_occurrence]
        a[0] = 1 + first_occurrence
        f[0] = q[0] / a[0]

        # Create t+1 forecasts:
        for t in range(0, n_timepoints):
            if y[t] > 0:
                q[t + 1] = smoothing * y[t] + (1 - smoothing) * q[t]
                a[t + 1] = smoothing * p + (1 - smoothing) * a[t]
                f[t + 1] = q[t + 1] / a[t + 1]
                p = 1
            else:
                q[t + 1] = q[t]
                a[t + 1] = a[t]
                f[t + 1] = f[t]
                p += 1
        self.fitted = f[1:]
        last_fitted_values = self.fitted[-1]
        self.model_params = last_fitted_values
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        return np.tile(model_params, forecast_horizon)






