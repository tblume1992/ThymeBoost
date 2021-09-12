# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
import numpy as np
from scipy.signal import savgol_filter


class LoessModel(TrendBaseModel):
    model = 'loess'

    def __init__(self):
        self.model_params = None
        self.fitted = None

    def __str__(self):
        return f'{self.model}({self.kwargs["poly"], self.kwargs["window_size"]})'

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
        window_size = kwargs['window_size']
        bias = kwargs['bias']
        poly = kwargs['poly']
        self.fitted = savgol_filter(y - bias, window_size, poly) + bias
        slope = self.fitted[-1] - self.fitted[-2]
        last_value = self.fitted[-1]
        self.model_params = (slope, last_value)
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        last_fitted_value = model_params[1]
        slope = model_params[0]
        predicted = np.arange(1, forecast_horizon + 1) * slope + last_fitted_value
        return predicted
