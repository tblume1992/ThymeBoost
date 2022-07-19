# -*- coding: utf-8 -*-
import numpy as np
import statsmodels as sm
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
from pmdarima.arima import auto_arima

class ArimaModel(TrendBaseModel):
    """ARIMA Model from Statsmodels"""
    model = 'arima'

    def __init__(self):
        self.model_params = None
        self.fitted = None
        self._online_steps = 0

    def __str__(self):
        return f'{self.model}({self.kwargs["arima_order"]})'

    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop for a arima model.

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
        self.order = kwargs['arima_order']
        self.arima_trend = kwargs['arima_trend']
        bias = kwargs['bias']
        if self.order == 'auto':
            ar_model = auto_arima(y,
                                  seasonal=False,
                                  error_action='warn',
                                  trace=False,
                                  supress_warnings=True,
                                  stepwise=True,
                                  random_state=20,
                                  n_fits=50)
            self.fitted = ar_model.predict_in_sample()
        else:
            ar_model = sm.tsa.arima.model.ARIMA(y - bias,
                                                order=self.order,
                                                trend=self.arima_trend).fit()
            self.fitted = ar_model.predict(start=0, end=len(y) - 1) + bias
        self.model_params = (ar_model, bias, len(y))
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        last_point = model_params[2] + forecast_horizon
        if self.order == 'auto':
            prediction = model_params[0].predict(n_periods=forecast_horizon)
        else:
            prediction = model_params[0].predict(start=model_params[2] + 1, end=last_point) + \
                         model_params[1]
        return prediction
