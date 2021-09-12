# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing, Holt

class EtsModel(TrendBaseModel):
    model = 'ets'
    
    def __init__(self):
        self.model_params = None
        self.fitted = None

    def __str__(self):
        return f'{self.model}()'

    def simple_exponential_smoothing(self, y, bias, alpha):
        smoother = SimpleExpSmoothing(y - bias)
        fit_model = smoother.fit(smoothing_level=alpha)  
        fitted = fit_model.fittedvalues
        self.model_params = (fit_model, bias, len(y))
        return fitted

    def double_exponential_smoothing(self, y, bias, alpha, beta):
        smoother = Holt(y - bias)
        fit_model = smoother.fit(smoothing_level=alpha, smoothing_trend=beta)
        fitted = fit_model.fittedvalues
        self.model_params = (fit_model, bias, len(y))
        return fitted
    
    def damped_double_exponential_smoothing(self, y, bias, alpha, beta):
        smoother = Holt(y - bias, damped_trend=True)
        fit_model = smoother.fit(smoothing_level=alpha, 
                                 smoothing_trend=beta)
        fitted = fit_model.fittedvalues
        self.model_params = (fit_model, bias, len(y))
        return fitted
    
    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop for a ets model.

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
        self.model = kwargs['model']
        bias = kwargs['bias']
        if self.model == 'ses':
            self.fitted = self.simple_exponential_smoothing(y, bias, kwargs['alpha'])
        elif self.model == 'des':
            self.fitted = self.double_exponential_smoothing(y, bias, kwargs['alpha'], kwargs['beta'])
        elif self.model == 'damped_des':
            self.fitted = self.damped_double_exponential_smoothing(y, bias, kwargs['alpha'], kwargs['beta'])
        else:
            raise ValueError('That model type is not implemented!')
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        _start = model_params[2]
        _end = _start + forecast_horizon - 1
        prediction = model_params[0].predict(start=_start, end=_end) + model_params[1]
        return prediction
    
