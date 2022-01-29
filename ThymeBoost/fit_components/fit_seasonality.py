# -*- coding: utf-8 -*-
import numpy as np
from ThymeBoost.seasonality_models import (classic_seasonality, fourier_seasonality,
                                           naive_seasonality)


class FitSeasonality:
    """Approximates the seasonal component
    """

    def __init__(self, seasonal_estimator,
                 seasonal_period,
                 seasonality_lr,
                 seasonality_weights,
                 additive,
                 normalize_seasonality,
                 **kwargs):
        self.seasonal_estimator = seasonal_estimator
        self.seasonal_period = seasonal_period
        self.seasonality_lr = seasonality_lr
        self.additive = additive
        self.seasonality_weights = seasonality_weights
        self.normalize_seasonality = normalize_seasonality
        self.kwargs = kwargs

    @staticmethod
    def set_estimator(seasonal_estimator):
        if seasonal_estimator == 'fourier':
            seasonal_obj = fourier_seasonality.FourierSeasonalityModel
        elif seasonal_estimator == 'classic':
            seasonal_obj = classic_seasonality.ClassicSeasonalityModel
        elif seasonal_estimator == 'naive':
            seasonal_obj = naive_seasonality.NaiveSeasonalityModel
        else:
            raise NotImplementedError('That seasonal estimation is not availale yet, add it to the road map!')
        return seasonal_obj

    def fit_seasonal_component(self, detrended):
        data_len = len(detrended)
        if not self.seasonal_period:
            seasonality = np.zeros(data_len)
            self.model_params = None
            self.model_obj = None
        else:
            seasonal_class = FitSeasonality.set_estimator(self.seasonal_estimator)
            self.model_obj = seasonal_class(self.seasonal_period,
                                            self.normalize_seasonality,
                                            self.seasonality_weights)
            seasonality = self.model_obj.fit(detrended,
                                             seasonality_lr=self.seasonality_lr,
                                             **self.kwargs)
            self.model_params = self.model_obj.model_params
        return seasonality
