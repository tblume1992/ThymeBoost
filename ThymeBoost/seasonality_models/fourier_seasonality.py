# -*- coding: utf-8 -*-
import numpy as np
from ThymeBoost.seasonality_models.seasonality_base_class import SeasonalityBaseModel


class FourierSeasonalityModel(SeasonalityBaseModel):
    """
    Seasonality for naive decomposition method.
    """
    model = 'fourier'

    def __init__(self,
                 seasonal_period,
                 normalize_seasonality,
                 seasonality_weights):
        self.seasonal_period = seasonal_period
        self.normalize_seasonality = normalize_seasonality
        self.seasonality_weights = seasonality_weights
        self.seasonality = None
        self.model_params = None
        return

    def __str__(self):
        return f'{self.model}({self.kwargs["fourier_order"]}, {self.seasonality_weights is not None})'

    def handle_seasonal_weights(self, y):
        if self.seasonality_weights is None:
            seasonality_weights = self.seasonality_weights
        elif self.seasonality_weights == 'regularize':
            seasonality_weights = 1/(0.0001 + y**2)
        elif self.seasonality_weights == 'explode':
            seasonality_weights = (y**2)
        elif callable(self.seasonality_weights):
            seasonality_weights = self.seasonality_weights(y)
        else:
            seasonality_weights = self.seasonality_weights
        return seasonality_weights

    def get_fourier_series(self, t, fourier_order):
        x = 2 * np.pi * (np.arange(1, fourier_order + 1) /
                         self.seasonal_period)
        x = x * t[:, None]
        fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
        return fourier_series

    def fit(self, y, **kwargs):
        """
        Fit the seasonal component for fourier basis function method in the boosting loop.

        Parameters
        ----------
        y : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.kwargs = kwargs
        fourier_order = kwargs['fourier_order']
        seasonality_weights = self.handle_seasonal_weights(y)
        X = self.get_fourier_series(np.arange(len(y)), fourier_order)
        if seasonality_weights is not None:
            weighted_X_T = X.T @ np.diag(seasonality_weights)
            beta = np.linalg.pinv(weighted_X_T.dot(X)).dot(weighted_X_T.dot(y))
        else:
            beta = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
        self.seasonality = X @ beta
#       If normalize_seasonality we call normalize function from base class
        if self.normalize_seasonality:
            self.seasonality = self.normalize()
        self.seasonality = self.seasonality * kwargs['seasonality_lr']
        single_season = self.seasonality[:self.seasonal_period]
        future_seasonality = np.resize(single_season, len(y) + self.seasonal_period)
        self.model_params = future_seasonality[-self.seasonal_period:]
        return self.seasonality

    def predict(self, forecast_horizon, model_params):
        return np.resize(model_params, forecast_horizon)
