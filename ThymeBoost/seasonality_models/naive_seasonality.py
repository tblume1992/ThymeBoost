# -*- coding: utf-8 -*-

import numpy as np
from ThymeBoost.seasonality_models.seasonality_base_class import SeasonalityBaseModel


class NaiveSeasonalityModel(SeasonalityBaseModel):
    """
    Seasonality for naive decomposition method.
    """
    model = 'naive'

    def __init__(self, seasonal_period, normalize_seasonality, seasonality_weights):
        self.seasonal_period = seasonal_period
        self.normalize_seasonality = normalize_seasonality
        self.seasonality_weights = seasonality_weights
        self.seasonality = None
        return

    def __str__(self):
        return f'{self.model}({self.seasonality_weights is not None})'

    def fit(self, y, **kwargs):
        """
        Fit the seasonal component for naive method in the boosting loop.

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
        self.seasonality = y
        if self.normalize_seasonality:
            self.seasonality = self.normalize()
        init_seasonality = self.seasonality[:self.seasonal_period]
        last_seasonality = self.seasonality[-self.seasonal_period:]
        self.seasonality = np.append(init_seasonality, self.seasonality[:-self.seasonal_period])
        self.seasonality = self.seasonality * kwargs['seasonality_lr']
        self.model_params = last_seasonality
        future_seasonality = np.resize(last_seasonality, len(y) + self.seasonal_period)
        self.model_params = future_seasonality[-self.seasonal_period:]
        return self.seasonality

    def predict(self, forecast_horizon, model_params):
        return np.resize(model_params, forecast_horizon)

