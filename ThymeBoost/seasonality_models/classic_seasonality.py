# -*- coding: utf-8 -*-
import numpy as np
from ThymeBoost.seasonality_models.seasonality_base_class import SeasonalityBaseModel


class ClassicSeasonalityModel(SeasonalityBaseModel):
    """
    Seasonality for naive decomposition method.
    """
    model = 'classic'

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
        if self.seasonality_weights is not None:
            y = y * self.seasonality_weights
        avg_seas = [np.mean(y[i::self.seasonal_period], axis=0) for i in range(self.seasonal_period)]
        avg_seas = np.array(avg_seas)
        self.seasonality = np.resize(avg_seas, len(y))
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
