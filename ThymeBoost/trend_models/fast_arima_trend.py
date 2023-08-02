# -*- coding: utf-8 -*-

from ThymeBoost.trend_models.trend_base_class import TrendBaseModel


class FastArimaModel(TrendBaseModel):
    """Fast ARIMA Model from Statsforecast"""
    model = 'fast_arima'

    def __init__(self):
        self.model_params = None
        self.fitted = None
        self._online_steps = 0

    def __str__(self):
        return f'{self.model}({self.kwargs["arima_order"]})'

    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop for a fast arima model.

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
        try:
            from statsforecast.models import AutoARIMA
        except Exception:
            raise ValueError('Using Fast implementations requires an optional dependency Statsforecast: pip install statsforecast')
        self.kwargs = kwargs
        bias = kwargs['bias']
        ar_model = AutoARIMA(season_length=0).fit(y - bias)
        self.fitted = ar_model.predict_in_sample()['fitted']
        self.model_params = (ar_model, bias)
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        prediction = model_params[0].predict(h=forecast_horizon)['mean'] + \
                         model_params[1]
        return prediction
#%%
