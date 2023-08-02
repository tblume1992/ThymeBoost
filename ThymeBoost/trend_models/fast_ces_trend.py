# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel


class FastCESModel(TrendBaseModel):
    """Fast ETS Model from Statsforecast"""
    model = 'fast_ces'

    def __init__(self):
        self.model_params = None
        self.fitted = None
        self._online_steps = 0

    def __str__(self):
        return f'{self.model}({self.kwargs["arima_order"]})'

    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop for a fast ets model.

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
            from statsforecast.models import AutoCES
        except Exception:
            raise ValueError('Using Fast implementations requires an optional dependency Statsforecast: pip install statsforecast')
        self.kwargs = kwargs
        bias = kwargs['bias']
        ets_model = AutoCES().fit(y - bias)
        self.fitted = ets_model.predict_in_sample()['fitted']
        self.model_params = (ets_model, bias)
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        prediction = model_params[0].predict(h=forecast_horizon)['mean'] + \
                         model_params[1]
        return prediction
#%%
