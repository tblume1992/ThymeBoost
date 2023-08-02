# -*- coding: utf-8 -*-

from ThymeBoost.trend_models.trend_base_class import TrendBaseModel


class FastSESModel(TrendBaseModel):
    """Fast simple exponential smoother Model from Statsforecast"""
    model = 'fast_ses'

    def __init__(self):
        self.model_params = None
        self.fitted = None
        self._online_steps = 0

    def __str__(self):
        return f'{self.model}(optimized)'

    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop for a fast ses model.

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
            from statsforecast.models import SimpleExponentialSmoothingOptimized
        except Exception:
            raise ValueError('Using Fast implementations requires an optional dependency Statsforecast: pip install statsforecast')
        self.kwargs = kwargs
        bias = kwargs['bias']
        ets_model = SimpleExponentialSmoothingOptimized().fit(y - bias)
        self.fitted = ets_model.predict_in_sample()['fitted']
        self.model_params = (ets_model, bias)
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        prediction = model_params[0].predict(h=forecast_horizon)['mean'] + \
                         model_params[1]
        return prediction
#%%
