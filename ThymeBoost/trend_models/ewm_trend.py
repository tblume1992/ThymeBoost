# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
import numpy as np
import pandas as pd

class EwmModel(TrendBaseModel):
    """
    The ewm method utilizes a Pandas ewm method: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html.
    Fitting this with a 'local' fit_type parameter is not advised.
    """
    model = 'ewm'
    
    def __init__(self):
        self.model_params = None
        self.fitted = None
        self._online_steps = 0

    def __str__(self):
        return f'{self.model}({self.kwargs["ewm_alpha"]})'

    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop for a ewm model using the 'ewm_alpha' parameter.

        Parameters
        ----------
        time_series : np.ndarray
            DESCRIPTION.
        **kwargs : 
            The key 'ewm_alpha' is passed to this method from the ThymeBoost fit method.

        Returns
        -------
        Fitted array.

        """
        self.kwargs = kwargs
        alpha = kwargs['ewm_alpha']
        bias = kwargs['bias']
        y = pd.Series(y - bias)
        self.fitted = np.array(y.ewm(alpha=alpha).mean()) + bias
        last_fitted_values = self.fitted[-1]
        self.model_params = last_fitted_values
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        return np.tile(model_params, forecast_horizon)
