# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

class SvrModel(TrendBaseModel):
    model = 'svr'
    
    def __init__(self):
        self.model_params = None
        self.fitted = None

    def __str__(self):
        return f'{self.model}()'



    def add_constant(self, X):
        """
        Add constant to X matrix.  Used to allow intercept changes in the split.
        But main purpose is to allow left split to have intercept but constrain right split for connectivity.

        Parameters
        ----------
        X : np.array
            Input X matrix.

        Returns
        -------
        np.array
            X matrix with constant term.

        """
        return np.append(X, np.asarray(np.ones(len(X))).reshape(len(X), 1), axis = 1)
    
    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop for a SVR.

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
        bias = kwargs['bias']
        fit_constant = kwargs['fit_constant']
        y = y - bias
        y = (y).reshape((-1, 1))
        X = np.array(list(range(len(y))), ndmin=1).reshape((-1, 1))
        if fit_constant:
            X = self.add_constant(X)
            # from Sklearn example
            # TODO: generalize
        svr = GridSearchCV(
            SVR(kernel="rbf", gamma=0.1),
            param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
        )
        svr.fit(X, y)
        self.fitted = svr.predict(X) + bias
        slope = self.fitted[-1] - self.fitted[-2]
        last_value = self.fitted[-1]
        self.model_params = (slope, last_value)
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        last_fitted_value = model_params[1]
        slope = model_params[0]
        predicted = np.arange(1, forecast_horizon + 1) * slope + last_fitted_value
        return predicted
