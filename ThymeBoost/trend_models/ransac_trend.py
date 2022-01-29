# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

class RansacModel(TrendBaseModel):
    """Uses sklearn's RANSACRegressor method to build a robust regression.
    The parameters that can be passed for this trend are:
        ransac_min_samples
        ransac_trials
        poly
        fit_constant
    For more info: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html
    """
    model = 'ransac'
    
    def __init__(self):
        self.model_params = None
        self.fitted = None

    def __str__(self):
        return f'{self.model}({self.kwargs["poly"]})'

    def get_polynomial_expansion(self, X, poly):
        """
        Polynomial expansion for curvey lines! 
        poly == 2 => trend = b1*x1 + b2*x1^2 

        Parameters
        ----------
        X : np.array
            Input X matrix.
        poly : int
            Order of the expansion.

        Returns
        -------
        np.array
            X matrix with expansion.
        """
        return PolynomialFeatures(degree=poly, include_bias=False).fit(X).transform(X) 

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
        Fit the trend component in the boosting loop for a collection of linear models.

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
        poly = kwargs['poly']
        fit_constant = kwargs['fit_constant']
        trials = kwargs['ransac_trials']
        min_samples = kwargs['ransac_min_samples']
        y = y - bias
        y = (y).reshape((-1, 1))
        X = np.array(list(range(len(y))), ndmin=1).reshape((-1, 1))
        if poly > 1:
            X = self.get_polynomial_expansion(X, poly)
        if fit_constant:
            X = self.add_constant(X)
        model = RANSACRegressor(LinearRegression(fit_intercept=False),
                                min_samples=min_samples,
                                max_trials=trials,
                                random_state= 32)
        model.fit(X, y)
        self.fitted = model.predict(X) + bias
        slope = self.fitted[-1] - self.fitted[-2]
        last_value = self.fitted[-1]
        self.model_params = (slope, last_value)
        return self.fitted.reshape(-1, )

    def predict(self, forecast_horizon, model_params):
        last_fitted_value = model_params[1]
        slope = model_params[0]
        predicted = np.arange(1, forecast_horizon + 1) * slope + last_fitted_value
        return predicted
