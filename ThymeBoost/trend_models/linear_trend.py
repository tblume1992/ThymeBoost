# -*- coding: utf-8 -*-
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class LinearModel(TrendBaseModel):
    model = 'linear'
    
    def __init__(self):
        self.model_params = None
        self.fitted = None

    def __str__(self):
        return f'{self.model}({self.kwargs["poly"], self.kwargs["l2"]})'

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
    
    def ridge_regression(self, X, y, l2):
        """
        Equation to derive coefficients with a ridge constrain: l2, may not be super useful but here it is.

        Parameters
        ----------
        X : np.array
            Input X matrix.
        y : np.array
            Input time series.
        l2 : float
            Ridge constraint, obviously scale dependent so beware!
        Returns
        -------
        np.array
            Our ridge beta coefficients to get predictions.

        """
        return np.linalg.pinv(X.T.dot(X) + l2*np.eye(X.shape[1])).dot(X.T.dot(y))
    
    def wls(self, X, y, weight):
        """
        Simple WLS where weighting is based on previous error.  If True, then our take on a IRLS scheme in the boosting loop.
        If iterable then apply those weights assuming these are sample weights.
        ToDo: Does IRLS like this even work ok?

        Parameters
        ----------
        X : np.array
            Input X matrix.
        y : np.array
            Input time series.
        weight : boolean/np.array
            if True then apply IRLS weighting scheme, else apply sample weights.
        Returns
        -------
        np.array
            Our beta coefficients to get predictions.

        """
        if isinstance(weight, bool) and weight:
            #since we are boosting y is our error term from last iteration, so this works right?
            weight = np.diag(1/(y.reshape(-1,)**2))
        weighted_X_T = X.T @ weight
        return np.linalg.pinv(weighted_X_T.dot(X)).dot(weighted_X_T.dot(y))  

    def ols(self, X, y):
        """
        Simple OLS with normal equation.  Obviously we have a singluar matrix so we use pinv.
        ToDo: Look to implement faster equations for simple trend lines to speed up comp time.

        Parameters
        ----------
        X : np.array
            Input X matrix.
        y : np.array
            Input time series.

        Returns
        -------
        np.array
            Our beta coefficients to get predictions.

        """
        return np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
    
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
        weight = kwargs['trend_weights']
        l2 = kwargs['l2']
        y = y - bias
        y = (y).reshape((-1, 1))
        X = np.array(list(range(len(y))), ndmin=1).reshape((-1, 1))  
        if poly > 1:
            X = self.get_polynomial_expansion(X, poly)
        if fit_constant:
            X = self.add_constant(X)
        if l2:
            beta = self.ridge_regression(X, y, l2)
        elif weight is not None:
            beta = self.wls(X, y, weight)
        else:
            beta =  self.ols(X, y)
        self.fitted = X.dot(beta) + bias
        slope = self.fitted[-1] - self.fitted[-2]
        self.model_params = (slope, self.fitted[-1])
        return self.fitted.reshape(-1, )

    def predict(self, forecast_horizon, model_params):
        last_fitted_value = model_params[1]
        slope = model_params[0]
        predicted = np.arange(1, forecast_horizon + 1) * slope + last_fitted_value
        return predicted.reshape(-1, )
