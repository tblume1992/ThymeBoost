# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from CostFunctions import get_split_cost
from SplitProposals import SplitProposals
import statsmodels as sm
from scipy.signal import savgol_filter
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
    

class FitTrend:
    """Approximates the trend component
    
        Parameters
        ----------
        poly : int
            Polynomial expansion for linear models.       
        
        trend_estimator : str
            The estimator to use to approximate trend.
        
        fit_type : str
            Whether a 'global' or 'local' fit is used.  This parameter
            should be set to 'global' for loess and ewm.
                   
        given_splits : list
            Splits to use when using fit_type='local'.
        
        exclude_splits : list
            exclude these index when considering splits for 
            fit_type='local'.  Must be idx not datetimeindex if 
            using a Pandas Series.
                         
        min_sample_pct : float 
            Percentage of samples required to consider a split. 
            Must be 0<min_sample_pct<=1.
        
        n_split_proposals : int
            Number of split proposals based on the gradients.
                            
        approximate_splits : boolean
            Whether to use proposal splits based on gradients
            or exhaustively try splits with at least 
            min_sample_pct samples.
            
        l2 : float
            l2 regularization to apply.  Must be >0.
            
        split_cost : str
            What cost function to use when selecting the split, selections are
            'mse' or 'mae'.
            
        trend_lr : float
            Applies a learning rate in accordance to standard gradient boosting
            such that the trend = trend * trend_lr at each iteration.
            
        forecast_horizon : int
            Number of steps to take in forecasting out-of-sample.
            
        window_size : int
            How many samples are taken into account for sliding window methods
            such as loess and ewm.
        
        smoothed : boolean
            Whether to smooth the resulting trend or not, by default not too
            much smoothing is applied just enough to smooth the kinks.
        
        RETURNS
        ----------
        numpy array : the trend component
        numpy array : the predicted trend component
        
    """
    def __init__(self, poly, 
                 trend_estimator, 
                 fit_type,
                 given_splits, 
                 exclude_splits, 
                 min_sample_pct,
                 n_split_proposals,
                 approximate_splits,
                 l2,
                 split_cost,
                 trend_lr,
                 time_series_index,
                 forecast_horizon,
                 arima_order,
                 window_size,
                 smoothed,
                 ):
        self.trend_estimator = trend_estimator
        self.fit_type = fit_type
        self.poly = poly
        self.given_splits = given_splits
        self.exclude_splits = exclude_splits
        self.min_sample_pct = min_sample_pct
        self.n_split_proposals = n_split_proposals
        self.approximate_splits = approximate_splits
        self.l2 = l2
        self.arima_order = arima_order
        self.split_cost = split_cost
        self.trend_lr = trend_lr
        self.time_series_index = time_series_index
        self.forecast_horizon = forecast_horizon
        self.smoothed = smoothed
        self.window_size = window_size
        
    def set_estimator(self, trend_estimator):
        """

        Parameters
        ----------
        trend_estimator : TYPE
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        fit_obj : TYPE
            DESCRIPTION.

        """
        if trend_estimator == 'mean':
            fit_obj = self.mean
        elif trend_estimator == 'median':
            fit_obj = self.median
        elif trend_estimator == 'linear':
            fit_obj = self.linear
        elif trend_estimator == 'ses':
            fit_obj = self.ses
        elif trend_estimator == 'des':
            fit_obj = self.des
        elif trend_estimator == 'ar':
            fit_obj = self.ar
        elif trend_estimator == 'loess':
            fit_obj = self.loess
        elif trend_estimator == 'ewm':
            fit_obj = self.ewm
        else:
            raise NotImplementedError('That trend estimation is not availale yet, add it to the road map!')        
        return fit_obj
        
    def linear(self,y, bias = 0, fit_constant = True): 
        y = y - bias
        y = (y).reshape((-1, 1))
        X = np.array(list(range(len(y))), ndmin=1).reshape((-1, 1))  
        if self.poly > 1:
            X = PolynomialFeatures(degree = self.poly, include_bias = False).fit(X).transform(X) 
        if fit_constant:
          X = np.append(X, np.asarray(np.ones(len(y))).reshape(len(y), 1), axis = 1)
        beta =  np.linalg.pinv(X.T.dot(X) + self.l2*np.eye(X.shape[1])).dot(X.T.dot(y))
        fitted = X.dot(beta) + bias
        fitted_slope = fitted[-1] - fitted[-2]
        predicted = np.arange(1, self.forecast_horizon + 1) * fitted_slope + fitted[-1]       
        return fitted, predicted 
    
    def ewm(self, y, bias = 0, fit_constant = False):
        y = pd.Series(y - bias)
        fitted = np.array(y.ewm(span = self.window_size).mean())
        fitted_slope = np.mean(np.gradient(fitted)[-self.window_size:])
        predicted = np.arange(1, self.forecast_horizon + 1) * fitted_slope + fitted[-1]       
        return fitted + bias, predicted + bias
    
    def loess(self, y, bias = 0, fit_constant = False):
        fitted = savgol_filter(y - bias, self.window_size, self.poly)
        fitted_slope = np.mean(np.gradient(fitted)[-int(.15*len(y)):])
        predicted = np.arange(1, self.forecast_horizon + 1) * fitted_slope + fitted[-1]       
        return fitted + bias, predicted + bias
    
    def ses(self, y, bias = 0, fit_constant = True):
        smoother = SimpleExpSmoothing(y - bias)
        fit_model = smoother.fit()  
        fitted = fit_model.fittedvalues
        predicted = fit_model.forecast(self.forecast_horizon)                                                                         
        return fitted + bias, predicted + bias
    
    def des(self, y, bias = 0, fit_constant = True):
        smoother = Holt(y - bias)
        fit_model = smoother.fit()  
        fitted = fit_model.fittedvalues
        predicted = fit_model.forecast(self.forecast_horizon)                                                                         
        return np.asarray(fitted + bias), np.asarray(predicted + bias)
    
    def ar(self, y, bias = 0, fit_constant = False):
        order = self.arima_order
        ar_model = sm.tsa.arima.model.ARIMA(y - bias, order = order).fit()
        results = ar_model.predict(start = 0, end = len(y) + self.forecast_horizon - 1) + bias
        fitted = results[:len(y)]
        predicted = results[len(y):]       
        return fitted, predicted
  
    def mean(self, y, bias = 0, fit_constant = False):
        mean_est = np.mean(y)
        fitted = np.tile(mean_est, len(y)) 
        predicted = np.tile(mean_est, self.forecast_horizon) 
        return fitted, predicted
    
    def median(self, y, bias = 0, fit_constant = False):
        median_est = np.median(y)
        fitted = np.tile(median_est, len(y)) 
        predicted = np.tile(median_est, self.forecast_horizon) 
        return fitted, predicted
    
    def fit(self, time_series):
        fit_obj = self.set_estimator(self.trend_estimator)
        if self.fit_type == 'local':
            proposals = SplitProposals(
                        given_splits = self.given_splits, 
                       exclude_splits = self.exclude_splits, 
                       min_sample_pct = self.min_sample_pct,
                       n_split_proposals = self.n_split_proposals,
                       approximate_splits = self.approximate_splits)
            proposals = proposals.get_split_proposals(time_series)
            for index, i in enumerate(proposals):  
              predicted1, prop_predicted = fit_obj(time_series[:i])
              predicted2, prop_predicted = fit_obj(time_series[i:], 
                                                   bias = float(predicted1[-1]),
                                                   fit_constant = False)
              iteration_cost = get_split_cost(
                                              time_series, 
                                              predicted1, 
                                              predicted2, 
                                              self.split_cost
                                              )             
              if index == 0:
                cost = iteration_cost
              if iteration_cost <= cost:
                split = self.time_series_index[i]
                cost = iteration_cost  
                fitted = np.append(predicted1,predicted2)
                predicted = prop_predicted
            try:
                self.split = split
            except:
                fitted, predicted = fit_obj(time_series, bias = 0, fit_constant = True)
                fitted = fitted.reshape(-1,)    
                self.split = None
            if self.smoothed:
                fitted = savgol_filter(fitted, self.window_size, self.poly)
        elif self.fit_type == 'global':
            fitted, predicted = fit_obj(time_series, bias = 0, fit_constant = True)
            fitted = fitted.reshape(-1,)            
            self.split = None
        else:
            raise NotImplementedError('Trend estimation must be local or global')         
        return fitted * self.trend_lr, predicted * self.trend_lr
    

  
