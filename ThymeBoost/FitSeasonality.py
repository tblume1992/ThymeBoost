# -*- coding: utf-8 -*-
import numpy as np

class FitSeasonality:
    """Approximates the seasonal component
        
    """
    def __init__(self, seasonal_estimator,
                 seasonal_period,
                 fourier_components,
                 forecast_horizon,
                 seasonality_lr,
                 seasonal_sample_weight,
                 additive):
        self.seasonal_estimator = seasonal_estimator
        self.seasonal_period = seasonal_period
        self.fourier_components = fourier_components
        self.forecast_horizon = forecast_horizon
        self.seasonality_lr = seasonality_lr
        self.additive = additive
        self.seasonal_sample_weight = seasonal_sample_weight
    
    def get_fourier_series(self, t):
        x = 2 * np.pi * (np.arange(1, self.fourier_components + 1) / 
                         self.seasonal_period)
        x = x * t[:, None]
        fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)     
        return  fourier_series 
    
    def set_estimator(self):
        if self.seasonal_estimator == 'harmonic':
            seasonal_obj = self.get_harmonic_seasonality 
        elif self.seasonal_estimator == 'naive':
            seasonal_obj = self.get_naive_seasonality        
        return seasonal_obj
    
    def get_harmonic_seasonality(self,y):
        X = self.get_fourier_series(np.arange(len(y)))
        if self.seasonal_sample_weight is None or self.seasonal_sample_weight == 1:
            seasonal_sample_weight = np.ones((len(y),))
        elif self.seasonal_sample_weight == 'regularize':
            seasonal_sample_weight = 1/(y**2)
        elif self.seasonal_sample_weight == 'explode':
            seasonal_sample_weight = (y**2)
        elif callable(self.seasonal_sample_weight):
            seasonal_sample_weight = self.seasonal_sample_weight(y)
        else:
            seasonal_sample_weight = self.seasonal_sample_weight
        weights = np.diag(seasonal_sample_weight)
        X = np.append(X, np.asarray(np.ones(len(y))).reshape(len(y), 1), axis = 1)
        weighted_X_T = X.T @ weights
        beta =  np.linalg.pinv(weighted_X_T.dot(X)).dot(weighted_X_T.dot(y))
        seasonality = X @ beta
        return seasonality 

    def get_naive_seasonality(self,y):
        avg_seas = np.array([np.mean(y[i::self.seasonal_period], axis=0) for i in range(self.seasonal_period)])
        seasonality = np.resize(avg_seas, len(y))
        return seasonality
    
    def normalize_seasonality(self, seasonality):
        return seasonality - np.mean(seasonality)
               
    def fit(self, boosted_data, detrended):
        if self.seasonal_period == 0:
            if self.additive:
                seasonality = np.zeros(len(detrended))
                predicted_seasonality = np.zeros(self.forecast_horizon)
            else:
                seasonality = np.ones(len(detrended))
                predicted_seasonality = np.ones(self.forecast_horizon)
        else:
            esti = self.set_estimator()
            seasonality = esti(detrended)
            seasonality = self.normalize_seasonality(seasonality)
            if self.seasonality_lr:
                # if self.additive:
                    seasonality = seasonality * self.seasonality_lr
                # else:
                #     seasonality = 1+ ((seasonality - 1) * self.seasonality_lr)
            if self.forecast_horizon:
                predicted_seasonality = np.resize(seasonality[:self.seasonal_period], len(detrended) + self.forecast_horizon)
                predicted_seasonality = predicted_seasonality[-self.forecast_horizon:]
            else:
                predicted_seasonality = None
            if np.isnan(seasonality[0]):
                if self.additive:
                    seasonality = np.zeros(len(detrended))
                    predicted_seasonality = np.zeros(self.forecast_horizon)
                else:
                    seasonality = np.ones(len(detrended))
                    predicted_seasonality = np.ones(self.forecast_horizon)   
        return seasonality, predicted_seasonality