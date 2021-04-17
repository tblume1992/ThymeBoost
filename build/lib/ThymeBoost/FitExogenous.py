# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor

class FitExogenous:
    def __init__(self, exo_estimator,
                 exogenous_lr,
                 forecast_horizon):
        self.exo_estimator = exo_estimator
        self.exogenous_lr = exogenous_lr
        self.forecast_horizon = forecast_horizon
        
        return
        
    def set_estimator(self, trend_estimator):
        if trend_estimator == 'ols':
            fit_obj = self.ols
        elif trend_estimator == 'glm':
            fit_obj = self.glm
        elif trend_estimator == 'decision_tree':
            fit_obj = self.decision_tree

        else:
            raise NotImplementedError('That Exo estimation is not availale yet, add it to the road map!')
        
        return fit_obj
    
    def ols(self, y, X):
        exo_model = sm.OLS(y, X)
        fitted_model = exo_model.fit()
        fitted_ = fitted_model.predict(X)
        #exo_impact = (exo_model.params, fitted_model.cov_params())
        exo_impact = None
        
        return fitted_, fitted_model, exo_impact
    
    def glm(self, y, X):
        exo_model = sm.GLM(y, X)
        fitted_model = exo_model.fit()
        fitted_ = fitted_model.predict(X)
        #exo_impact = (exo_model.params, fitted_model.cov_params())
        exo_impact = None
        
        return fitted_, fitted_model, exo_impact
    
    def decision_tree(self, y, X):
        exo_model = DecisionTreeRegressor(max_depth = 3)
        fitted_model = exo_model.fit(X, y)
        fitted_ = fitted_model.predict(X)
        #exo_impact = (exo_model.params, fitted_model.cov_params())
        exo_impact = None
        
        return fitted_, fitted_model, exo_impact
        
        return
    
    def fit(self, time_residual, exogenous, future_exogenous): 
        fit_obj = self.set_estimator(self.exo_estimator)
        exo_fitted, fitted_model, exo_impact = fit_obj(time_residual, exogenous)
        self.fitted_model = fitted_model
        if self.forecast_horizon:
            exo_predicted = self.predict(future_exogenous)
        else:
            exo_predicted = None

        return self.exogenous_lr*np.array(exo_fitted), exo_predicted
    
    def predict(self, future_exogenous):
        return self.exogenous_lr*np.array(self.fitted_model.predict(future_exogenous))
        
    
    
        