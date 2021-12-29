# -*- coding: utf-8 -*-
import numpy as np
from ThymeBoost.exogenous_models import (ols_exogenous,
                                         decision_tree_exogenous,
                                         glm_exogenous)


class FitExogenous:
    def __init__(self,
                 exo_estimator='ols',
                 exogenous_lr=1,
                 **kwargs):
        self.exo_estimator = exo_estimator
        self.exogenous_lr = exogenous_lr
        self.kwargs = kwargs
        return
        
    def set_estimator(self, trend_estimator):
        if trend_estimator == 'ols':
            fit_obj = ols_exogenous.OLS
        elif trend_estimator == 'glm':
            fit_obj = glm_exogenous.GLM
        elif trend_estimator == 'decision_tree':
            fit_obj = decision_tree_exogenous.DecisionTree
        else:
            raise NotImplementedError('That Exo estimation is not availale yet, add it to the road map!')
        return fit_obj

    def fit_exogenous_component(self, time_residual, exogenous):
        self.model_obj = self.set_estimator(self.exo_estimator)()
        exogenous = np.array(exogenous).reshape((-1, 1))
        exo_fitted = self.model_obj.fit(time_residual, exogenous, **self.kwargs)
        return self.exogenous_lr*np.array(exo_fitted)
