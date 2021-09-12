# -*- coding: utf-8 -*-
import statsmodels.api as sm
from ThymeBoost.exogenous_models.exogenous_base_class import ExogenousBaseModel


class OLS(ExogenousBaseModel):
    model = 'ols'

    def __init__(self):
        self.model_obj = None
        self.fitted = None

    def fit(self, y, X, **kwargs):
        exo_model = sm.OLS(y, X)
        self.model_obj = exo_model.fit()
        self.fitted = self.model_obj.predict(X)
        #exo_impact = (exo_model.params, fitted_model.cov_params())
        return self.fitted

    def predict(self, future_exogenous):
        return self.model_obj.predict(future_exogenous)
