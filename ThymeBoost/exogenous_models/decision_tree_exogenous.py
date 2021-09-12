# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeRegressor
from ThymeBoost.exogenous_models.exogenous_base_class import ExogenousBaseModel


class DecisionTree(ExogenousBaseModel):
    model = 'decision_tree'

    def __init__(self):
        self.model_obj = None
        self.fitted = None

    def fit(self, y, X, **kwargs):
        tree_depth = kwargs['tree_depth']
        exo_model = DecisionTreeRegressor(max_depth=tree_depth)
        self.model_obj = exo_model.fit(X, y)
        self.fitted = self.model_obj.predict(X)
        #exo_impact = (exo_model.params, fitted_model.cov_params())
        return self.fitted

    def predict(self, future_exogenous):
        return self.model_obj.predict(future_exogenous)
