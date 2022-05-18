# -*- coding: utf-8 -*-
"""
This will become the autoforecaster which will utilize autofit and then fit a regression to combine forecasts
"""

import numpy as np
import pandas as pd
from ThymeBoost import ThymeBoost as tb


class AutoForecast(tb.ThymeBoost):
    def __init__(self):
        super().__init__
        self.verbose=0

    def forecast(self,
                 y,
                 seasonal_period,
                 forecast_horizon,
                 ):
        output = self.autofit(y.values,
                               seasonal_period=seasonal_period,
                               optimization_type='grid_search',
                               optimization_strategy='holdout',
                               optimization_steps=3,
                               lag=50,
                               optimization_metric='mse',
                               test_set='all',
                                    )
        test_results = self.optimizer.opt_results
        model = []
        predictions = []
        actuals = []
        for key, val in test_results.items():
            for key_2, val_2 in val.items():
                n_preds = len(val_2['predictions'])
                model.append([str(val_2['params'])]*n_preds)
                predictions.append(val_2['predictions'])
                actuals.append(val_2['actuals'])
        model = [item for sublist in model for item in sublist]
        actuals = pd.concat(actuals)
        predictions = pd.concat(predictions)
        smart_ensemble = actuals.to_frame()
        smart_ensemble.columns = ['actuals']
        smart_ensemble['predictions'] = predictions
        smart_ensemble['model'] = model
        pivot_smart_ensemble = smart_ensemble.pivot(
                                                    values='predictions',
                                                    columns=['model'])
        pivot_smart_ensemble['actuals'] = smart_ensemble['actuals'].values[:len(pivot_smart_ensemble)]
        y_ = pivot_smart_ensemble['actuals']
        X = pivot_smart_ensemble.drop('actuals', axis=1).values
        import statsmodels.api as sm
        ols = sm.OLS(y_, X)
        fitted = ols.fit()
        fitted.summary()
        predicted = fitted.predict(X)
        output = self.ensemble(y.values,
                                        trend_estimator=['linear', 'croston'],
                                        fit_type=['global'],
                                        seasonal_estimator=['fourier'],
                                        alpha=[.2],
                                        seasonal_period=[0, 24],
                                        additive=[True, False],)




