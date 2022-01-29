# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import numpy as np
from ThymeBoost.trend_models import *

def testing_data():
    seasonality = ((np.cos(np.arange(1, 101))*10 + 50))
    np.random.seed(100)
    true = np.linspace(-1, 1, 100)
    noise = np.random.normal(0, 1, 100)
    y = true + seasonality# + noise
    return y

class BaseModelTest():
    """Allows self without overriding unitTest __init__"""
    def setUp(self):
        self.model_obj = None

    def set_model_obj(self, child_model_obj):
        self.model_obj = child_model_obj
        self._params = {'arima_order': 'auto',
                        'model': 'ses',
                        'bias': 0,
                        'arima_trend': None,
                        'alpha': None,
                        'poly': 1,
                        'fit_constant': True,
                        'l2': 0,
                        'trend_weights': None,
                        'ewm_alpha': .5,
                        'window_size': 13,
                        'ransac_trials': 20,
                        'ransac_min_samples': 5}

    def test_fitted_series(self):
        y = testing_data()
        fitted_values = self.model_obj.fit(y, **self._params)
        self.assertTrue(isinstance(fitted_values, np.ndarray))
        
    def test_predicted_series(self):
        y = testing_data()
        self.model_obj.fit(y, **self._params)
        predictions = self.model_obj.predict(24, self.model_obj.model_params)
        self.assertTrue(isinstance(predictions, np.ndarray))
        
    def test_fitted_null(self):
        y = testing_data()
        fitted_values = self.model_obj.fit(y, **self._params)
        self.assertFalse(pd.Series(fitted_values).isnull().values.any())

    def test_prediction_null(self):
        y = testing_data()
        self.model_obj.fit(y, **self._params)
        predictions = self.model_obj.predict(24, self.model_obj.model_params)
        self.assertFalse(pd.Series(predictions).isnull().values.any())


class ArimaTest(BaseModelTest, unittest.TestCase):
    def setUp(self):
        self.set_model_obj(arima_trend.ArimaModel())

class MeanTest(BaseModelTest, unittest.TestCase):
    def setUp(self):
        self.set_model_obj(mean_trend.MeanModel())

class MedianTest(BaseModelTest, unittest.TestCase):
    def setUp(self):
        self.set_model_obj(median_trend.MedianModel())

class MovingAverageTest(BaseModelTest, unittest.TestCase):
    def setUp(self):
        self.set_model_obj(moving_average_trend.MovingAverageModel())

class ZeroTest(BaseModelTest, unittest.TestCase):
    def setUp(self):
        self.set_model_obj(zero_trend.ZeroModel())

class EwmTest(BaseModelTest, unittest.TestCase):
    def setUp(self):
        self.set_model_obj(ewm_trend.EwmModel())

class EtsTest(BaseModelTest, unittest.TestCase):
    def setUp(self):
        self.set_model_obj(ets_trend.EtsModel())

class LinearTest(BaseModelTest, unittest.TestCase):
    def setUp(self):
        self.set_model_obj(linear_trend.LinearModel())

class LoessTest(BaseModelTest, unittest.TestCase):
    def setUp(self):
        self.set_model_obj(loess_trend.LoessModel())

class SvrTest(BaseModelTest, unittest.TestCase):
    def setUp(self):
        self.set_model_obj(svr_trend.SvrModel())

class RansacTest(BaseModelTest, unittest.TestCase):
    def setUp(self):
        self.set_model_obj(ransac_trend.RansacModel())


if __name__ == '__main__':
    unittest.main()
