
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from ThymeBoost.cost_functions import get_split_cost

class TrendBaseModel(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop.

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
        pass
    
    @abstractmethod
    def predict(self, forecast_horizon):
        pass
    
    def __add__(self, trend_obj):
        """
        Add two trend obj together, useful for ensembling or just quick updating of trend components.

        Parameters
        ----------
        trend_obj : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.fitted + trend_obj.fitted

    def __mul__(self, trend_obj):
        return self.fitted * trend_obj.fitted
    
    def __div__(self, trend_obj):
        return self.fitted / trend_obj.fitted

    def __sub__(self, trend_obj):
        return self.fitted - trend_obj.fitted
    
    def append(self, trend_obj):
        return np.append(self.fitted, trend_obj.fitted)
    
    def to_series(self, array):
        return pd.Series(array)

    def _split_cost(self, time_series, cost_function: str, trend_obj=None):
        return get_split_cost(time_series, 
                              self.fitted, 
                              trend_obj.fitted, 
                              cost_function)

    def __str__(cls):
        return f'{cls.model} model'

    def get_frequency(self, index):
        if len(index) > 2:
            freq = pd.infer_freq(index)
        elif len(index) == 2:
            time_delta = (index[-1] - index[0]).days
            if time_delta > 2 and time_delta < 20:
                freq = 'W'
            elif time_delta < 2:
                freq = 'D'
            elif time_delta > 20 and time_delta < 40:
                freq = 'M'
            elif time_delta > 300:
                freq = 'Y'
        else:
            #safe setting to month
            freq = 'M'
        return freq
