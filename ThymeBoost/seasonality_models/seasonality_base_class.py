
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class SeasonalityBaseModel(ABC):
    """
    Seasonality Abstract Base Class.
    """
    model = None
    
    @abstractmethod
    def __init__(self):
        self.seasonality = None
        pass

    def __str__(cls):
        return f'{cls.model} model'

    @abstractmethod
    def fit(self, y, **kwargs):
        """
        Fit the seasonal component in the boosting loop.

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
    
    def __add__(self, seas_obj):
        """
        Add two seasonal obj together, useful for ensembling or just quick updating of seasonal components.

        Parameters
        ----------
        trend_obj : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.fitted + seas_obj.fitted

    def __mul__(self, seas_obj):
        return self.fitted * seas_obj.fitted

    def __div__(self, seas_obj):
        return self.fitted / seas_obj.fitted

    def __sub__(self, seas_obj):
        return self.fitted - seas_obj.fitted

    def append(self, seas_obj):
        return np.append(self.fitted, seas_obj.fitted)

    def to_series(self, array):
        return pd.Series(array)

    def normalize(self):
        """Enforce average seasonlaity of 0 for 'add' seasonality and 1 for 'mult' seasonality"""
        self.seasonality -= np.mean(self.seasonality)
        return self.seasonality
