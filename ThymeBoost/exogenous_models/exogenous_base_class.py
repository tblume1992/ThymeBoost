# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class ExogenousBaseModel(ABC):
    """
    Exogenous Abstract Base Class.
    """
    model = None
    
    @abstractmethod
    def __init__(self):
        self.fitted = None
        pass

    def __str__(cls):
        return f'{cls.model} model'

    @abstractmethod
    def fit(self, y, **kwargs):
        """
        Fit the exogenous component in the boosting loop.

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
    def predict(self, future_exogenous, forecast_horizon):
        pass
    
    def __add__(self, exo_object):
        """
        Add two exo obj together, useful for ensembling or just quick updating of exo components.

        Parameters
        ----------
        exo_object : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.fitted + exo_object.fitted

    def __mul__(self, exo_object):
        return self.fitted * exo_object.fitted

    def __div__(self, exo_object):
        return self.fitted / exo_object.fitted

    def __sub__(self, exo_object):
        return self.fitted - exo_object.fitted

    def append(self, exo_object):
        return np.append(self.fitted, exo_object.fitted)

    def to_series(self, array):
        return pd.Series(array)
