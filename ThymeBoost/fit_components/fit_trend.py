# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import savgol_filter
from ThymeBoost.cost_functions import get_split_cost
from ThymeBoost.split_proposals import SplitProposals
from ThymeBoost.trend_models import (linear_trend, mean_trend, median_trend,
                                     loess_trend, ransac_trend, ewm_trend,
                                     ets_trend, arima_trend, moving_average_trend,
                                     zero_trend, svr_trend, naive_trend, croston_trend)


class FitTrend:
    """Approximates the trend component

        Parameters
        ----------
        poly : int
            Polynomial expansion for linear models.

        trend_estimator : str
            The estimator to use to approximate trend.

        fit_type : str
            Whether a 'global' or 'local' fit is used.  This parameter
            should be set to 'global' for loess and ewm.

        given_splits : list
            Splits to use when using fit_type='local'.

        exclude_splits : list
            exclude these index when considering splits for
            fit_type='local'.  Must be idx not datetimeindex if
            using a Pandas Series.

        min_sample_pct : float
            Percentage of samples required to consider a split.
            Must be 0<min_sample_pct<=1.

        n_split_proposals : int
            Number of split proposals based on the gradients.

        approximate_splits : boolean
            Whether to use proposal splits based on gradients
            or exhaustively try splits with at least
            min_sample_pct samples.

        l2 : float
            l2 regularization to apply.  Must be >0.

        split_cost : str
            What cost function to use when selecting the split, selections are
            'mse' or 'mae'.

        trend_lr : float
            Applies a learning rate in accordance to standard gradient boosting
            such that the trend = trend * trend_lr at each iteration.

        forecast_horizon : int
            Number of steps to take in forecasting out-of-sample.

        window_size : int
            How many samples are taken into account for sliding window methods
            such as loess and ewm.

        smoothed : boolean
            Whether to smooth the resulting trend or not, by default not too
            much smoothing is applied just enough to smooth the kinks.

        RETURNS
        ----------
        numpy array : the trend component
        numpy array : the predicted trend component
    """

    def __init__(self,
                 trend_estimator,
                 fit_type,
                 given_splits,
                 exclude_splits,
                 min_sample_pct,
                 n_split_proposals,
                 approximate_splits,
                 split_cost,
                 trend_lr,
                 time_series_index,
                 smoothed,
                 connectivity_constraint,
                 split_strategy,
                 **kwargs
                 ):
        self.trend_estimator = trend_estimator.lower()
        self.fit_type = fit_type
        self.given_splits = given_splits
        self.exclude_splits = exclude_splits
        self.min_sample_pct = min_sample_pct
        self.n_split_proposals = n_split_proposals
        self.approximate_splits = approximate_splits
        self.split_cost = split_cost
        self.trend_lr = trend_lr
        self.time_series_index = time_series_index
        self.smoothed = smoothed
        self.connectivity_constraint = connectivity_constraint
        self.split = None
        self.split_strategy = split_strategy
        self.kwargs = kwargs

    @staticmethod
    def set_estimator(trend_estimator):
        """

        Parameters
        ----------
        trend_estimator : TYPE
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        fit_obj : TYPE
            DESCRIPTION.

        """
        if trend_estimator == 'mean':
            fit_obj = mean_trend.MeanModel
        elif trend_estimator == 'median':
            fit_obj = median_trend.MedianModel
        elif trend_estimator == 'linear':
            fit_obj = linear_trend.LinearModel
        elif trend_estimator in ('ses', 'des', 'damped_des'):
            fit_obj = ets_trend.EtsModel
        elif trend_estimator == 'arima':
            fit_obj = arima_trend.ArimaModel
        elif trend_estimator == 'loess':
            fit_obj = loess_trend.LoessModel
        elif trend_estimator == 'ewm':
            fit_obj = ewm_trend.EwmModel
        elif trend_estimator == 'ransac':
            fit_obj = ransac_trend.RansacModel
        elif trend_estimator == 'moving_average':
            fit_obj = moving_average_trend.MovingAverageModel
        elif trend_estimator == 'zero':
            fit_obj = zero_trend.ZeroModel
        elif trend_estimator == 'svr':
            fit_obj = svr_trend.SvrModel
        elif trend_estimator == 'naive':
            fit_obj = naive_trend.NaiveModel
        elif trend_estimator == 'croston':
            fit_obj = croston_trend.CrostonModel
        else:
            raise NotImplementedError('That trend estimation is not available yet, add it to the road map!')
        return fit_obj

    def fit_trend_component(self, time_series):
        fit_obj = self.set_estimator(self.trend_estimator)
        if self.fit_type == 'local':
            split = None
            proposals = SplitProposals(given_splits=self.given_splits,
                                       exclude_splits=self.exclude_splits,
                                       min_sample_pct=self.min_sample_pct,
                                       n_split_proposals=self.n_split_proposals,
                                       split_strategy=self.split_strategy,
                                       approximate_splits=self.approximate_splits)
            proposals = proposals.get_split_proposals(time_series)
            for index, i in enumerate(proposals):
                split_1_obj = fit_obj()
                fitted_1_split = split_1_obj.fit(time_series[:i],
                                                 fit_constant=True,
                                                 bias=0,
                                                 model=self.trend_estimator,
                                                 **self.kwargs)
                split_2_obj = fit_obj()
                fitted_2_split = split_2_obj.fit(time_series[i:],
                                                 bias=float(fitted_1_split[-1]),
                                                 fit_constant=(not self.connectivity_constraint),
                                                 model=self.trend_estimator,
                                                 **self.kwargs)
                iteration_cost = get_split_cost(time_series,
                                                fitted_1_split,
                                                fitted_2_split,
                                                self.split_cost)
                if index == 0:
                    cost = iteration_cost
                if iteration_cost <= cost:
                    split = self.time_series_index[i]
                    cost = iteration_cost
                    fitted = split_1_obj.append(split_2_obj)
                    self.model_obj = split_2_obj
                    self.model_params = split_2_obj.model_params
            self.split = split
            if self.split is None:
                raise ValueError('No split found, series length my be too small or some error occurred')
            if self.smoothed:
                fitted = savgol_filter(fitted, self.kwargs['window_size'], self.kwargs['poly'])
        elif self.fit_type == 'global':
            global_obj = fit_obj()
            fitted = global_obj.fit(time_series,
                                    bias=0,
                                    fit_constant=True,
                                    model=self.trend_estimator,
                                    **self.kwargs)
            self.split = None
            self.model_params = global_obj.model_params
            self.model_obj = global_obj
        else:
            raise NotImplementedError('Trend estimation must be local or global')
        return fitted * self.trend_lr
