# -*- coding: utf-8 -*-

import copy
import pandas as pd
import numpy as np
from ThymeBoost.fit_components.fit_trend import FitTrend
from ThymeBoost.fit_components.fit_seasonality import FitSeasonality
from ThymeBoost.fit_components.fit_exogenous import FitExogenous


class Decompose:

    def __init__(self,
                 time_series,
                 given_splits,
                 verbose,
                 n_split_proposals,
                 approximate_splits,
                 exclude_splits,
                 cost_penalty,
                 normalize_seasonality,
                 regularization,
                 n_rounds,
                 smoothed_trend,
                 additive,
                 split_strategy,
                 **kwargs):
        time_series = pd.Series(time_series)
        self.time_series_index = time_series.index
        self.time_series = time_series.values
        self.boosted_data = self.time_series
        self.kwargs = kwargs
        self.boosting_params = copy.deepcopy(self.kwargs)
        self.verbose = verbose
        self.n_split_proposals = n_split_proposals
        self.approximate_splits = approximate_splits
        self.exclude_splits = exclude_splits
        self.given_splits = given_splits
        self.cost_penalty = cost_penalty
        self.normalize_seasonality = normalize_seasonality
        self.regularization = regularization
        self.n_rounds = n_rounds
        self.smoothed_trend = smoothed_trend
        self.additive = additive
        self.split_strategy = split_strategy

    def update_iterated_features(self):
        self.boosting_params = {k: next(v) for k, v in self.kwargs.items()}

    def multiplicative_fit(self):
        raise ValueError('Multiplicative Seasonality is not enabled!')

    def get_init_trend_component(self, time_series):
        self.trend_obj = FitTrend(trend_estimator=next(self.boosting_params['init_trend']),
                                  fit_type='global',
                                  given_splits=self.given_splits,
                                  exclude_splits=self.exclude_splits,
                                  min_sample_pct=.01,
                                  poly=1,
                                  trend_weights=None,
                                  l2=None,
                                  n_split_proposals=self.n_split_proposals,
                                  approximate_splits=self.approximate_splits,
                                  split_cost='mse',
                                  trend_lr=1,
                                  time_series_index=self.time_series_index,
                                  smoothed=False,
                                  connectivity_constraint=True,
                                  split_strategy=self.split_strategy
                                  )
        trend = self.trend_obj.fit_trend_component(time_series)
        self.trends.append(trend)
        self.split = self.trend_obj.split
        return trend

    def get_trend_component(self, time_series):
        self.trend_obj = FitTrend(given_splits=self.given_splits,
                                  exclude_splits=self.exclude_splits,
                                  approximate_splits=self.approximate_splits,
                                  time_series_index=self.time_series_index,
                                  smoothed=self.smoothed_trend,
                                  n_split_proposals=self.n_split_proposals,
                                  additive=self.additive,
                                  split_strategy=self.split_strategy,
                                  **self.boosting_params)
        trend = self.trend_obj.fit_trend_component(time_series)
        self.trends.append(trend)
        self.split = self.trend_obj.split
        return trend

    def get_seasonal_component(self, detrended):
        self.seasonal_obj = FitSeasonality(normalize_seasonality=self.normalize_seasonality,
                                           additive=self.additive,
                                           **self.boosting_params)
        seasonality = self.seasonal_obj.fit_seasonal_component(detrended)
        self.seasonalities.append(seasonality)
        return seasonality

    def get_exogenous_component(self, residual):
        self.exo_class = FitExogenous(self.boosting_params['exogenous_estimator'],
                                      **self.boosting_params)
        exo_fit = self.exo_class.fit_exogenous_component(self.boosted_data,
                                                         self.boosting_params['exogenous'],
                                                         )
        self.fitted_exogenous.append(exo_fit)
        self.boosted_data = self.boosted_data - exo_fit
        return exo_fit

    def additive_boost_round(self, round_number):
        if round_number == 0:
            trend = self.get_init_trend_component(self.boosted_data)
        else:
            trend = self.get_trend_component(self.boosted_data)
        self.update_iterated_features()
        detrended = self.boosted_data - trend
        self.boosted_data = detrended
        seasonality = self.get_seasonal_component(detrended)
        self.boosted_data -= seasonality
        if self.boosting_params['exogenous'] is not None:
            self.get_exogenous_component(self.boosted_data)
        #self.errors.append(np.mean(np.abs(self.boosted_data)))
        total_trend = np.sum(self.trends, axis=0)
        total_seasonalities = np.sum(self.seasonalities, axis=0)
        total_exo = np.sum(self.fitted_exogenous, axis=0)
        current_prediction = (total_trend +
                              total_seasonalities +
                              total_exo)
        return current_prediction, total_trend, total_seasonalities, total_exo
