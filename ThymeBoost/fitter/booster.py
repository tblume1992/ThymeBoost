# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import copy

from ThymeBoost.utils.get_complexity import get_complexity
from ThymeBoost.cost_functions import calc_cost
from ThymeBoost.fitter.decompose import Decompose


class booster(Decompose):

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
                 **kwargs):
        time_series = pd.Series(time_series).copy(deep=True)
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

    def initialize_booster_values(self):
        self.split = None
        self.i = -1
        self.trend_objs = []
        self.seasonal_objs = []
        self.exo_objs = []
        self.trend_pred_params = []
        self.seasonal_pred_params = []
        self.exo_pred_params = []
        self.trends = []
        self.seasonalities = []
        self.errors = []
        self.fitted_exogenous = []
        self.exo_class = None
        self.trend_strengths = []

    def update_params(self,
                      total_trend,
                      total_seasonal,
                      total_exo):
        self.seasonal_pred_params.append(self.seasonal_obj.model_params)
        self.seasonal_objs.append(self.seasonal_obj)
        self.trend_pred_params.append(self.trend_obj.model_params)
        self.trend_objs.append(self.trend_obj)
        self.exo_objs.append(self.exo_class)
        self.trend_strengths.append(self.trend_strength)
        self.total_trend = total_trend
        self.total_seasonalities = total_seasonal
        if self.boosting_params['exogenous'] is None:
            self.total_fitted_exogenous = None
        else:
            self.total_fitted_exogenous = total_exo

    @staticmethod
    def calc_trend_strength(resids, deseasonalized):
        return max(0, 1-(np.var(resids)/np.var(deseasonalized)))

    def boosting_log(self, round_cost):
        #quick printing
        #TODO replace with logging
            print(f'''{"*"*10} Round {self.i+1} {"*"*10}''')
            print(f'''Using Split: {self.split}''')
            if self.i == 0:
                print(f'''Fitting initial trend globally with trend model:''')
            else:
                print(f'''Fitting {self.trend_objs[-1].fit_type} with trend model:''')
            print(f'''{str(self.trend_objs[-1].model_obj)}''')
            print(f'''seasonal model:''')
            print(f'''{str(self.seasonal_objs[-1].model_obj)}''')
            if self.exo_class is not None:
                print(f'''exogenous model:''')
                print(f'''{str(self.exo_objs[-1].model_obj)}''')
            print(f'''cost: {round_cost}''')

    def boost(self):
        self.initialize_booster_values()
        __boost = True
        while __boost:
            self.i += 1
            if self.i == self.n_rounds:
                break
            round_results = self.additive_boost_round(self.i)
            current_prediction, total_trend, total_seasonal, total_exo = round_results
            resids = self.time_series - current_prediction
            self.trend_strength = booster.calc_trend_strength(resids,
                                                              resids + total_trend)
            self.c = get_complexity(self.i,
                                    self.boosting_params['poly'],
                                    self.boosting_params['fit_type'],
                                    self.boosting_params['trend_estimator'],
                                    self.boosting_params['arima_order'],
                                    self.boosting_params['window_size'],
                                    self.time_series,
                                    self.boosting_params['fourier_order'],
                                    self.boosting_params['seasonal_period'],
                                    self.boosting_params['exogenous'])
            round_cost = calc_cost(self.time_series,
                                   current_prediction,
                                   self.c,
                                   self.regularization,
                                   self.boosting_params['global_cost'])

            if self.i == 0:
                self.cost = round_cost
            if (round_cost <= self.cost and self.n_rounds == -1) or self.i < self.n_rounds:
                if self.cost > 0:
                    self.cost = round_cost - self.cost_penalty*round_cost
                else:
                    EPS = .000000000000001
                    self.cost = round_cost + self.cost_penalty*round_cost - EPS
                self.update_params(total_trend,
                                   total_seasonal,
                                   total_exo)
                if self.verbose:
                    self.boosting_log(round_cost)
            else:
                assert self.i > 0, 'Boosting terminated before beginning'
                __boost = False
                if self.verbose:
                    print(f'{"="*30}')
                    print(f'Boosting Terminated \nUsing round {self.i}')
                break
        return self.total_trend, self.total_seasonalities, self.total_fitted_exogenous


