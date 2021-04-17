# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import stats
from itertools import cycle
import matplotlib.pyplot as plt
from FitTrend import FitTrend
from FitSeasonality import FitSeasonality
from FitExogenous import FitExogenous
from CostFunctions import calc_cost
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings("ignore")


class ThymeBoost:
  def __init__(
                self,
                seasonal_period=0,
                trend_estimator='mean',
                exo_estimator='ols',
                n_rounds=None,
                l2=0,
                poly=1,
                ols_constant=False,
                fourier_components=10,
                approximate_splits=True,
                regularization=1.2,
                fit_type='local',
                seasonal_estimator='harmonic',
                split_cost='mse',
                global_cost='maic',
                trend_dampening=0,
                exogenous=None,
                verbose=0,
                n_split_proposals=10,
                min_sample_pct=.1,
                exclude_splits=[],
                given_splits=[],
                cost_penalty=.001,
                trend_lr=1,
                additive=True,
                seasonality_lr=1,
                exogenous_lr=1,
                smoothed_trend=False,
                window_size=3,
                seasonal_sample_weight=None,
                damp_factor=0,
                trend_cap_target=None,
                params={},
                arima_order = (1,0,1),
                future_exogenous = None,
                scale_type = None
               ):
    self.l2 = l2
    self.fit_type = fit_type
    self.poly = poly
    if type(seasonal_period) == int:
        seasonal_period = [seasonal_period]
    self.seasonal_period = seasonal_period
    self.trend_estimator = trend_estimator
    if len(arima_order) < 3:
        raise KeyError
    self.arima_order = arima_order
    self.ols_constant = ols_constant
    self.fourier_components = fourier_components
    self.approximate_splits = approximate_splits
    self.regularization = regularization
    self.seasonal_estimator = seasonal_estimator
    self.split_cost = split_cost
    self.global_cost = global_cost
    self.trend_dampening = trend_dampening
    self.exogenous = exogenous
    self.verbose = verbose
    self.n_split_proposals = n_split_proposals
    self.min_sample_pct = min_sample_pct
    self.exclude_splits = exclude_splits
    if n_rounds is None:
        n_rounds = -1
    self.n_rounds = n_rounds
    self.given_splits = given_splits
    self.cost_penalty = cost_penalty
    self.trend_lr = trend_lr
    self.exogenous_lr = exogenous_lr
    self.additive = additive
    if additive:
        self.scale_type = scale_type
    else:
        self.scale_type = 'log'
    self.seasonality_lr = seasonality_lr
    self.smoothed_trend = smoothed_trend
    self.window_size = window_size
    self.params = params
    self.seasonal_sample_weight = seasonal_sample_weight
    self.damp_factor = damp_factor
    self.trend_cap_target = trend_cap_target
    self.exo_estimator = exo_estimator
    self.future_exogenous = future_exogenous
  
  @staticmethod
  def calc_seasonal_strength(resids, detrended):
    try:
        return max(0, 1-(np.var(resids)/np.var(detrended)))
    except:
        return 0 
  
  @staticmethod
  def calc_trend_strength(resids, deseasonalized):
    try:
        return max(0, 1-(np.var(resids)/np.var(deseasonalized)))
    except:
        return 0

  @staticmethod
  def get_fitted_intervals(y, fitted, c):
    sd_error = np.std(y - fitted)
    t_stat = stats.t.ppf(.9, len(y))
    upper = fitted + t_stat*sd_error
    lower = fitted - t_stat*sd_error
    return upper, lower

  @staticmethod
  def get_predicted_intervals(y, fitted, predicted, c):
    sd_error = np.std(y - fitted)
    t_stat = stats.t.ppf(.9, len(y))
    len_frac = len(predicted)/len(fitted)
    interval_uncertainty_param = (
                                  np.linspace(1.0, 1.0 + 3*len_frac, 
                                  num=len(predicted))
                                  )
    upper = predicted + t_stat*sd_error*interval_uncertainty_param
    lower = predicted - t_stat*sd_error*interval_uncertainty_param
    return upper, lower

  def scale_input(self, time_series):
    if self.scale_type == 'standard':
        self.time_series_mean = time_series.mean()
        self.time_series_std = time_series.std()
        time_series = (time_series - self.time_series_mean) / self.time_series_std
    elif self.scale_type == 'log':
        assert time_series.all(), 'Series can not contain 0 for mult. fit or log scaling'
        assert (time_series > 0).all(), 'Series can not contain neg. values for mult. fit or log scaling'
        time_series = np.log(time_series)
    elif self.scale_type is None:
        time_series = time_series
    else:
        print('Scaler not recognized!')
    return time_series

  def unscale_input(self, scaled_series):
    if self.scale_type == 'standard':
        return scaled_series * self.time_series_std + self.time_series_mean
    elif self.scale_type == 'log':
        return np.exp(scaled_series)
    elif self.scale_type is None:
        return scaled_series

  def get_complexity(self, boosting_round):
      #Get a measure of complexity: number of splits + any extra variables
      if self.fit_type == 'global' and self.trend_estimator == 'linear':
          c = 1
      elif self.fit_type == 'global' and self.trend_estimator == 'loess':
          c = int(len(self.time_series) / self.window_size)
      elif self.fit_type == 'global' and self.trend_estimator == 'ar':
          c = np.sum(self.arima_order)
          
      else:
          if self.seasonal_period != 0:
            c = boosting_round + self.fourier_components + 1
          else:
            c = boosting_round + 1
          if self.trend_estimator == 'linear' and self.fit_type == 'local':
            c = self.poly + self.fourier_components + boosting_round
          if self.exogenous is not None:
              c += np.shape(self.exogenous)[1]  
      return c
  
  def additive_fit(self):
    boost_ = True
    i = -1
    while boost_:
      i += 1
      if i == 0:
          trend = np.tile(np.median(self.time_series), len(self.time_series))
          predicted_trend = np.tile(np.median(self.time_series), self.forecast_horizon)
      else: 
          trend, predicted_trend = self.trend_class.fit(self.boosted_data)
          if self.verbose and self.trend_class.split is not None:
              print(f'Using Split: {self.trend_class.split}')
      resid = self.boosted_data - trend
      seasonality, predicted_seasonality = self.seasonal_class.fit(self.boosted_data, resid)
      self.seasonal_class.seasonality_period = next(self.seasonality_cycle)
      self.boosted_data = self.boosted_data-(trend+seasonality)  
      if self.exogenous is not None:
           exo_fit, exo_predicted = self.exo_class.fit(self.boosted_data, 
                                                  self.exogenous,
                                                  self.future_exogenous)
           self.fitted_exogenous.append(exo_fit)
           self.boosted_data = self.boosted_data - exo_fit
           self.predicted_exogenous.append(exo_predicted)
      else:
          self.predicted_exogenous = 0
          self.fitted_exogenous = []
      #self.errors.append(np.mean(np.abs(self.boosted_data)))
      self.trends.append(trend)
      self.predicted_seasonalities.append(predicted_seasonality)
      self.predicted_trends.append(predicted_trend)
      self.seasonalities.append(seasonality)
      total_trend = np.sum(self.trends, axis=0)
      total_seasonalities = np.sum(self.seasonalities, axis=0)
      total_fitted_exogenous = np.sum(self.fitted_exogenous, axis=0)
      if self.forecast_horizon:
          total_predicted_trends = np.sum(self.predicted_trends, axis=0)
          total_predicted_seasonalities = np.sum(
                                                 self.predicted_seasonalities, 
                                                 axis=0
                                                 )
          if self.future_exogenous is not None:
              total_predicted_exogenous = np.sum(self.predicted_exogenous, axis=0)
          else:
              total_predicted_exogenous = 0
      else:
          total_predicted_trends = self.predicted_trends
          total_predicted_seasonalities = self.predicted_seasonalities
          total_predicted_exogenous = 0
      self.c = self.get_complexity(boosting_round=i)
      current_prediction = (total_trend + 
                         total_seasonalities + 
                         total_fitted_exogenous)
      round_cost = calc_cost(self.time_series,
                           current_prediction, 
                           self.c, 
                           self.regularization, 
                           self.global_cost)   
      if self.verbose:
          print(f'Round {i} cost: {round_cost}')
      if i == 0:
        cost = round_cost
      if (round_cost <= cost and self.n_rounds == -1) or i < self.n_rounds:
        if cost > 0:
            cost = round_cost - self.cost_penalty*round_cost
        else:
            cost = round_cost + self.cost_penalty*round_cost
        self.total_trend = total_trend
        self.total_seasonalities = total_seasonalities
        self.total_predicted_exogenous = total_predicted_exogenous
        self.total_fitted_exogenous = total_fitted_exogenous
        self.total_predicted_trends = total_predicted_trends
        self.total_predicted_seasonalities = total_predicted_seasonalities
      else:
        assert i > 0, 'Boosting terminated before beginning'
        boost_ = False
        if self.verbose:
            print(f'Boosting Terminated \nUsing round {i-1}')           
        break    
    return self.build_output(i)

  def init_fit_values(self, time_series, forecast_horizon):
    self.time_series_index = time_series.index
    self.time_series = time_series.values
    self.forecast_horizon = forecast_horizon
    self.seasonal_period = [i if len(time_series) > i else 0 for i in self.seasonal_period]
    self.trends = []
    self.predicted_trends = []
    self.seasonalities = []
    self.predicted_seasonalities = []
    #self.errors = []
    self.predicted_exogenous = []
    self.fitted_exogenous = []
    self.boosted_data = self.time_series
    return

  def init_fit_classes(self):
    self.trend_class = FitTrend(
                            poly=self.poly,
                            trend_estimator=self.trend_estimator,
                            fit_type=self.fit_type,
                            given_splits=self.given_splits,
                            exclude_splits=self.exclude_splits,
                            min_sample_pct=self.min_sample_pct,
                            n_split_proposals=self.n_split_proposals,
                            approximate_splits=self.approximate_splits,
                            l2=self.l2,
                            arima_order = self.arima_order,
                            split_cost=self.split_cost,
                            trend_lr=self.trend_lr,
                            time_series_index=self.time_series_index,
                            forecast_horizon=self.forecast_horizon,
                            window_size=self.window_size,
                            smoothed=self.smoothed_trend,
                            )
    self.seasonal_class = FitSeasonality(
                                  seasonal_estimator=self.seasonal_estimator,
                                  seasonal_period=next(self.seasonality_cycle),
                                  fourier_components=self.fourier_components,
                                  forecast_horizon=self.forecast_horizon,
                                  seasonality_lr=self.seasonality_lr,
                                  additive=self.additive,
                                  seasonal_sample_weight=self.seasonal_sample_weight
                                  )
    if self.exogenous is not None:
        self.exo_class = FitExogenous(exo_estimator=self.exo_estimator,
                                 exogenous_lr=self.exogenous_lr,
                                 forecast_horizon = self.forecast_horizon)
    return

  def fit(self, time_series, forecast_horizon=0):
    time_series = pd.Series(time_series)
    assert not all([i == 0 for i in time_series]), 'All inputs are 0'
    time_series = self.scale_input(time_series)
    self.init_fit_values(time_series, forecast_horizon)
    self.seasonality_cycle = cycle(self.seasonal_period)
    self.init_fit_classes()
    output = self.additive_fit()
    return output
    
    
  def build_output(self, i):
    output = {}
    if self.trend_cap_target is not None:
            predicted_trend_perc = (self.total_predicted_trends[-1] - self.total_predicted_trends[0]) / self.total_predicted_trends[0]
            trend_change = self.trend_cap_target / predicted_trend_perc
            self.damp_factor = max(0, 1 - trend_change)
            
    if self.damp_factor:
            self.total_predicted_trends = self.trend_dampen(self.damp_factor, 
                                                            self.total_predicted_trends).values
    yhat = pd.Series(self.total_trend + 
                                self.total_seasonalities + self.total_fitted_exogenous, 
                                index = self.time_series_index).astype(float)

    if isinstance(self.time_series_index, pd.DatetimeIndex):
        freq = pd.infer_freq(self.time_series_index)
        if freq == 'M':
            forecast_end_date = self.time_series_index[-1] + pd.DateOffset(months = self.forecast_horizon)
        elif freq == 'D':
            forecast_end_date = self.time_series_index[-1] + pd.DateOffset(days = self.forecast_horizon)
        elif freq == 'Y':
            forecast_end_date = self.time_series_index[-1] + pd.DateOffset(years = self.forecast_horizon)
        full_index = pd.date_range(self.time_series_index[-1], forecast_end_date, freq = freq)
    else:
        full_index = np.arange(len(self.time_series_index) + self.forecast_horizon)
    if self.forecast_horizon:
        predictions = self.total_predicted_trends + self.total_predicted_seasonalities + self.total_predicted_exogenous
        predictions = pd.Series(predictions, index = full_index[-self.forecast_horizon:])
        self.total_predicted_trends = pd.Series(self.total_predicted_trends, index = full_index[-self.forecast_horizon:])
        self.total_predicted_seasonalities = pd.Series(self.total_predicted_seasonalities, index = full_index[-self.forecast_horizon:])
        self.total_predicted_exogenous = pd.Series(self.total_predicted_exogenous, index = full_index[-self.forecast_horizon:])
        
    trend = pd.Series(self.total_trend, index = self.time_series_index)
    seasonality = pd.Series(self.total_seasonalities, 
                                      index = self.time_series_index)
    exogenous = pd.Series(self.total_fitted_exogenous, index = self.time_series_index)
    self.seasonal_strength = self.calc_seasonal_strength(self.time_series - yhat,
                                                         self.time_series - trend)
    if self.seasonal_strength > 0 and self.seasonal_strength <= .15 and self.verbose:
        print('Seasonal Signal is weak, try a different frequency or disable seasonality with freq=0')
    self.trend_strength = self.calc_trend_strength(self.time_series - yhat,
                                      self.time_series - seasonality)
    
    
    upper_fitted, lower_fitted = self.get_fitted_intervals(self.time_series,
                                                                       yhat,
                                                                       c = self.c)
    if self.forecast_horizon:
        upper_prediction, lower_prediction = self.get_predicted_intervals(self.time_series,
                                                                           yhat,
                                                                           predictions,
                                                                           c = self.c)  
    else:
        upper_prediction, lower_prediction = 0, 0
    if self.exogenous is not None:
        output['exogenous'] = exogenous
        output['predicted_exogenous'] = self.total_predicted_exogenous
        # output['Exogenous Summary'] = self.get_boosted_exo_results(exo_impact)
        # self.exo_impact = exo_impact
    n = len(self.time_series)
    
    self.model_cost = (2*self.c**2 + 2*self.c)/max(1,(n-self.c-1)) + 2*(self.c**self.regularization) + \
                       n*np.log(np.sum((self.time_series - yhat )**2)/n) 
                       
    output['y'] = self.unscale_input(pd.Series(self.time_series, index = self.time_series_index))
    output['yhat'] = self.unscale_input(yhat)
    output['yhat_upper'] = self.unscale_input(upper_fitted)
    output['yhat_lower'] = self.unscale_input(lower_fitted)
    output['seasonality'] = self.unscale_input(seasonality)
    output['trend'] = self.unscale_input(trend)
    if self.forecast_horizon:
        output['predicted_trend'] = self.unscale_input(self.total_predicted_trends)
        output['predicted_seasonality'] = self.unscale_input(self.total_predicted_seasonalities)
        output['predicted'] = self.unscale_input(predictions)
        output['predicted_upper'] = self.unscale_input(upper_prediction)
        output['predicted_lower'] = self.unscale_input(lower_prediction)
    
    self.output = output
    self.number_of_rounds = i

    return output
  
  @staticmethod
  def trend_dampen(damp_fact, trend):
      zeroed_trend = trend - trend[0]
      damp_fact = 1 - damp_fact
      if damp_fact < 0:
        damp_fact = 0
      if damp_fact > 1:
        damp_fact = 1
      if damp_fact == 1:
        dampened_trend = zeroed_trend
      else:       
        tau = (damp_fact*1.15+(1.15*damp_fact/.85)**9)*\
                (2*len(zeroed_trend))
        dampened_trend = (zeroed_trend*np.exp(-pd.Series(range(1, len(zeroed_trend) + 1))/(tau)))
        crossing = np.where(np.diff(np.sign(np.gradient(dampened_trend))))[0]
        if crossing.size > 0:
            crossing_point = crossing[0]
            dampened_trend[crossing_point:] = dampened_trend[(np.mean(np.gradient(zeroed_trend))*dampened_trend).idxmax()]
      
      return dampened_trend + trend[0]
  

  def plot_components(self, figsize = (8,8)):
      summary_dict = self.output
      if 'exogenous' in summary_dict.keys():
          fig, ax = plt.subplots(4, figsize = figsize)
          ax[-2].plot(summary_dict['exogenous'], color = 'orange')
          ax[-2].set_title('Exogenous')
      else:
          fig, ax = plt.subplots(3, figsize = figsize)
      ax[0].plot(summary_dict['trend'], color = 'orange')
      ax[0].set_title('Trend')
      ax[1].plot(summary_dict['seasonality'], color = 'orange')
      ax[1].set_title('Seasonality')
      ax[-1].plot(summary_dict['y'], color = 'black')
      ax[-1].plot(summary_dict['yhat'], color = 'orange')
      ax[-1].plot(summary_dict['yhat_upper'], 
                 linestyle = 'dashed', 
                 alpha = .5,
                 color = 'orange')
      ax[-1].plot(summary_dict['yhat_lower'],
                 linestyle = 'dashed', 
                 alpha = .5,
                 color = 'orange')
      ax[-1].set_title('Fitted')
      
      plt.show()

  def plot_results(self, figsize = (6,4)):
      summary_dict = self.output
      fig, ax = plt.subplots(figsize = figsize)
      ax.plot(summary_dict['y'], color = 'black')
      ax.plot(summary_dict['yhat'], color = 'orange')
      ax.plot(summary_dict['yhat_upper'], 
                 linestyle = 'dashed', 
                 alpha = .5,
                 color = 'orange')
      ax.plot(summary_dict['yhat_lower'],
                 linestyle = 'dashed', 
                 alpha = .5,
                 color = 'orange')
      if self.forecast_horizon:
          ax.plot(summary_dict['yhat'].tail(1).append(summary_dict['predicted']), 
                  color= 'red', 
                  linestyle = 'dashed')        
          ax.fill_between(x = summary_dict['yhat_lower'].tail(1).append(summary_dict['predicted_lower']).index,
                          y1 = summary_dict['yhat_lower'].tail(1).append(summary_dict['predicted_lower']).values,
                          y2 = summary_dict['yhat_upper'].tail(1).append(summary_dict['predicted_upper']).values,
                          alpha = .5,
                          color = 'orange')
      ax.set_title('ThymeBoost Results')
      plt.show()
      
  def plot_rounds(self, figsize = (6,4)):
    if self.exogenous is not None:
        fig, ax = plt.subplots(3, figsize = figsize)
        for iteration in range(len(self.fitted_exogenous)-1):
            ax[2].plot(
                    np.sum(self.fitted_exogenous[:iteration], axis=0), 
                    label=iteration
                    )
        ax[2].set_title('Exogenous')
    else:
        fig, ax = plt.subplots(2, figsize = figsize)
    for iteration in range(len(self.trends)-1):
        ax[0].plot(np.sum(self.trends[:iteration], axis=0), label=iteration)
    ax[0].set_title('Trends')
    for iteration in range(len(self.seasonalities)-1):
        ax[1].plot(np.sum(self.seasonalities[:iteration], axis=0), label=iteration)
    ax[1].set_title('Seasonalities')
    plt.legend()
    plt.show()
    

