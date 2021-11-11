# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import stats
from ThymeBoost.utils.trend_dampen import trend_dampen


class BuildOutput:

    def __init__(self,
                 time_series,
                 time_series_index,
                 scaler_obj,
                 c):
        self.time_series = time_series
        self.time_series_index = time_series_index
        self.scaler_obj = scaler_obj
        self.c = c

    def handle_future_index(self, forecast_horizon):
        if isinstance(self.time_series_index, pd.DatetimeIndex):
            last_date = self.time_series_index[-1]
            freq = pd.infer_freq(self.time_series_index)
            future_index = pd.date_range(last_date,
                                         periods=forecast_horizon + 1,
                                         freq=freq)[1:]
        else:
            future_index = np.arange(len(self.time_series_index) + forecast_horizon)
            future_index = future_index[-forecast_horizon:]
        return future_index

    @staticmethod
    def get_fitted_intervals(y, fitted, c):
        """
        A interval calculation based on lienar regression, only useful for non-smoother/state space/local models.
        TODO: Look for generalized approach, possibly using rolling predicted residuals?

        Parameters
        ----------
        y : TYPE
            DESCRIPTION.
        fitted : TYPE
            DESCRIPTION.
        c : TYPE
            DESCRIPTION.

        Returns
        -------
        upper : TYPE
            DESCRIPTION.
        lower : TYPE
            DESCRIPTION.
        """
        sd_error = np.std(y - fitted)
        t_stat = stats.t.ppf(.9, len(y))
        upper = fitted + t_stat*sd_error
        lower = fitted - t_stat*sd_error
        return upper, lower

    @staticmethod
    def get_predicted_intervals(y, fitted, predicted, c):
        """
        A interval calculation based on linear regression with forecast penalty,
        only semi-useful for non-smoother/state space/local models.
        TODO: Look for generalized approach, possibly using rolling predicted residuals?

        Parameters
        ----------
        y : TYPE
            DESCRIPTION.
        fitted : TYPE
            DESCRIPTION.
        c : TYPE
            DESCRIPTION.

        Returns
        -------
        upper : TYPE
            DESCRIPTION.
        lower : TYPE
            DESCRIPTION.

        """
        sd_error = np.std(y - fitted)
        t_stat = stats.t.ppf(.9, len(y))
        len_frac = len(predicted)/len(fitted)
        interval_uncertainty_param = np.linspace(1.0,
                                                 1.0 + 3*len_frac,
                                                 len(predicted))
        upper = predicted + t_stat*sd_error*interval_uncertainty_param
        lower = predicted - t_stat*sd_error*interval_uncertainty_param
        return upper, lower

    def build_fitted_df(self,
                        trend,
                        seasonality,
                        exogenous):
        time_series = self.scaler_obj(pd.Series(self.time_series))
        output = pd.DataFrame(time_series.values,
                              index=self.time_series_index,
                              columns=['y'])
        yhat = trend + seasonality
        if exogenous is not None:
            yhat += exogenous
        upper_fitted, lower_fitted = self.get_fitted_intervals(self.time_series,
                                                               yhat,
                                                               c=self.c)
        if exogenous is not None:
            output['exogenous'] = exogenous
            # output['Exogenous Summary'] = self.get_boosted_exo_results(exo_impact)
            # self.exo_impact = exo_impact
        output['yhat'] = self.scaler_obj(yhat)
        output['yhat_upper'] = self.scaler_obj(upper_fitted)
        output['yhat_lower'] = self.scaler_obj(lower_fitted)
        output['seasonality'] = self.scaler_obj(seasonality)
        output['trend'] = self.scaler_obj(trend)
        return output

    def build_predicted_df(self,
                           fitted_output,
                           forecast_horizon,
                           trend,
                           seasonality,
                           exogenous,
                           predictions,
                           trend_cap_target,
                           damp_factor):
        if trend_cap_target is not None:
            predicted_trend_perc = (trend[-1] - trend[0]) / trend[0]
            trend_change = trend_cap_target / predicted_trend_perc
            damp_factor = max(0, 1 - trend_change)
        if damp_factor is not None:
            trend = trend_dampen(damp_factor,
                                 trend).values
        future_index = self.handle_future_index(forecast_horizon)
        predicted_output = pd.DataFrame(self.scaler_obj(predictions),
                                        index=future_index,
                                        columns=['predictions'])
        bounds = self.get_predicted_intervals(self.time_series,
                                              fitted_output['yhat'],
                                              self.scaler_obj(predictions),
                                              c=self.c)
        upper_prediction, lower_prediction = bounds
        predicted_output['predicted_trend'] = self.scaler_obj(trend)
        predicted_output['predicted_seasonality'] = self.scaler_obj(seasonality)
        if exogenous is not None:
            predicted_output['predicted_exogenous'] = self.scaler_obj(exogenous)
        predicted_output['predicted_upper'] = upper_prediction
        predicted_output['predicted_lower'] = lower_prediction
        return predicted_output
