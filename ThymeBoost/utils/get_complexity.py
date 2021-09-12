# -*- coding: utf-8 -*-

import numpy as np


def get_complexity(boosting_round,
                   poly,
                   fit_type,
                   trend_estimator,
                   arima_order,
                   window_size,
                   time_series,
                   fourier_order,
                   seasonal_period,
                   exogenous):
    #Get a measure of complexity: number of splits + any extra variables
    if fit_type == 'global' and trend_estimator == 'linear':
        c = 1
    elif fit_type == 'global' and trend_estimator == 'loess':
        c = int(len(time_series) / window_size)
    elif fit_type == 'global' and trend_estimator == 'ar':
        c = np.sum(arima_order)
    else:
        if seasonal_period != 0 and not None:
            c = boosting_round + fourier_order
        else:
            c = boosting_round
        if trend_estimator == 'linear' and fit_type == 'local':
            c = poly + fourier_order + boosting_round
        if exogenous is not None:
            c += np.shape(exogenous)[1]
    return c
