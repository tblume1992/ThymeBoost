# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


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
        tau = (damp_fact * 1.15 + (1.15 * damp_fact / .85)**9) *\
                (2*len(zeroed_trend))
        dampened_trend = (zeroed_trend*np.exp(-pd.Series(range(1, len(zeroed_trend) + 1))/(tau)))
        crossing = np.where(np.diff(np.sign(np.gradient(dampened_trend))))[0]
        if crossing.size > 0:
            crossing_point = crossing[0]
            avg_grad = (np.mean(np.gradient(zeroed_trend))*dampened_trend)
            dampened_trend[crossing_point:] = dampened_trend[avg_grad.idxmax()]
    return dampened_trend + trend[0]
