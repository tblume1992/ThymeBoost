# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import entropy


def get_split_cost(time_series, split1, split2, split_cost):
    if split_cost == 'mse':
        cost = np.mean((time_series - np.append(split1, split2))**2)
    elif split_cost == 'mae':
        cost = np.mean(np.abs(time_series - np.append(split1, split2)))
    elif split_cost == 'entropy':
        residuals = time_series - np.append(split1, split2)
        entropy_safe_residuals = residuals + min(residuals) + 1
        cost = entropy(entropy_safe_residuals)
    else:
        raise ValueError('That split cost is not recognized')
    return cost


def calc_cost(time_series, prediction, c, regularization, global_cost):
    n = len(time_series)
    if global_cost == 'maic':
        cost = 2*(c**regularization) + \
              n*np.log(np.sum((time_series - prediction)**2)/n)
    elif global_cost == 'maicc':
        cost = (2*c**2 + 2*c)/max(1, (n-c-1)) + 2*(c**regularization) + \
               n*np.log(np.sum((time_series - prediction)**2)/n)
    elif global_cost == 'mbic':
        cost = n*np.log(np.sum((time_series - prediction)**2)/n) + \
               (c**regularization) * np.log(n)
    elif global_cost == 'mse':
        cost = np.mean((time_series - prediction)**2)
    else:
        raise ValueError('That global cost is not recognized')
    if np.isinf(cost):
        cost = 0
    return cost


def calc_smape(actuals, predicted):
    return 100 * (2 * np.sum(np.abs(predicted - actuals)) / np.sum((np.abs(actuals) + np.abs(predicted))))


def calc_mape(actuals, predicted, epsilon=.00000001):
    return np.sum(np.abs(predicted - actuals)) / (epsilon + np.sum((np.abs(actuals))))


def calc_mse(actuals, predicted):
    return np.mean((predicted - actuals)**2)


def calc_mae(actuals, predicted):
    return np.mean(np.abs(predicted - actuals))


def calc_entropy(actuals, predicted):
    residuals = predicted - actuals
    #Don't remember what this is doing
    entropy_safe_residuals = residuals + min(residuals) + 1
    return entropy(entropy_safe_residuals)
