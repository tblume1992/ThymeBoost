# -*- coding: utf-8 -*-

import numpy as np

def get_split_cost(time_series, split1, split2, split_cost):
  if split_cost == 'mse':
    cost = np.mean((time_series - np.append(split1,split2))**2)
  elif split_cost == 'mae':
    cost = np.mean(np.abs(time_series - np.append(split1,split2)))
  return cost

def calc_cost(time_series, prediction, c, regularization, global_cost):
  n = len(time_series)
  if global_cost == 'maic':
    cost = 2*(c**regularization) + n*np.log(np.sum((time_series - prediction )**2)/n)
  if global_cost == 'maicc':
    cost = (2*c**2 + 2*c)/max(1,(n-c-1)) + 2*(c**regularization) + \
            n*np.log(np.sum((time_series - prediction )**2)/n)    
  elif global_cost == 'mbic':
    cost = n*np.log(np.sum((time_series - prediction )**2)/n) + \
          (c**regularization) * np.log(n)
  elif global_cost == 'mse':
      cost = np.mean((time_series - prediction)**2)
  return cost

def calc_smape(A, F):
    return 100 * (2 * np.sum(np.abs(F - A)) / np.sum((np.abs(A) + np.abs(F))))

def calc_wfa(A, F, epsilon = .00000001):
    return (np.sum(np.abs(A - F)))/(epsilon + np.sum(F + A))

def calc_mape(A, F, epsilon = .00000001):
    return np.sum(np.abs(F - A)) / (epsilon + np.sum((np.abs(A))))

def calc_mse(A, F):
    return np.mean((F - A)**2)

def calc_mae(A, F):
    return np.mean(np.abs(F - A))