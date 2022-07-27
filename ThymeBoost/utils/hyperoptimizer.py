# -*- coding: utf-8 -*-

# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_squared_error
# import time
# from hyperopt import space_eval
# from sklearn.model_selection import cross_val_score
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# from hyperopt.pyll import scope
# from hyperopt.fmin import fmin
# import numpy as np


# class Optimize:
#     def __init__(self,
#                  y,
#                  seasonal_period=0,
#                  n_folds=3,
#                  test_size=None):
#         self.y = y
#         self.seasonal_period = seasonal_period
#         self.n_folds = n_folds
#         self.test_size = test_size

#     def logic_layer(self):
#         time_series = pd.Series(self.y).copy(deep=True)
#         if isinstance(self.seasonal_period, list):
#             generator_seasonality = True
#         else:
#             generator_seasonality = False
#         if generator_seasonality:
#             max_seasonal_pulse = self.seasonal_period[0]
#         else:
#             max_seasonal_pulse = self.seasonal_period
#         if not generator_seasonality:
#             self.seasonal_period = [self.seasonal_period]
#         if self.seasonal_period[0]:
#             self.seasonal_period = [0, self.seasonal_period, self.seasonal_period.append(0)]

#         _contains_zero = not (time_series > 0).all()
#         if _contains_zero:
#             self.additive = [True]
#         else:
#             self.additive = [True, False]

#         if len(time_series) > 2.5 * max_seasonal_pulse and max_seasonal_pulse:
#             seasonal_sample_weights = []
#             weight = 1
#             for i in range(len(y)):
#                 if (i) % max_seasonal_pulse == 0:
#                     weight += 1
#                 seasonal_sample_weights.append(weight)
#             self.seasonal_sample_weights = [None,
#                                                  np.array(seasonal_sample_weights)]

#     def get_space(self):
#         self.space = {
#             'trend_estimator': hp.choice('trends', [{'trend_estimator': 'linear',
#                                                      'poly': hp.choice('exp', [1, 2]),
#                                                      'arima_order': 'auto',
#                                                      'fit_type': hp.choice('cp1',
#                                                                            ['local', 'global']),
#                                                      },
#                                                     {'trend_estimator': 'arima',
#                                                       'poly': 1,
#                                                       'arima_order': 'auto',
#                                                       'fit_type': 'global',
#                                                       },
#                                                     {'trend_estimator': 'naive',
#                                                       'poly': 1,
#                                                       'arima_order': 'auto',
#                                                       'fit_type': 'global',
#                                                       },
#                                                     {'trend_estimator': 'ses',
#                                                      'poly': 1,
#                                                      'arima_order': 'auto',
#                                                      'fit_type': 'global'
#                                                      },
#                                                     {'trend_estimator': ['linear', 'ses'],
#                                                      'poly': 1,
#                                                      'arima_order': 'auto',
#                                                      'fit_type': hp.choice('cp2',
#                                                                            [['global'], ['local', 'global']])
#                                                      },
#                                                     {'trend_estimator': 'mean',
#                                                      'poly': 1,
#                                                      'arima_order': 'auto',
#                                                      'fit_type': hp.choice('cp3',
#                                                                            ['global', 'local'])
#                                                      }]),
#             'global_cost': hp.choice('gcost', ['mse', 'maicc']),
#             'additive': hp.choice('add', self.additive),
#             'seasonal_estimator': hp.choice('seas', ['fourier', 'naive']),
#             'seasonal_period': hp.choice('seas_period', self.seasonal_period)
#         }
#         return self.space

#     def scorer(self, model_obj, y, metric, cv, params):
#         cv_splits = cv.split(y)
#         mses = []
#         for train_index, test_index in cv_splits:
#             # print(np.shape(y[train_index]))
#             fitted = model_obj.fit(y[train_index], **params)
#             predicted = model_obj.predict(fitted, len(y[test_index]))
#             predicted = predicted['predictions'].values
#             mses.append(mean_squared_error(y[test_index], predicted))
#         return_dict = {'loss': np.mean(mses),
#                        'eval_time': time.time(),
#                        'status': STATUS_OK}
#         return return_dict


#     def objective(self, params):
#         # print(params)
#         if isinstance(params['trend_estimator']['trend_estimator'], tuple):
#             params['trend_estimator']['trend_estimator'] = list(params['trend_estimator']['trend_estimator'])
#         if isinstance(params['seasonal_period'], tuple):
#             params['seasonal_period'] = list(params['seasonal_period'])
#         if isinstance(params['trend_estimator']['fit_type'], tuple):
#             params['trend_estimator']['fit_type'] = list(params['trend_estimator']['fit_type'])
#         params = {
#             'trend_estimator': params['trend_estimator']['trend_estimator'],
#             'fit_type': params['trend_estimator']['fit_type'],
#             'arima_order': params['trend_estimator']['arima_order'],
#             'poly': params['trend_estimator']['poly'],
#             'seasonal_period': params['seasonal_period'],
#             'additive': params['additive'],
#             'global_cost': params['global_cost'],

#         }
#         # print(params)
#         clf = ThymeBoost()
#         # score = cross_val_score(clf, self.y, self.y, scoring=mean_squared_error, cv=TimeSeriesSplit(self.n_folds)).mean()
#         score = self.scorer(clf,
#                             self.y,
#                             mean_squared_error,
#                             TimeSeriesSplit(self.n_folds, test_size=self.test_size),
#                             params)
#         # print(f"MSE {score} params {params}")
#         return score

#     def fit(self):
#         self.logic_layer()
#         # trials = Trials()
#         best = fmin(fn=self.objective,
#                     space=self.get_space(),
#                     algo=tpe.suggest,
#                     max_evals=100,
#                     # early_stop_fn=fn,
#                     verbose=False)
#         # print(best)
#         return best