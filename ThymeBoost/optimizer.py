# -*- coding: utf-8 -*
#TODO: fix exogenous looping.
import types
import traceback
import itertools as it
import pandas as pd
import copy
from tqdm import tqdm
import numpy as np
from ThymeBoost.cost_functions import calc_smape, calc_mape, calc_mse, calc_mae
from ThymeBoost.param_iterator import ParamIterator
import ThymeBoost


class Optimizer(ParamIterator):
    __framework__ = 'optimizer'

    def __init__(self,
                 model_object,
                 y,
                 optimization_type,
                 optimization_strategy,
                 optimization_steps,
                 lag,
                 optimization_metric,
                 test_set,
                 verbose,
                 **kwargs
                 ):
        self.verbose = verbose
        self.optimization_type = optimization_type
        self.optimization_strategy = optimization_strategy
        self.lag = lag
        self.optimization_metric = optimization_metric
        if self.optimization_strategy == 'holdout':
            optimization_steps = 1
        self.optimization_steps = optimization_steps
        self.y = y
        self.model_object = model_object
        self.test_set = test_set
        self.search_space = {'trend_estimator': ['mean', 'median', 'linear'],
                             'fit_type': ['local', 'global'],
                             'seasonal_period': [None]
                             }
        self.search_space.update(kwargs)
        self.search_keys = self.search_space.keys()

    def set_optimization_metric(self):
        if self.optimization_metric == 'smape':
            self.optimization_metric = calc_smape
        if self.optimization_metric == 'mape':
            self.optimization_metric = calc_mape
        if self.optimization_metric == 'mse':
            self.optimization_metric = calc_mse
        if self.optimization_metric == 'mae':
            self.optimization_metric = calc_mae
        return

    def get_search_space(self):
        thymeboost_search_space = list(it.product(*self.search_space.values()))
        run_settings = []
        for params in thymeboost_search_space:
            run_settings.append(dict(zip(self.search_keys, params)))
        cleaned_space = self.sanitize_params(run_settings)
        return cleaned_space

    @staticmethod
    def combiner_check(param_dict, wrap_values=True):
        ensemble_dict = {}
        if any(isinstance(v, types.FunctionType)
               for v in list(param_dict.values())):
            ensemble = True
            for k, v in param_dict.items():
                if isinstance(v, types.FunctionType):
                    ensemble_dict[k] = v
                    param_dict[k] = v()
                else:
                    if wrap_values:
                        param_dict[k] = [v]
        else:
            ensemble = False
        return ensemble, ensemble_dict

    @staticmethod
    def exo_check(param_dict, ensemble):
        exo = False
        if ensemble:
            if param_dict['exogenous'][0] is not None:
                exo = True
        else:
            if param_dict['exogenous'] is not None:
                exo = True
        return exo

    def fit(self):
        #This needs to be refactored
        self.parameters = self.get_search_space()
        self.set_optimization_metric()
        results = {}
        for num_steps in range(1, self.optimization_steps + 1):
            y_copy = self.y.copy(deep=True)
            if self.optimization_strategy == 'cv':
                test_y = y_copy[-self.lag * num_steps + 1:]
                train_y = y_copy[:-self.lag * num_steps + 1]
                test_y = test_y[:self.lag]
            else:
                test_y = y_copy[-self.lag - num_steps + 1:]
                train_y = y_copy[:-self.lag - num_steps + 1]
                test_y = test_y[:self.lag]
            results[str(num_steps)] = {}
            if self.verbose:
                param_iters = tqdm(self.parameters)
            else:
                param_iters = self.parameters
            for settings in param_iters:
                # try:
                    run_settings = copy.deepcopy(settings)
                    ensemble, ensemble_dict = Optimizer.combiner_check(run_settings)
                    exo = Optimizer.exo_check(run_settings, ensemble)
                    if exo:
                        if ensemble:
                            X_test = run_settings['exogenous'][0].loc[test_y.index]
                            X_train = run_settings['exogenous'][0].iloc[:len(train_y), :]
                            params = copy.deepcopy(run_settings)
                            run_settings['exogenous'] = [X_train]
                            output = self.model_object.ensemble(train_y,
                                                                **run_settings)
                        else:
                            X_test = run_settings['exogenous'].loc[test_y.index]
                            X_train = run_settings['exogenous'].iloc[:len(train_y), :]
                            params = copy.deepcopy(run_settings)
                            run_settings['exogenous'] = X_train
                            output = self.model_object.fit(train_y,
                                                           **run_settings)
                        predicted_output = self.model_object.predict(output,
                                                                     self.lag,
                                                                     future_exogenous=X_test)
                        run_settings.pop('exogenous')
                    else:
                        if ensemble:
                            output = self.model_object.ensemble(train_y,
                                                                **run_settings)
                        else:
                            output = self.model_object.fit(train_y, **run_settings)
                        predicted_output = self.model_object.predict(output,
                                                                     self.lag)

                        params = copy.deepcopy(run_settings)
                    predicted = predicted_output['predictions']
                    if self.test_set == 'all':
                        test_error = self.optimization_metric(actuals=test_y,
                                                              predicted=predicted)
                    elif self.test_set == 'last':
                        test_error = self.optimization_metric(actuals=test_y.iloc[-1],
                                                              predicted=predicted.iloc[-1])
                    key = ','.join(map(str, run_settings.values()))
                    results[str(num_steps)][key] = {}
                    results[str(num_steps)][key]['error'] = test_error
                    params.update(ensemble_dict)
                    results[str(num_steps)][key]['params'] = params
                    results[str(num_steps)][key]['predictions'] = predicted
                # except Exception as e:
                #     results[str(num_steps)][','.join(map(str, run_settings))] = np.inf
                #     print(f'{e} Error running settings: {run_settings}')
                #     traceback.print_exc()
        return results

    def optimize(self):
        opt_results = self.fit()
        average_result = {}
        for key in opt_results['1'].keys():
            summation = 0
            for step in opt_results.keys():
                summation += opt_results[step][key]['error']
            average_result[key] = summation / len(opt_results.keys())
        average_result = pd.Series(average_result)
        average_result = average_result.sort_values()
        best_setting = average_result.index[0]
        self.run_settings = opt_results['1'][best_setting]['params']
        self.cv_predictions = []
        for k, v in opt_results.items():
            self.cv_predictions.append(opt_results[k][best_setting]['predictions'])
        ensemble, _ = Optimizer.combiner_check(self.run_settings, wrap_values=False)
        if ensemble:
            output = self.model_object.ensemble(self.y, **self.run_settings)
        else:
            output = self.model_object.fit(self.y, **self.run_settings)
        if self.verbose:
            print(f'Optimal model configuration: {self.run_settings}')
            print(f'Params ensembled: {ensemble}')
        return output



