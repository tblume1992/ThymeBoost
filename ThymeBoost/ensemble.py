# -*- coding: utf-8 -*-

import traceback
import itertools as it
import pandas as pd
from tqdm import tqdm
from ThymeBoost.param_iterator import ParamIterator


class Ensemble(ParamIterator):
    __framework__ = 'ensemble'

    def __init__(self,
                 model_object,
                 y,
                 verbose=0,
                 **kwargs
                 ):
        self.model_object = model_object
        self.y = y
        self.verbose = verbose
        self.kwargs = kwargs
        self.settings_keys = self.kwargs.keys()

    def get_space(self):
        ensemble_space = list(it.product(*self.kwargs.values()))
        run_settings = []
        for params in ensemble_space:
            run_settings.append(dict(zip(self.settings_keys, params)))
        cleaned_space = self.sanitize_params(run_settings)
        return cleaned_space

    def ensemble_fit(self):
        parameters = self.get_space()
        ensemble_parameters = []
        if self.verbose:
            param_iters = tqdm(parameters)
        else:
            param_iters = parameters
        outputs = []
        for run_settings in param_iters:
            y_copy = self.y.copy(deep=True)
            try:
                output = self.model_object.fit(y_copy, **run_settings)
                #key = ','.join(map(str, run_settings.values()))
                outputs.append(output)
                ensemble_parameters.append(self.model_object.booster_obj)
            except Exception as e:
                print(f'{e} Error running settings: {run_settings}')
                traceback.print_exc()
        output = pd.concat(outputs)
        output = output.groupby(by=output.index).mean()
        return output, ensemble_parameters
