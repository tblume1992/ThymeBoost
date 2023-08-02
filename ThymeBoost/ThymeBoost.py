# -*- coding: utf-8 -*-
r"""
ThymeBoost combines time series decomposition with gradient boosting to 
provide a flexible mix-and-match time series framework. At the most granular 
level are the trend/level (going forward this is just referred to as 'trend') 
models, seasonal models,  and exogenous models. These are used to approximate 
the respective components at each 'boosting round'. Concurrent rounds are 
fit on residuals in usual boosting fashion.

The boosting procedure is heavily influenced by traditional boosting theory [1]_ 
where the initial round's trend estimation is a simple median, although this 
can be changed to other similarly 'weak' trend estimation methods. Each round 
involves approximating each component and passing the 'psuedo' residuals to 
the next boosting round.

Gradient boosting allows us to use a single procedure to mix-and-match different 
types of models. A common question when decomposing time series is the order of 
decomposition. Some methods require approximating trend after seasonality or vice 
versa due to a underlying model eating too much of the other's component. This can 
be overcome by using the respective component's learning rate parameter to penalize 
it at each round.

References
----------
.. [1] Jerome H. Friedman. 2002. Stochastic gradient boosting. Comput. Stat. Data Anal. 38, 4
   (28 February 2002), 367–378. DOI:https://doi.org/10.1016/S0167-9473(01)00065-2

"""

from itertools import cycle
import warnings
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from ThymeBoost.fitter.booster import booster
from ThymeBoost.optimizer import Optimizer
from ThymeBoost.ensemble import Ensemble
from ThymeBoost.utils import plotting
from ThymeBoost.utils import build_output
from ThymeBoost.predict_functions import predict_rounds
warnings.filterwarnings("ignore")


class ThymeBoost:
    """
    ThymeBoost main class which wraps around the optimizer and ensemble classes.

    Parameters
    ----------
    verbose : bool, optional
        Truey/Falsey for printing some info, logging TODO. The default is 0.
    n_split_proposals : int, optional
        Number of splits to propose for changepoints. If fit_type in the fit 
        method is 'global' then this parameter is ignored. The default is 10.
    approximate_splits : bool, optional
        Whether to reduce the search space of changepoints. If False then the 
        booster will exhaustively try each point as a changepoint. If fit_type 
        in the fit method is 'global' then this parameter is ignored.
        The default is True.
    exclude_splits : list, optional
        List of indices to exclude when searching for changepoints. The default is None.
    given_splits : list, optional
        List of indices to use when searching for changepoints. The default is None.
    cost_penalty : float, optional
        A penalty which is applied at each boosting round. This keeps the 
        booster from making too many miniscule improvements. 
        The default is .001.
    normalize_seasonality : bool, optional
        Whether to enforce seasonality average to be 0 for add or 1 for mult. 
        The default is True.
    additive: bool, optional
        FIXME
        Whether the whole process is additive or multiplicative. Definitely 
        unstable in certain parameter combinations. User beware! 
    regularization : float, optional
        A parameter which further penalizes the global cost at each boosting round. 
        The default is 1.2.
    n_rounds : int, optional
        A set number of boosting rounds until termination. If not set then 
        boosting terminates when the current round does not improve enough over 
        the previous round. The default is None.
    smoothed_trend : bool, optional
        FIXME
        Whether to apply some smoothing to the trend compoennt. 
        The default is False.
    scale_type : str, optional
        FIXME
        The type of scaling to apply. Options are ['standard', 'log'] for classic
        standardization or taking the log. The default is None.

    """
    __framework__ = 'main'
    version = '0.1.15'
    author = 'Tyler Blume'

    def __init__(self,
                 verbose=0,
                 n_split_proposals=10,
                 approximate_splits=True,
                 exclude_splits=None,
                 given_splits=None,
                 cost_penalty=.001,
                 normalize_seasonality=True,
                 regularization=1.2,
                 n_rounds=None,
                 smoothed_trend=False,
                 scale_type=None,
                 split_strategy='gradient',
                 error_handle='raise'):
        self.verbose = verbose
        self.n_split_proposals = n_split_proposals
        self.approximate_splits = approximate_splits
        if exclude_splits is None:
            exclude_splits = []
        self.exclude_splits = exclude_splits
        if given_splits is None:
            given_splits = []
        self.given_splits = given_splits
        self.cost_penalty = cost_penalty
        self.scale_type = scale_type
        self.normalize_seasonality = normalize_seasonality
        self.regularization = regularization
        if n_rounds is None:
            n_rounds = -1
        self.n_rounds = n_rounds
        self.smoothed_trend = smoothed_trend
        self.trend_cap_target = None
        self.ensemble_boosters = None
        self._online_learning_ignore_params = None
        self.online_learning = None
        self.split_strategy = split_strategy
        self.error_handle = error_handle

    def scale_input(self, time_series):
        """
        Simple scaler method to scale and unscale the time series.  
        Used if 'additive' is False.

        Parameters
        ----------
        time_series : pd.Series
            The time series to scale.

        Raises
        ------
        ValueError
            Thrown if unsupport scale type is provided.

        Returns
        -------
        time_series : pd.Series
            Scale time series.

        """
        #FIXME
        #seems unstable with different combinations
        if self.scale_type == 'standard':
            self.time_series_mean = time_series.mean()
            self.time_series_std = time_series.std()
            time_series = (time_series - self.time_series_mean) / \
                           self.time_series_std
        elif self.scale_type == 'log':
            if self.error_handle == 'raise':
                assert time_series.all(), 'Series can not contain 0 for mult. fit or log scaling'
                assert (time_series > 0).all(), 'Series can not contain neg. values for mult. fit or log scaling'
                time_series = np.log(time_series)
            elif self.error_handle == 'warn':
                if not (time_series > 0).all():
                    print('Series can not contain neg. values or 0 for multiplicative fit or log scaling')
                    self.scale_type = None
            else:
                if not (time_series > 0).all():
                    self.scale_type = None
        elif self.scale_type is None:
            pass
        else:
            raise ValueError('Scaler not recognized!')
        return time_series

    def unscale_input(self, scaled_series):
        """
        Unscale the time series to return it to the OG scale.

        Parameters
        ----------
        scaled_series : pd.Series
            The previously scaled sereis that needs to be rescaled before returning.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.scale_type == 'standard':
            return scaled_series * self.time_series_std + self.time_series_mean
        elif self.scale_type == 'log':
            return np.exp(scaled_series)
        else:
            return scaled_series

    @staticmethod
    def create_iterated_features(variable):
        """
        This function creates the 'generator' features which are cycled through each boosting round.

        Parameters
        ----------
        variable : TYPE
            The variable to convert into a 'generator' feature.

        Returns
        -------
        variable : it.cycle
            The 'generator' to cycle through.

        """
        if not isinstance(variable, list):
            variable = [variable]
        variable = cycle(variable)
        return variable

    @staticmethod
    def combine(param_list: list):
        """
        A function used to denote ensembled parameters for optimization

        Parameters
        ----------
        param_list : list
            list of param values to ensemble.

        Returns
        -------
        funcion
            Returns a function to call and pass to ensembling.

        """
        def combiner():
            return param_list
        return combiner

    def get_changepoints(self):
        self.changepoints = [i.split for i in self.booster_obj.trend_objs]
        return

    def fit(self,
            time_series,
            seasonal_period=0,
            trend_estimator='linear',
            seasonal_estimator='fourier',
            exogenous_estimator='ols',
            l2=None,
            poly=1,
            arima_order=(1, 0, 1),
            arima_trend=None,
            connectivity_constraint=True,
            fourier_order=10,
            fit_type='global',
            window_size=3,
            trend_weights=None,
            seasonality_weights=None,
            trend_lr=1,
            seasonality_lr=1,
            exogenous_lr=1,
            min_sample_pct=.01,
            split_cost='mse',
            global_cost='maicc',
            exogenous=None,
            damp_factor=None,
            ewm_alpha=.5,
            alpha=None,
            beta=None,
            ransac_trials=100,
            ransac_min_samples=10,
            tree_depth=1,
            n_changepoints=3,
            additive=True,
            init_trend='median'):
        self.additive = additive
        if seasonal_period is None:
            seasonal_period = 0
        if additive:
            self.scale_type = None
        else:
            self.scale_type = 'log'
        #grab all variables to create 'generator' variables
        _params = locals()
        self._params = _params
        _params.pop('self', None)
        _params.pop('time_series', None)
        _params.pop('additive', None)
        _params = {k: ThymeBoost.create_iterated_features(v) for k, v in _params.items()}
        _params['additive'] = additive
        time_series = pd.Series(time_series).copy(deep=True)
        self.time_series_index = time_series.index
        self.time_series = time_series.values
        assert not all([i == 0 for i in time_series]), 'All inputs are 0'
        assert len(time_series) > 1, 'ThymeBoost requires at least 2 data points'
        time_series = self.scale_input(time_series)
        self.booster_obj = booster(time_series=time_series,
                                   given_splits=self.given_splits,
                                   verbose=self.verbose,
                                   n_split_proposals=self.n_split_proposals,
                                   approximate_splits=self.approximate_splits,
                                   exclude_splits=self.exclude_splits,
                                   cost_penalty=self.cost_penalty,
                                   normalize_seasonality=self.normalize_seasonality,
                                   regularization=self.regularization,
                                   n_rounds=self.n_rounds,
                                   smoothed_trend=self.smoothed_trend,
                                   split_strategy=self.split_strategy,
                                   **_params)
        booster_results = self.booster_obj.boost()
        fitted_trend = booster_results[0]
        fitted_seasonality = booster_results[1]
        fitted_exogenous = booster_results[2]
        self.c = self.booster_obj.c
        self.builder = build_output.BuildOutput(time_series,
                                                self.time_series_index,
                                                self.unscale_input,
                                                self.c)
        output = self.builder.build_fitted_df(fitted_trend,
                                              fitted_seasonality,
                                              fitted_exogenous)
        #ensure we do not fall into ensemble prediction for normal fit
        self.ensemble_boosters = None
        self.get_changepoints()
        return output

    def predict(self,
                fitted_output,
                forecast_horizon,
                future_exogenous=None,
                damp_factor=None,
                trend_cap_target=None,
                trend_penalty=None,
                uncertainty=True) -> pd.DataFrame:
        """
        ThymeBoost predict method which uses the booster to generate 
        predictions that are a sum of each component's round.

        Parameters
        ----------
        fitted_output : pd.DataFrame
            The output from the ThymeBoost.fit method.
        forecast_horizon : int
            The number of periods to forecast.
        damp_factor : float, optional
            Damp factor to apply, constrained to (0, 1) where .5 is 50% of the 
            current predicted trend.
            The default is None.
        trend_cap_target : float, optional
            Instead of a predetermined damp_factor, this will only dampen the 
            trend to a certain % growth if it exceeds that growth. 
            The default is None.

        Returns
        -------
        predicted_output : pd.DataFrame
            The predicted output dataframe.

        """
        if future_exogenous is not None:
            assert len(future_exogenous) == forecast_horizon, 'Given future exogenous len not equal to forecast horizon'
        if self.online_learning is not None:
            self._online_learning_ignore_params = {'scale_type': self.scale_type}
        if self._online_learning_ignore_params is not None:
            self.scale_type = self._online_learning_ignore_params['scale_type']
        if self.ensemble_boosters is None:
            trend, seas, exo, predictions = predict_rounds(self.booster_obj,
                                                           forecast_horizon,
                                                           trend_penalty,
                                                           future_exogenous,
                                                           self.online_learning
                                                           )
            fitted_output = copy.deepcopy(fitted_output)
            predicted_output = self.builder.build_predicted_df(fitted_output,
                                                               forecast_horizon,
                                                               trend,
                                                               seas,
                                                               exo,
                                                               predictions,
                                                               trend_cap_target,
                                                               damp_factor,
                                                               uncertainty)
        else:
            ensemble_predictions = []
            for booster_obj in self.ensemble_boosters:
                self.booster_obj = booster_obj
                trend, seas, exo, predictions = predict_rounds(self.booster_obj,
                                                               forecast_horizon,
                                                               trend_penalty,
                                                               future_exogenous,
                                                               )
                fitted_output = copy.deepcopy(fitted_output)
                predicted_output = self.builder.build_predicted_df(fitted_output,
                                                                   forecast_horizon,
                                                                   trend,
                                                                   seas,
                                                                   exo,
                                                                   predictions,
                                                                   trend_cap_target,
                                                                   damp_factor,
                                                                   uncertainty)
                ensemble_predictions.append(predicted_output)
            predicted_output = pd.concat(ensemble_predictions)
            predicted_output = predicted_output.groupby(predicted_output.index).mean()
        return predicted_output

    def optimize(self,
                 time_series,
                 optimization_type='grid_search',
                 optimization_strategy='rolling',
                 optimization_steps=3,
                 lag=2,
                 optimization_metric='smape',
                 test_set='all',
                 verbose=1,
                 **kwargs):
        """
        Grid search lazily through search space in roder to find the params
        which result in the 'best' forecast depending on the given optimization
        parameters.

        Parameters
        ----------
        time_series : pd.Series
            The time series.
        optimization_type : str, optional
            How to search the space, only 'grid_search' is implemented. 
            TODO: add bayesian optimization.
            The default is 'grid_search'.
        optimization_strategy : str, optional
            The strategy emplyed when determing the 'best' params. The options
            are ['rolling', 'holdout'] where rolling uses a cross validation 
            strategy to 'roll' through the test set. Holdout simply hold out 
            the last 'lag' data points for testing.
            The default is 'rolling'.
        optimization_steps : int, optional
            When performing 'rolling' optimization_strategy the number of steps
            to test on. The default is 3.
        lag : int, optional
            How many data points to use as the test set. When using 'rolling',
            this parameter and optimization_steps determines the total number of
            testing points. Fore example, a lag of 2 with 3 steps means 3 * 2
            or 6 total points.  In step one we holdout the last 6 and test only 
            using the first 3 periods of the test set. In step two we include 
            the last step's test set in the train to test the final 3 periods.
            The default is 2.
        optimization_metric : str, optional
            The metric to judge the test forecast by. Options are :
            ['smape', 'mape', 'mse', 'mae'].
            The default is 'smape'.
        test_set : TYPE, optional
            DESCRIPTION. The default is 'all'.
        verbose : TYPE, optional
            DESCRIPTION. The default is 1.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        optimized_params : pd.DataFrame
            The predicted output dataframe of the optimal parameters.

        """
        time_series = pd.Series(time_series).copy(deep=True)
        self.time_series = time_series
        self.optimizer = Optimizer(self,
                                   time_series,
                                   optimization_type,
                                   optimization_strategy,
                                   optimization_steps,
                                   lag,
                                   optimization_metric,
                                   test_set,
                                   verbose,
                                   **kwargs)
        optimized = self.optimizer.optimize()
        self.optimized_params = self.optimizer.run_settings
        return optimized

    def autofit(self,
                 time_series,
                 seasonal_period=[0],
                 optimization_type='grid_search',
                 optimization_strategy='cv',
                 optimization_steps=3,
                 lag=2,
                 optimization_metric='mse',
                 test_set='all',
                 verbose=1,
                 fast=False):
        """
        Grid search lazily through predefined search space in order to find the 
        params which result in the 'best' forecast depending on the given 
        optimization parameters.

        Parameters
        ----------
        time_series : pd.Series
            The time series.
        optimization_type : str, optional
            How to search the space, only 'grid_search' is implemented. 
            TODO: add bayesian optimization.
            The default is 'grid_search'.
        optimization_strategy : str, optional
            The strategy emplyed when determing the 'best' params. The options
            are ['rolling', 'holdout'] where rolling uses a cross validation 
            strategy to 'roll' through the test set. Holdout simply hold out 
            the last 'lag' data points for testing.
            The default is 'rolling'.
        optimization_steps : int, optional
            When performing 'rolling' optimization_strategy the number of steps
            to test on. The default is 3.
        lag : int, optional
            How many data points to use as the test set. When using 'rolling',
            this parameter and optimization_steps determines the total number of
            testing points. Fore example, a lag of 2 with 3 steps means 3 * 2
            or 6 total points.  In step one we holdout the last 6 and test only 
            using the first 3 periods of the test set. In step two we include 
            the last step's test set in the train to test the final 3 periods.
            The default is 2.
        optimization_metric : str, optional
            The metric to judge the test forecast by. Options are :
            ['smape', 'mape', 'mse', 'mae'].
            The default is 'smape'.
        test_set : TYPE, optional
            DESCRIPTION. The default is 'all'.
        verbose : TYPE, optional
            DESCRIPTION. The default is 1.
        fast : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        optimized_params : pd.DataFrame
            The predicted output dataframe of the optimal parameters.

        """
        time_series = pd.Series(time_series).copy(deep=True)
        self.time_series = time_series
        if isinstance(seasonal_period, list):
            generator_seasonality = True
        else:
            generator_seasonality = False
        if generator_seasonality:
            max_seasonal_pulse = seasonal_period[0]
        else:
            max_seasonal_pulse = seasonal_period
        if not generator_seasonality:
            seasonal_period = [seasonal_period]
        if seasonal_period[0]:
                seasonal_period = [0, seasonal_period]

        _contains_zero = not (time_series > 0).all()
        if _contains_zero:
            additive = [True]
        else:
            additive = [False]

        if fast:
            param_dict = {'trend_estimator': ['linear',
                                              ['linear', 'ses'],
                                              'fast_arima',
                                              'fast_ets'],
                              'seasonal_estimator': ['fourier'],
                              'seasonal_period': seasonal_period,
                              'fit_type': ['global'],
                              'global_cost': ['mse'],
                              'additive': additive
                              }
        else:
            param_dict = {'trend_estimator': ['linear',
                                              ['linear', 'ses'],
                                              'ses',
                                               'arima',
                                              ThymeBoost.combine(['ses', 'des', 'damped_des'])],
                           'arima_order': ['auto'],
                          'seasonal_estimator': ['fourier'],
                          'seasonal_period': seasonal_period,
                          'fit_type': ['global'],
                          'global_cost': ['mse'],
                          'additive': additive
                          }
        if len(time_series) > 2.5 * max_seasonal_pulse and max_seasonal_pulse:
            seasonal_sample_weights = []
            weight = 1
            for i in range(len(time_series)):
                if (i) % max_seasonal_pulse == 0:
                    weight += 1
                seasonal_sample_weights.append(weight)
            param_dict['seasonality_weights'] = [None,
                                                 np.array(seasonal_sample_weights)]
        self.optimizer = Optimizer(self,
                                   time_series,
                                   optimization_type,
                                   optimization_strategy,
                                   optimization_steps,
                                   lag,
                                   optimization_metric,
                                   test_set,
                                   verbose,
                                   **param_dict)
        optimized = self.optimizer.optimize()
        self.optimized_params = self.optimizer.run_settings
        return optimized

    def update(self,
               output,
               new_input
               ):
        """
        Experimental feature to do online learning.
        TODO : More testing!

        Parameters
        ----------
        output : pd.DataFrame
            Output from fit method.
        new_input : TYPE
            The new data to update the model on. Use same input type as fitted.

        Returns
        -------
        full_output : pd.DataFrame
            The same output format that is returned from the fit method.

        """
        assert self.additive == True, 'Additive must be True for online learning'
        if not isinstance(new_input, pd.Series) or isinstance(new_input, np.ndarray):
            new_input = np.array(new_input).reshape(-1, )
        original_booster = self.booster_obj
        self.online_learning = True
        predictions = self.predict(output, len(new_input), uncertainty=False)
        update_series = output['y'] - output['yhat']
        update_series = update_series.append(new_input - predictions['predictions'])
        updated_output = self.fit(update_series, **self._params)
        column_names = ['yhat',
                        'trend',
                        'seasonality',
                        'exogenous',
                        'yhat_upper',
                        'yhat_lower'
                         ]
        predictions.columns = column_names
        predictions['y'] = new_input
        predictions = predictions[output.columns]
        full_output = output.append(predictions)
        if self.additive:
            full_output.iloc[:, 1:] += updated_output.iloc[:, 1:]
        else:
            full_output[column_names - ['y']] += updated_output[column_names - ['y']]
            full_output['seasonality'] = full_output['seasonality'] + \
                                        full_output['seasonality']
            raise
        self.booster_obj = original_booster + self.booster_obj
        self.builder.time_series = full_output['y']
        bounds = self.builder.get_fitted_intervals(full_output['y'],
                                                   full_output['yhat'])
        full_output['yhat_upper'], full_output['yhat_lower'] = bounds
        self.online_learning = None
        return full_output

    def ensemble(self,
                 time_series,
                 verbose=0,
                 **kwargs):
        """
        Perform ensembling aka a simple average of each combination of inputs.
        For example: passing trend_estimator=['mean', 'linear'] will fit using 
        BOTH mean and linear then average the results. We can use generator 
        features here as well such as: tren_estimator=['mean', ['linear', 'mean']].
        Notice that now we pass a list of lists.

        Parameters
        ----------
        time_series : pd.Series
            The time series.
        verbose : bool, optional
            Print statments. The default is 1.
        **kwargs : list
            list of features that are typically passed to fit that you want to
            ensemble.

        Returns
        -------
        output : : pd.DataFrame
            The predicted output dataframe from the ensembled params.

        """
        time_series = pd.Series(time_series).copy(deep=True)
        self.ensembler = Ensemble(model_object=self,
                                  y=time_series,
                                  verbose=verbose,
                                  **kwargs)
        output, ensemble_params = self.ensembler.ensemble_fit()
        self.ensemble_boosters = ensemble_params
        return output

    def detect_outliers(self,
                        time_series,
                        trend_estimator='linear',
                        fit_type='global',
                        **kwargs):
        """
        This is an off-the-cuff helper method for outlier detection. 
        Definitely do not use ETS, ARIMA, or Loess estimators.
        User beware!

        Parameters
        ----------
        time_series : pd.Series
            the time series.
        trend_estimator : str, optional
            'Approved' options are ['mean', 'median', 'linear', 'ransac']. 
            The default is 'ransac'.
        fit_type : str, optional
            Whether to use global or local fitting.
            Options are ['local', 'global']. The default is 'local'.
        seasonal_estimator : str, optional
            The method to approximate the seasonal component. 
            The default is 'fourier'.
        seasonal_period : int, optional
            The seasonal frequency. The default is None.

        Returns
        -------
        fitted_results : pd.DataFrame
            Output from booster with a new 'outliers' column added with 
            True/False denoting outlier classification.

        """
        fitted_results = self.fit(time_series=time_series,
                                  trend_estimator=trend_estimator,
                                  fit_type=fit_type,
                                  **kwargs)
        fitted_results['outliers'] = (fitted_results['y'].gt(fitted_results['yhat_upper'])) | \
                                     (fitted_results['y'].lt(fitted_results['yhat_lower']))
        return fitted_results

    @staticmethod
    def plot_results(fitted, predicted=None, figsize=None):
        """
        Plotter helper function to plot the results. 
        Plot adapts depending on the inputs.

        Parameters
        ----------
        fitted : pd.DataFrame
            Output df from either fit, optimize, or ensemble method.
        predicted : pd.DataFrame, optional
            Dataframe from predict method. The default is None.
        figsize : tuple, optional
            Matplotlib's figsize. The default is None.

        Returns
        -------
        None.

        """
        plotting.plot_results(fitted, predicted, figsize)

    @staticmethod
    def plot_components(fitted, predicted=None, figsize=None):
        """
        Plotter helper function to plot each component. 
        Plot adapts depending on the inputs.

        Parameters
        ----------
        fitted : pd.DataFrame
            Output df from either fit, optimize, or ensemble method.
        predicted : pd.DataFrame, optional
            Dataframe from predict method. The default is None.
        figsize : tuple, optional
            Matplotlib's figsize. The default is None.

        Returns
        -------
        None.

        """
        plotting.plot_components(fitted, predicted, figsize)

    def plot_optimization(self, fitted):
        opt_predictions = self.optimizer.cv_predictions
        opt_type = self.optimizer.optimization_strategy
        plotting.plot_optimization(fitted, opt_predictions, opt_type=opt_type, figsize=(12,8))
