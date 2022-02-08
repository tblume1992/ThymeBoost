# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from itertools import cycle

def plot_components(fitted_df, predicted_df, figsize):
    """Simple plot of components for convenience"""
    if predicted_df is not None:
        rename_dict = {'predictions': 'yhat',
                       'predicted_trend': 'trend',
                       'predicted_seasonality': 'seasonality',
                       'predicted_upper': 'yhat_upper',
                       'predicted_lower': 'yhat_lower',
                       'predicted_exogenous': 'exogenous'}
        predicted_df = predicted_df.rename(rename_dict,
                                           axis=1)
        component_df = fitted_df.append(predicted_df)
    else:
        component_df = fitted_df
    if 'exogenous' in fitted_df.columns:
        fig, ax = plt.subplots(4, figsize=figsize)
        ax[-2].plot(component_df['exogenous'], color='orange')
        ax[-2].set_title('Exogenous')
        ax[-2].xaxis.set_visible(False)
    else:
        fig, ax = plt.subplots(3, figsize=figsize)
    ax[0].plot(component_df['trend'], color='orange')
    ax[0].set_title('Trend')
    ax[0].xaxis.set_visible(False)
    ax[1].plot(component_df['seasonality'], color='orange')
    ax[1].set_title('Seasonality')
    ax[1].xaxis.set_visible(False)
    ax[-1].plot(component_df['y'], color='black')
    ax[-1].plot(component_df['yhat'], color='orange')
    ax[-1].plot(component_df['yhat_upper'],
                linestyle='dashed',
                alpha=.5,
                color='orange')
    ax[-1].plot(component_df['yhat_lower'],
                linestyle='dashed',
                alpha=.5,
                color='orange')
    ax[-1].set_title('Fitted')
    plt.tight_layout()
    plt.show()


def plot_results(fitted_df, predicted_df, figsize):
    """Simple plot of results for convenience"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fitted_df['y'], color='black')
    ax.plot(fitted_df['yhat'], color='orange')
    ax.plot(fitted_df['yhat_upper'],
            linestyle='dashed',
            alpha=.5,
            color='orange')
    ax.plot(fitted_df['yhat_lower'],
            linestyle='dashed',
            alpha=.5,
            color='orange')
    if predicted_df is not None:
        ax.plot(fitted_df['yhat'].tail(1).append(predicted_df['predictions']),
                color='red',
                linestyle='dashed')
        ax.fill_between(x=fitted_df['yhat_lower'].tail(1).append(predicted_df['predicted_lower']).index,
                        y1=fitted_df['yhat_lower'].tail(1).append(predicted_df['predicted_lower']).values,
                        y2=fitted_df['yhat_upper'].tail(1).append(predicted_df['predicted_upper']).values,
                        alpha=.5,
                        color='orange')
    ax.set_title('ThymeBoost Results')
    if 'outliers' in fitted_df.columns:
        outlier_df = fitted_df[fitted_df['outliers'] == True]
        ax.scatter(outlier_df.index, outlier_df['y'], marker='x', color='red')
    plt.show()

def plot_optimization(fitted_df, opt_predictions, opt_type, figsize):
    """Simple plot of optimization results for convenience"""
    step_colors = ['tab:blue',
                   'tab:orange',
                   'tab:green',
                   'tab:red',
                   'tab:purple',
                   'tab:cyan']
    step_colors = cycle(step_colors)
    min_opt_idx = opt_predictions[-1].index[0]
    fitted_y = fitted_df['yhat'].loc[:min_opt_idx]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fitted_df['y'], color='black')
    ax.plot(fitted_y, color='orange')
    for idx, i in enumerate(opt_predictions):
        section_color = next(step_colors)
        ax.plot(i, color=section_color, linestyle='dashed')
        if opt_type in ['holdout', 'cv']:
            ax.axvspan(i.index[0], i.index[-1], alpha=0.2, color=section_color, linestyle='dashed')
    ax.set_title('ThymeBoost Optimization')
    plt.show()

# def plot_rounds(self, figsize = (6,4)):
#     if self.exogenous is not None:
#         fig, ax = plt.subplots(3, figsize = figsize)
#         for iteration in range(len(self.fitted_exogenous)-1):
#             ax[2].plot(
#                     np.sum(self.fitted_exogenous[:iteration], axis=0),
#                     label=iteration
#                     )
#         ax[2].set_title('Exogenous')
#     else:
#         fig, ax = plt.subplots(2, figsize = figsize)
#     for iteration in range(len(self.trends)-1):
#         ax[0].plot(np.sum(self.trends[:iteration], axis=0), label=iteration)
#     ax[0].set_title('Trends')
#     for iteration in range(len(self.seasonalities)-1):
#         ax[1].plot(np.sum(self.seasonalities[:iteration], axis=0), label=iteration)
#     ax[1].set_title('Seasonalities')
#     plt.legend()
#     plt.show()
