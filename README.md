# ThymeBoost
ThymeBoost combines time series decomposition with gradient boosting to provide a flexible mix-and-match time series framework for spicy forecasting. At the most granular level are the trend/level (going forward this is just referred to as 'trend') models, seasonal models, and edogenous models. These are used to approximate the respective components at each 'boosting round' and sequential rounds are fit on residuals in usual boosting fashion.

Basic flow of the algorithm:



![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lp_flow.PNG?raw=true "Output 1")


Quick Start.
```
pip install ThymeBoost
```



# Some basic examples: 
