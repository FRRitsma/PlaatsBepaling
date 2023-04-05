# %%
from scipy import optimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from functools import cache
import numpy as np

""" Module with a function for hyperparameter optimization in an efficient 
    way using Golden Section optimization and caching results
"""


def _float_to_int(f: callable):
    """
    Wrapper that ensures that converts float values generated during optimization
    to valid integers for hyperparameter settings. (i.e. int and larger than zero)

    Args:
        f (callable): _description_
    """

    def inner(args):
        if type(args) is list:
            args = [max(int(i), 1) for i in args]
        else:
            args = max(int(args), 1)
        return f(args)

    return inner


def fit_model(x: np.ndarray, y: np.ndarray, start_value: int) -> GradientBoostingRegressor:
    """ Performs both model fitting and hyperparameter optimization.

    Args:
        x (np.ndarray): Model input values
        y (np.ndarray): Model output values
        start_value (int): Starting value of hyperparameter optimization

    Returns:
        GradientBoostingRegressor: Model fitted to data and hyperparameter optimized
    """


    # Enforce valid input:
    assert (
        start_value > 0
    ), "'start value' denotes number of estimators in gradient boosting regressor and must be positive"

    assert (
        type(start_value) is int
    ), "'start value' denotes number of estimators in gradient boosting regressor and must be of type 'int'"


    # Data split:
    perm = np.random.rand(len(y))
    val_frac = 0.2
    mask = perm < np.quantile(perm, val_frac)
    y_val = y[mask]
    x_val = x[mask, :]
    mask = np.logical_not(mask)
    y_train = y[mask]
    x_train = x[mask, :]

    # Function scope hardcode:
    METRIC = mean_squared_error

    # These functions are declared in function scope because their cache must be clear at start of run.
    @_float_to_int
    @cache
    def inner_train_model(n_estimators: int):
        gbr = GradientBoostingRegressor(n_estimators=n_estimators)
        gbr.fit(x_train, y_train)
        return gbr

    @cache
    def score_model(n_estimators: int):
        gbr = inner_train_model(n_estimators)
        return METRIC(y_val, gbr.predict(x_val))

    # Perform optimization:
    minimum = optimize.golden(
        score_model, brack=(start_value, start_value + 10), full_output=True
    )

    return inner_train_model(minimum[0])
