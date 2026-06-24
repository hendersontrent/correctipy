import numpy as np
import pandas as pd
from scipy import stats


def _check_tailed(tailed):
    """Validate the ``tailed`` argument shared by all corrections."""
    if tailed not in ("two", "one"):
        raise ValueError("tailed should be one of 'two' or 'one'.")


def _p_value(statistic, df, tailed):
    """Compute the p-value for a t-statistic to match correctR.

    Two-tailed: 2 * pt(-|t|, df). One-tailed: 1 - pt(t, df).
    """
    if tailed == "two":
        return 2 * stats.t.cdf(-abs(statistic), df)
    return stats.t.sf(statistic, df)  # 1 - pt(statistic, df)


def resampled_ttest(x, y, n=None, n1=None, n2=None, tailed="two", greater=None):

    """
    Compute correlated t-statistic and p-value for resampled data.
    Args:
        x (array): vector of values for model A
        y (array): vector of values for model B
        n (int): number of repeat samples. Defaults to len(x)
        n1 (int): train set size
        n2 (int): test set size
        tailed (str): whether to perform a 'two'-tailed or 'one'-tailed test. Defaults to 'two'
        greater (str): which of 'x' or 'y' is greater for the one-tailed test if tailed='one'. Defaults to None

    Returns:
        dataframe: Pandas dataframe containing the test statistic and the p-value.
    """

    # Arg checks
    _check_tailed(tailed)
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y are not the same length.")

    if not all(np.isreal(x)) or not all(np.isreal(y)):
        raise ValueError("x and y should be numeric vectors of the same length.")

    if n is None:
        n = len(x)
        print("n argument missing. Using length(x) as default.")

    if not all(np.isscalar(i) for i in (n, n1, n2)):
        raise ValueError("n, n1, and n2 should all be integer scalars.")

    if tailed == "one" and greater not in ("x", "y"):
        raise ValueError("If tailed = 'one', greater must be either 'x' or 'y'.")

    # Calculations
    if tailed == "two" or greater == "x":
        d = x - y  # Calculate differences
    else:
        d = y - x

    if np.sum(d) == 0:
        return pd.DataFrame({"statistic": [0], "p_value": [1]})

    statistic = np.mean(d) / np.sqrt(np.var(d, ddof=1) * (1/n + n2/n1))  # Calculate t-statistic
    p_value = _p_value(statistic, n - 1, tailed)

    stat_df = pd.DataFrame({"statistic": [statistic], "p_value": [p_value]})
    return stat_df


def kfold_ttest(x, y, n, k, tailed="two", greater=None):

    """
    Compute correlated t-statistic and p-value for k-fold cross-validated results.
    Args:
        x (array): vector of values for model A
        y (array): vector of values for model B
        n (int): number of resamples
        k (int): number of folds used in k-fold
        tailed (str): whether to perform a 'two'-tailed or 'one'-tailed test. Defaults to 'two'
        greater (str): which of 'x' or 'y' is greater for the one-tailed test if tailed='one'. Defaults to None

    Returns:
        dataframe: Pandas dataframe containing the test statistic and the p-value.
    """

    # Arg checks
    _check_tailed(tailed)
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y are not the same length.")

    if not all(np.isreal(x)) or not all(np.isreal(y)):
        raise ValueError("x and y should be numeric vectors of the same length.")

    if not (np.isscalar(n) and np.isscalar(k) and np.isreal(n) and np.isreal(k)):
        raise ValueError("n and k should be integer scalars.")

    if tailed == "one" and greater not in ("x", "y"):
        raise ValueError("If tailed = 'one', greater must be either 'x' or 'y'.")

    # Calculations
    if tailed == "two" or greater == "x":
        d = x - y  # Calculate differences
    else:
        d = y - x

    if np.sum(d) == 0:
        return pd.DataFrame({"statistic": [0], "p_value": [1]})

    statistic = np.mean(d) / np.sqrt(np.var(d, ddof=1) * ((1/n + (1/k)) / (1 - 1/k)))  # Calculate t-statistic
    p_value = _p_value(statistic, n - 1, tailed)

    stat_df = pd.DataFrame({"statistic": [statistic], "p_value": [p_value]})
    return stat_df


def repkfold_ttest(data, n1, n2, k, r, tailed="two", greater=None):

    """
    Compute correlated t-statistic and p-value for repeated k-fold cross-validated results.
    Args:
        data (dataframe): dataframe of values for model A and model B over repeated k-fold cross-validation
        n1 (int): train set size
        n2 (int): test set size
        k (int): number of folds used in k-fold
        r (int): number of repeats per fold
        tailed (str): whether to perform a 'two'-tailed or 'one'-tailed test. Defaults to 'two'
        greater: which label in the 'model' column is greater for the one-tailed test if tailed='one'. Defaults to None

    Returns:
        dataframe: Pandas dataframe containing the test statistic and the p-value.
    """

    # Arg checks
    _check_tailed(tailed)

    for col in ("model", "values", "k", "r"):
        if col not in data.columns:
            raise ValueError("data should contain at least four columns called 'model', 'values', 'k', and 'r'.")

    if not (np.issubdtype(data['values'].dtype, np.number) and np.issubdtype(data['k'].dtype, np.number) and np.issubdtype(data['r'].dtype, np.number)):
        raise ValueError("data should be a data.frame with only numerical values in columns 'values', 'k', and 'r'.")

    if not (np.isscalar(n1) and np.isscalar(n2) and np.isscalar(k) and np.isscalar(r) and np.isreal(n1) and np.isreal(n2) and np.isreal(k) and np.isreal(r)):
        raise ValueError("n1, n2, k, and r should all be integer scalars.")

    if len(data['model'].unique()) != 2:
        raise ValueError("Column 'model' in data should only have two unique labels (one for each model to compare).")

    if tailed == "one" and greater not in data['model'].unique():
        raise ValueError("greater must correspond to a value in the 'model' column of data.")

    # Calculations
    d = []

    for i in range(1, k + 1):
        for j in range(1, r + 1):
            cell = data[(data['k'] == i) & (data['r'] == j)]
            models = cell['model'].unique()  # Order of appearance, matching R's unique()

            if tailed == "two":
                first, second = models[0], models[1]
            else:
                first = greater
                second = models[models != greater][0]

            a = cell[cell['model'] == first]['values'].to_numpy()
            b = cell[cell['model'] == second]['values'].to_numpy()
            d.extend((a - b).tolist())

    d = np.asarray(d)

    if np.sum(d) == 0:
        return pd.DataFrame({"statistic": [0], "p_value": [1]})

    statistic = np.mean(d) / np.sqrt(np.var(d, ddof=1) * ((1/(k * r)) + (n2/n1)))  # Calculate t-statistic
    p_value = _p_value(statistic, (k * r) - 1, tailed)

    stat_df = pd.DataFrame({"statistic": [statistic], "p_value": [p_value]})
    return stat_df
