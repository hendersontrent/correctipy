import numpy as np
import pandas as pd
import scipy.stats as stats

def resampled_ttest(x, y, n=None, n1=None, n2=None):

    """
    Compute correlated t-statistic and p-value for resampled data.
    Args:
        x (array): vector of values for model A
        y (array): vector of values for model B
        n (array): number of repeat samples
        n1 (array): train set size
        n2 (array): test set size

    Returns:
        dataframe: Pandas dataframe containing the test statistic and the p-value.
    """

    # Arg checks
    if len(x) != len(y):
        raise ValueError("x and y are not the same length.")

    if not all(np.isreal(x)) or not all(np.isreal(y)):
        raise ValueError("x and y should be numeric vectors of the same length.")

    if n is None:
        n = len(x)
        print("n argument missing. Using length(x) as default.")

    if not all(np.isscalar(i) for i in (n, n1, n2)):
        raise ValueError("n, n1, and n2 should all be scalars.")

    # Calculations
    d = x - y  # Calculate differences
    statistic = np.mean(d) / (np.std(d, ddof=1) * (1/n + n2/n1))  # Calculate t-statistic

    if statistic < 0:
        p_value = stats.t.cdf(statistic, n-1) # p-value for left tail
    else:
        p_value = stats.t.sf(statistic, n-1) # p-value for right tail

    stat_df = pd.DataFrame({"statistic": [statistic], "p_value": [p_value]})
    return stat_df


def kfold_ttest(x, y, n, k):

    """
    Compute correlated t-statistic and p-value for resampled data.
    Args:
        x (array): vector of values for model A
        y (array): vector of values for model B
        n (array): total sample size
        k (array): train set size

    Returns:
        dataframe: Pandas dataframe containing the test statistic and the p-value.
    """
    
    # Arg checks
    if len(x) != len(y):
        raise ValueError("x and y are not the same length.")

    if not all(np.isreal(x)) or not all(np.isreal(y)):
        raise ValueError("x and y should be numeric vectors of the same length.")
    
    if not (np.isscalar(n) and np.isscalar(k) and np.isreal(n) and np.isreal(k)):
        raise ValueError("n and k should be integer scalars.")

    # Calculations
    d = x - y  # Calculate differences
    statistic = np.mean(d) / (np.std(d, ddof=1) * ((1/n + (1/k)) / (1 - 1/k)))  # Calculate t-statistic

    if statistic < 0:
        p_value = stats.t.cdf(statistic, n-1) # p-value for left tail
    else:
        p_value = stats.t.sf(statistic, n-1) # p-value for right tail

    stat_df = pd.DataFrame({"statistic": [statistic], "p_value": [p_value]})
    return stat_df


def repkfold_ttest(data, n1, n2, k, r):

    """
    Compute correlated t-statistic and p-value for repeated k-fold cross-validated results.
    Args:
        data (dataframe): dataframe of values for model A and model B over repeated k-fold cross-validation
        n1 (array): train set size
        n2 (array): test set size
        k (array): number of folds used in k-fold
        r (array): number of repeats per fold

    Returns:
        dataframe: Pandas dataframe containing the test statistic and the p-value.
    """
    
    # Arg checks
    if 'model' not in data.columns:
        raise ValueError("data should contain at least four columns called 'model', 'values', 'k', and 'r'.")
    if 'values' not in data.columns:
        raise ValueError("data should contain at least four columns called 'model', 'values', 'k', and 'r'.")
    if 'k' not in data.columns:
        raise ValueError("data should contain at least four columns called 'model', 'values', 'k', and 'r'.")
    if 'r' not in data.columns:
        raise ValueError("data should contain at least four columns called 'model', 'values', 'k', and 'r'.")

    if not (np.issubdtype(data['values'].dtype, np.number) and np.issubdtype(data['k'].dtype, np.number) and np.issubdtype(data['r'].dtype, np.number)):
        raise ValueError("data should be a data.frame with only numerical values in columns 'values', 'k', and 'r'.")

    if not (np.isscalar(n1) and np.isscalar(n2) and np.isscalar(k) and np.isscalar(r) and np.isreal(n1) and np.isreal(n2) and np.isreal(k) and np.isreal(r)):
        raise ValueError("n1, n2, k, and r should all be integer scalars.")

    if len(data['model'].unique()) != 2:
        raise ValueError("Column 'model' in data should only have two unique labels (one for each model to compare).")

    d = []

    for i in range(1, k+1):
        for j in range(1, r+1):
            x = data[(data['k']==i) & (data['r']==j)]
            model_values = x.groupby(by='model').agg({'values': 'mean'})
            d.append(model_values.values[0,0] - model_values.values[1,0])

    statistic = np.mean(d) / (np.sqrt(np.sqrt(np.var(d, ddof=1, keepdims=True)) * ((1/(k * r)) + (n2/n1)))) # Calculate t-statistic

    if statistic < 0:
        p_value = stats.t.cdf(statistic, (k * r) - 1) # p-value for left tail
    else:
        p_value = stats.t.sf(statistic, (k * r) - 1) # p-value for right tail

    stat_df = pd.DataFrame({"statistic": [statistic], "p_value": [p_value]})
    return stat_df
