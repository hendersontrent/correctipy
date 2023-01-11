# correctipy

Corrected test statistics for comparing machine learning models on
correlated samples

## Installation

You can install the stable version of `correctipy` from PyPI using:

``` python
pip install correctipy
```

## General purpose

Often in machine learning, we want to compare the performance of
different models to determine if one statistically outperforms another.
However, the methods used (e.g., data resampling, $k$-fold
cross-validation) to obtain these performance metrics (e.g.,
classification accuracy) violate the assumptions of traditional
statistical tests such as a $t$-test. The purpose of these methods is to
either aid generalisability of findings (i.e., through quantification of
error as they produce multiple values for each model instead of just
one) or to optimise model hyperparameters. This makes them invaluable,
but unusable with traditional tests, as [Dietterich
(1998)](https://pubmed.ncbi.nlm.nih.gov/9744903/) found that the
standard $t$-test underestimates the variance, therefore driving a high
Type I error. `correctipy` is a lightweight package that implements a
small number of corrected test statistics for cases when samples are not
independent (and therefore are correlated), such as in the case of
resampling, $k$-fold cross-validation, and repeated $k$-fold
cross-validation. These corrections were all originally proposed by
[Nadeau and Bengio
(2003)](https://link.springer.com/article/10.1023/A:1024068626366).
Currently, only cases where two models are to be compared are supported.

If you are interested in the version for R, please see [`correctR`](https://github.com/hendersontrent/correctR).

## Basic usage

`correctipy` is a lightweight package that implements a small number of corrected test statistics for cases when samples of two machine learning model metrics (e.g., classification accuracy) are not independent (and therefore are correlated), such as in the case of resampling and $k$-fold cross-validation. We demonstrate the basic functionality here using some trivial examples for the following corrected tests that are currently implemented in `correctipy`:

* Random subsampling
* $k$-fold cross-validation
* Repeated $k$-fold cross-validation

These corrections were all originally proposed by [Nadeau and Bengio (2003)](https://link.springer.com/article/10.1023/A:1024068626366) with additional representations in [Bouckaert and Frank (2004)](https://link.springer.com/chapter/10.1007/978-3-540-24775-3_3).

### Random subsampling correction

In random subsampling, the standard $t$-test inflates Type I error when used in conjunction with random subsampling due to an underestimation of the variance, as found by [Dietterich (1998)](https://pubmed.ncbi.nlm.nih.gov/9744903/). Nadeau and Bengio (2003) proposed a solution (which we implement as `resampled_ttest` in `correctipy`) in the form of:

$$
t = \frac{\frac{1}{n} \sum_{j=1}^{n}x_{j}}{\sqrt{(\frac{1}{n} + \frac{n_{2}}{n_{1}})\sigma^{2}}}
$$

where $n$ is the number of resamples (NOTE: $n$ is *not* sample size), $n_{1}$ is the number of samples in the training data, and $n_{2}$ is the number of samples in the test data. $\sigma^{2}$ is the variance estimate used in the standard paired $t$-test (which simply has $\frac{\sigma}{\sqrt{n}}$ in the denominator where $n$ is the sample size in this case).

### k-fold cross-validation correction

There is an alternate formulation of the random subsampling correction, devised in terms of the unbiased estimator $\rho$, discussed in [Corani et al. (2016)](https://link.springer.com/article/10.1007/s10994-017-5641-9) which we implement as `kfold_tttest` in `correctipy`:

$$
t = \frac{\frac{1}{n} \sum_{j=1}^{n}x_{j}}{\sqrt{(\frac{1}{n} + \frac{\rho}{1-\rho})\sigma^{2}}}
$$

where $n$ is the number of resamples and $\rho = \frac{1}{k}$ where $k$ is the number of folds in the $k$-fold cross-validation procedure. This formulation stems from the fact that Nadeau and Bengio (2003) proved there is no unbiased estimator, but it can be approximated with $\rho = \frac{1}{k}$.

### Repeated k-fold cross-validation correction

Repeated $k$-fold cross-validation is more complex than the previous case(s) as we now have $r$ repeats for every fold $k$. Bouckaert and Frank (2004) present a nice representation of the corrected test for this case which we implement as `repkfold_ttest` in `correctipy`:

$$
t = \frac{\frac{1}{k \cdot r} \sum_{i=1}^{k} \sum_{j=1}^{r} x_{ij}}{\sqrt{(\frac{1}{k \cdot r} + \frac{n_{2}}{n_{1}})\sigma^{2}}}
$$

## Setup

In the real world, we would have proper results obtained through fitting two models according to one or more of the procedures outlined above. For simplicity here, we are just going to simulate three datasets so we can get to the package functionality cleaner and easier. We are going to assume we are in a classification context and generate classification accuracy values. These values are purposefully egregious---we are going to (in the case of the random subsampling) just fix the train set sample size (`n1`) to 80 and the test set sample size (`n2`) to 20, and assume (using the same data) for the $k$-fold cross-validation correction that the same numbers were obtained on such a method. Again, the values are not important here, it is the corrections we are going to apply next that are crucial.

In the case of repeated $k$-fold cross-validation, take note of the column names. While your `data.frame` you pass in to `repkfold_ttest` can have more than the four columns specified here, it **must** contain at least these four with the exact corresponding names. The function explicitly searches for them. They are:

1. `"model"` --- contains a label for each of the two models to compare
2. `"values"` --- the numerical values of the performance metric (i.e., classification accuracy)
3. `"k"` --- which fold the values correspond to
4. `"r"` --- which repeat of the fold the values correspond to

```python
import numpy as np
import pandas as pd

x = np.random.normal(0.6, 0.1, 30)
y = np.random.normal(0.4, 0.1, 30)

tmp = pd.DataFrame({'model':np.repeat([1, 2], 60), 
                   'values':np.concatenate((np.random.normal(0.6, 0.1, 60), np.random.normal(0.4, 0.1, 60))),
                   'k':np.repeat([1, 1, 2, 2], 15),
                   'r':np.repeat(np.array([1, 2]), 30)
                  })

```

## Package functionality

We can fit all the corrections in one-line functions:

```python
import correctipy 

rss = resampled_ttest(x, y, 30, 80, 20) # Random subsampling
kcv = kfold_ttest(x, y, 100, 30) # k-fold cross-validation
rkcv = repkfold_ttest(tmp, 80, 20, 2, 2) # Repeated k-fold cross-validation
```

All the functions return a Pandas dataframe with two named columns: `"statistic"` (the $t$-statistic) and `"p_value"` (the associated $p$-value), meaning they can be easily integrated into complex machine pipelines.