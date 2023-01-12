Quickstart
==========

Installation
------------

You can install correctipy using pip

.. code::
   
     pip install git+https://github.com/hendersontrent/correctipy

Usage
-----

In the real world, we would have proper results obtained through fitting two models according to one or more of the procedures contained in correctipy (random subsampling, k-fold cross validation, repeated k-fold cross-validation). For simplicity here, we are just going to simulate three datasets so we can get to the package functionality cleaner and easier. We are going to assume we are in a classification context and generate classification accuracy values. These values are purposefully egregious---we are going to (in the case of the random subsampling) just fix the train set sample size (``n1``) to 80 and the test set sample size (``n2``) to 20, and assume (using the same data) for the $k$-fold cross-validation correction that the same numbers were obtained on such a method. Again, the values are not important here, it is the corrections we are going to apply next that are crucial.

In the case of repeated $k$-fold cross-validation, take note of the column names. While your dataframe you pass in to ``repkfold_ttest`` can have more than the four columns specified here, it must contain at least these four with the exact corresponding names. The function explicitly searches for them. They are:

* ``"model"`` --- contains a label for each of the two models to compare
* ``"values"`` --- the numerical values of the performance metric (i.e., classification accuracy)
* ``"k"`` --- which fold the values correspond to
* ``"r"`` --- which repeat of the fold the values correspond to

Here is the simulated data:

.. code::
   
   >>> import numpy as np
   >>> import pandas as pd
   >>> x = np.random.normal(0.6, 0.1, 30)
   >>> y = np.random.normal(0.4, 0.1, 30)

   >>> tmp = pd.DataFrame({'model':np.repeat([1, 2], 60), 
   >>>                    'values':np.concatenate((np.random.normal(0.6, 0.1, 60),
   >>>                    np.random.normal(0.4, 0.1, 60))),
   >>>                    'k':[1, 1, 2, 2]*30,
   >>>                    'r':[1, 2]*60
   >>>                   })

We can fit all the corrections in one-line functions:

.. code::
   
   >>> from correctipy import resampled_ttest
   >>> from correctipy import kfold_ttest
   >>> from correctipy import repkfold_ttest

   >>> rss = resampled_ttest(x, y, 30, 80, 20) # Random subsampling
   >>> kcv = kfold_ttest(x, y, 100, 30) # k-fold cross-validation
   >>> kcv = kfold_ttest(x, y, 100, 30) # k-fold cross-validation
   >>> rkcv = repkfold_ttest(tmp, 80, 20, 2, 2) # Repeated k-fold cross-validation

All the functions return a Pandas dataframe with two named columns: ``"statistic"`` (the t-statistic) and ``"p_value"`` (the associated p-value), meaning they can be easily integrated into complex machine pipelines. Here is an example for the ``resampled_ttest`` case:

.. code::

   >>> print(rss)

          statistic       p_value
     0    6.09829  6.083703e-07
