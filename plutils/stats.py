#!/usr/bin/env python3

from scipy.stats import rankdata, norm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def orders_of_magnitude(x):
    if len(x) < 2:
        return 0
    else:
        return np.log10(max(x)) - np.log10(min(x))


def pca(x, return_loadings=False):
    """
    Perform PCA. Centering is performed automatically. No scaling is performed.
    
    Input:
    mat: dataframe/array. rows = samples, columns = features

    Output:
    (dataframe of PC scores, np.array of fraction variance explained)
    """
    pca = PCA().fit(x)
    pcs = pd.DataFrame(pca.transform(x))
    pcs.columns = ['PC{}'.format(i) for i in range(1, len(pcs.columns)+1)]
    pcs.index = x.index
    explained_variance = pca.explained_variance_ratio_
    if not return_loadings:
        return (pcs, explained_variance)
    else:
        loadings = pd.DataFrame(pca.components_, columns=x.columns, index=[f'PC{x}' for x in range(1, len(pca.components_)+1)])
        return (pcs, explained_variance, loadings)


def inverse_normalize(x, c = 3/8):
    """
    Inverse normalize a numpy array/pandas series.

    x: array or series
    c: float (see equation 1 here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2921808/)
    """
    assert(isinstance(x, (np.ndarray, pd.Series)))
    ranks = rankdata(x)
    return norm.ppf( (ranks - c) / (len(x) - 2*c + 1) )


def expected_pvalues(n):
    """
    Produce a list of n expected p-values.

    n: int
    """
    if not n > 0:
        raise ValueError('n must be > 0')
    if not isinstance(n, int):
        raise TypeError('n must be an int')
    return [i/(n+1) for i in range(1, n+1)]


def median_from_dict(d):
    """
    Given a dictionary of value counts (d[value] = count), calculate the median value.

    d: dictionary, where values are non-negative integers.
    """
    if not isinstance(d, dict):
        raise TypeError('d must be a dict')
    if not len(d) > 0:
        raise ValueError('Cannot find median of an empty dict')
    for i in d.values():
        if not isinstance(i, (int, np.integer)):
            raise TypeError('All values in dict d must be integers')
        if i < 0:
            raise ValueError('All values in dict d must be >= 0')
    count = sum(d.values())
    middle = [int(count / 2), int(count / 2) + 1] if count % 2 == 0 else [int(count / 2) + 1]
    middle_values = []
    index = 0
    for key, val in sorted(d.items()):
        for i in middle:
            if i > index and i <= (index + val):
                middle_values.append(key)
        index += val
        if len(middle_values) == len(middle):
            break
    return np.mean(middle_values)


def empirical_p (actual, permuted, lower_tail = True, pseudocount = 0):
    """
    actual should be an actual value
    permutated should be a list-like of permuted values
    """
    permuted_s = pd.Series(permuted) if not isinstance(permuted, pd.Series) else permuted.copy()
    if lower_tail:
        number_less_than_or_equal_to = (permuted_s<=actual).sum()
        return (number_less_than_or_equal_to + pseudocount) / (len(permuted_s)+pseudocount)
    else:
        number_greater_than_or_equal_to = (permuted_s>=actual).sum()
        return (number_greater_than_or_equal_to + pseudocount) / (len(permuted_s)+pseudocount)
