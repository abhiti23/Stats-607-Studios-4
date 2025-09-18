
"""
Strong linear model in regression
    Y = X beta + eps, where eps~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution
    (if you have an intercept beta_0), 
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""
import numpy as np

def bootstrap_sample(X, y, compute_stat, n_bootstrap=1000):
    """
    Generate bootstrap distribution of a statistic

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y)
    n_bootstrap : int, default 1000
        Number of bootstrap samples to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    Raises
    ------
    ValueError
        If X.shape[0] != len(y)
    TypeError
        If compute_stat is not callable
    TypeError
        If n_bootstrap is not an integer
    TypeError
        If X or y are not array-like
    
    """
    """Check that the inputs are the appropriate types"""
    if not callable(compute_stat):
        raise TypeError("compute_stat must be callable")
    if not isinstance(n_bootstrap, int):
        raise TypeError("n_bootstrap must be a positive integer")
    if not isinstance(X, (list, np.ndarray)):
        raise TypeError("X must be a 1D or 2D array-like")
    if not isinstance(y, (list, np.ndarray)):
        raise TypeError("y must be a 1D array-like")
    
    """Check that X and y have compatible shapes"""
    if X.shape[0] != len(y):
        raise ValueError("X and y must have the same length")
    
    """Test that compute_stat returns a scalar"""
    test_stat = compute_stat(X, y)
    if not np.isscalar(test_stat):
        raise ValueError("output of compute_stat must be a scalar")
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        stat = compute_stat(X_boot, y_boot)
        bootstrap_stats.append(stat)
    return np.array(bootstrap_stats)
    

def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate confidence interval from the bootstrap samples

    Parameters
    ----------
    bootstrap_stats : array-like
        Array of bootstrap statistics
    alpha : float, default 0.05
        Significance level (e.g. 0.05 gives 95% CI)

    Returns
    -------
    tuple 
        (lower_bound, upper_bound) of the CI
    
    ....
    """
    pass

def R_squared(X, y):
    """
    Calculate R-squared from multiple linear regression.

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)

    Returns
    -------
    float
        R-squared value (between 0 and 1) from OLS
    
    Raises
    ------
    ValueError
        If X.shape[0] != len(y)
    """
    pass