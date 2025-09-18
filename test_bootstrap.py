import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, r_squared
from sklearn.linear_model import LinearRegression

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    # This test should initially fail
    """small_design_matrix = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    small_response_vector = np.array([1, 2])
    n_bootstrap = 1000
    def compute_beta(X, y):
        This function performs linear regression and returns beta
        reg = LinearRegression().fit(X, y)
        return reg.coef_
    
    bootstrap_stats = bootstrap_sample(small_design_matrix, small_response_vector, compute_beta, n_bootstrap)
    ci = bootstrap_ci(bootstrap_stats, alpha=0.05)"""

    pass

class TestBootstrapCI:
    def test_bootstrap_ci_length(self):
        """Test that bootstrap_ci returns a tuple of length 2"""
        bootstrap_stats = np.random.rand(1000)
        ci = bootstrap_ci(bootstrap_stats, alpha=0.05)
        assert isinstance(ci, tuple)
        assert len(ci) == 2

    def test_bootstrap_ci_order(self):
        """Test that the lower bound is less than the upper bound"""
        bootstrap_stats = np.random.rand(1000)
        ci = bootstrap_ci(bootstrap_stats, alpha=0.05)
        assert ci[0] < ci[1]
    
    def test_bootstrap_size(self):
        """Test raises a warning if the number of bootstrap samples is too small"""
        with pytest.warns(UserWarning, match="Number of bootstrap samples is small"):
            bootstrap_ci(np.random.rand(10), alpha=0.05)
    
    def test_bootstrap_ci_edge_cases(self):
        """Test edge cases for number of bootstrap statistics and alpha"""
        with pytest.raises(ValueError, match="bootstrap_stats must contain at least 2 values"):
            bootstrap_ci([1], alpha=0.05)
        
        with pytest.raises(ValueError, match="bootstrap_stats must contain at least 2 values"):
            bootstrap_ci([], alpha=0.05)
        
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            bootstrap_ci(np.random.rand(1000), alpha=-0.1)
    
    def test_bootstrap_type_errors(self):
        """Test that type errors are raised for invalid inputs"""
        with pytest.raises(TypeError, match="bootstrap_stats must be array-like"):
            bootstrap_ci("not an array", alpha=0.05)
        
        with pytest.raises(TypeError, match="alpha must be a float"):
            bootstrap_ci(np.random.rand(1000), alpha="not a float")

class TestRSquared:
    def test_r_squared_value(self):
        """Tests that the dimensions of the input are correct"""
        with pytest.raises(ValueError, match="X.shape[0] must equal len(y)"):
            r_squared(np.array([[1, 2], [3, 4]]), np.array([1]))

    def test_r_squared_bounds(self):
        """Tests that R-squared is between 0 and 1 for valid inputs"""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([1, 2, 3, 4])
        r2 = r_squared(X, y)
        assert 0 <= r2 <= 1
        
    def test_r_squared_type(self):
        """Tests that type errors are raised for invalid inputs"""
        with pytest.raises(TypeError, match="X must be array-like"):
            r_squared("not an array", np.array([1, 2]))
        
        with pytest.raises(TypeError, match="y must be array-like"):
            r_squared(np.array([[1, 2], [3, 4]]), "not an array")


# TODO: Add your unit tests here

