import pytest
import numpy as np
import statsmodels.api as sm
from bootstrap import bootstrap_sample, bootstrap_ci, R_squared

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    # This test should initially fail
    small_design_matrix = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    small_response_vector = np.array([1, 2])
    n_bootstrap = 1000
    def mean_response(X, y):
        return np.mean(y)
    bootstrap_stats = bootstrap_sample(small_design_matrix, small_response_vector, mean_response, n_bootstrap)
    ci = bootstrap_ci(bootstrap_stats, alpha=0.05)
    
    """Test that the bootstrap confidence interval is a tuple of length 2"""
    assert isinstance(ci, tuple)
    assert len(ci) == 2
    assert ci[0] < ci[1]
    

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
            R_squared(np.array([[1, 2], [3, 4]]), np.array([1]))

    def test_r_squared_bounds(self):
        """Tests that R-squared is between 0 and 1 for valid inputs"""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([1, 2, 3, 4])
        r2 = R_squared(X, y)
        assert 0 <= r2 <= 1
        
    def test_r_squared_type(self):
        """Tests that type errors are raised for invalid inputs"""
        with pytest.raises(TypeError, match="X must be array-like"):
            R_squared("not an array", np.array([1, 2]))
        
        with pytest.raises(TypeError, match="y must be array-like"):
            R_squared(np.array([[1, 2], [3, 4]]), "not an array")



class TestBootstrapSample:
    def test_input_type_errors(self):
        """Test that output length matches n_bootstrap"""
        """Test that TypeErrors are raised for invalid input types"""
        with pytest.raises(TypeError, match="X must be a 1D or 2D array-like"):
            X = "invalid"
            y = np.random.normal(size=(10,))
            bootstrap_sample(X=X, y=y, compute_stat=np.corrcoef, n_bootstrap=10)
        
        with pytest.raises(TypeError, match="y must be a 1D array-like"):
            X = np.random.normal(size=(10,))
            y = "invalid"
            bootstrap_sample(X=X, y=y, compute_stat=np.corrcoef, n_bootstrap=10)
        
        with pytest.raises(TypeError, match="compute_stat must be callable"):
            X = np.random.normal(size=(10,))
            y = np.random.normal(size=(10,))
            compute_stat = "not_callable"
            bootstrap_sample(X=X, y=y, compute_stat=compute_stat, n_bootstrap=10)
    
        with pytest.raises(ValueError, match="n_bootstrap must be a positive integer"):
            X = np.random.normal(size=(10,))
            y = np.random.normal(size=(10,))
            n_bootstrap = -5
            bootstrap_sample(X=X, y=y, compute_stat=np.corrcoef, n_bootstrap=n_bootstrap)
        
        with pytest.raises(ValueError, match="X and y must have the same length"):
            X = [1,2,4]
            y = [3,4]
            bootstrap_sample(X=X, y=y, compute_stat=np.corrcoef, n_bootstrap=10)
            
        with pytest.raises(ValueError, match="output of compute_stat must be a scalar"):
            X = np.random.normal(size=(10,))
            y = np.random.normal(size=(10,))
            def invalid_stat(X, y):
                return [1, 2]  # Not a scalar
            bootstrap_sample(X=X, y=y, compute_stat=invalid_stat, n_bootstrap=10)

    def test_bootstrap_sample_output_type_errors(self):
        with pytest.raises(ValueError, match="output must be a 1D array-like"):
            X = np.random.normal(size=(10,))
            y = np.random.normal(size=(10,))
            compute_stat = np.corrcoef  # Returns a 2D array
            results = bootstrap_sample(X=X, y=y, compute_stat=compute_stat, n_bootstrap=10)
            assert type(results) == np.ndarray
            
        with pytest.raises(ValueError, match="output must have the length n_bootstrap"):
            X = np.random.normal(size=(10,))
            y = np.random.normal(size=(10,))
            compute_stat = np.corrcoef  # Returns a 2D array
            n_bootstrap = 10
            results = bootstrap_sample(X=X, y=y, compute_stat=compute_stat, n_bootstrap=n_bootstrap)
            assert len(results) == n_bootstrap
            
        

