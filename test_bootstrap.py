import pytest
import numpy as np
import statsmodels.api as sm
from bootstrap import bootstrap_sample, bootstrap_ci, r_squared

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    # This test should initially fail
    pass


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
            
        with pytest.raises(ValueError, match="output must be a 1D array-like"):
            X = np.random.normal(size=(10,))
            y = np.random.normal(size=(10,))
            compute_stat = np.corrcoef  # Returns a 2D array
            n_bootstrap = 10
            results = bootstrap_sample(X=X, y=y, compute_stat=compute_stat, n_bootstrap=n_bootstrap)
            assert len(results) == n_bootstrap
            
        

