import pytest
import numpy as np
from token_sim.optimization.utils.parameter_space import ParameterSpace

def test_parameter_space_initialization():
    """Test parameter space initialization and basic functionality."""
    param_space = ParameterSpace()
    
    # Test parameter bounds
    bounds = param_space.get_param_bounds()
    assert isinstance(bounds, dict)
    assert len(bounds) > 0
    
    # Test default parameters
    default_params = param_space.get_default_params()
    assert isinstance(default_params, dict)
    assert len(default_params) == len(bounds)
    
    # Test parameter validation
    valid_params = param_space.validate_parameters(default_params)
    assert valid_params == default_params
    
    # Test parameter bounds validation
    for param, (min_val, max_val) in bounds.items():
        assert min_val < max_val
        assert default_params[param] >= min_val
        assert default_params[param] <= max_val

def test_parameter_space_validation():
    """Test parameter space validation functionality."""
    param_space = ParameterSpace()
    bounds = param_space.get_param_bounds()
    
    # Test with valid parameters
    valid_params = param_space.get_default_params()
    assert param_space.validate_parameters(valid_params) == valid_params
    
    # Test with invalid parameters (out of bounds)
    invalid_params = valid_params.copy()
    for param, (min_val, max_val) in bounds.items():
        invalid_params[param] = max_val + 1
        with pytest.raises(ValueError):
            param_space.validate_parameters(invalid_params)
        invalid_params[param] = min_val - 1
        with pytest.raises(ValueError):
            param_space.validate_parameters(invalid_params)
    
    # Test with missing parameters
    missing_params = valid_params.copy()
    del missing_params[list(bounds.keys())[0]]
    with pytest.raises(ValueError):
        param_space.validate_parameters(missing_params)
    
    # Test with extra parameters
    extra_params = valid_params.copy()
    extra_params['extra_param'] = 1.0
    with pytest.raises(ValueError):
        param_space.validate_parameters(extra_params) 