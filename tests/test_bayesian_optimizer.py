import pytest
import numpy as np
from token_sim.optimization.algorithms.bayesian_optimizer import BayesianOptimizer
from token_sim.optimization.utils.parameter_space import ParameterSpace

def test_bayesian_optimizer_initialization():
    """Test Bayesian optimizer initialization and basic functionality."""
    # Initialize parameter space
    param_space = ParameterSpace()
    
    # Create optimizer instance
    optimizer = BayesianOptimizer(
        param_bounds=param_space.get_param_bounds(),
        n_initial_points=3
    )
    
    # Test random sampling
    sample = optimizer._random_sample()
    assert isinstance(sample, dict)
    assert len(sample) == len(param_space.get_param_bounds())
    
    # Test parameter conversion
    default_params = param_space.get_default_params()
    param_array = np.array([default_params[param] for param in param_space.get_param_bounds().keys()])
    assert len(param_array) == len(param_space.get_param_bounds())
    
    # Test objective function evaluation
    def test_objective(params):
        return -sum(params.values())  # Simple test objective
    
    result = optimizer.optimize(
        objective_function=test_objective,
        n_iterations=5
    )
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], dict)  # best parameters
    assert isinstance(result[1], float)  # best score

def test_bayesian_optimizer_acquisition_functions():
    """Test different acquisition functions."""
    param_space = ParameterSpace()
    
    # Test EI acquisition function
    optimizer_ei = BayesianOptimizer(
        param_bounds=param_space.get_param_bounds(),
        n_initial_points=3,
        acquisition_function='ei'
    )
    
    # Test UCB acquisition function
    optimizer_ucb = BayesianOptimizer(
        param_bounds=param_space.get_param_bounds(),
        n_initial_points=3,
        acquisition_function='ucb'
    )
    
    # Test PI acquisition function
    optimizer_pi = BayesianOptimizer(
        param_bounds=param_space.get_param_bounds(),
        n_initial_points=3,
        acquisition_function='pi'
    )
    
    def test_objective(params):
        return -sum(params.values())
    
    # Test optimization with each acquisition function
    for optimizer in [optimizer_ei, optimizer_ucb, optimizer_pi]:
        result = optimizer.optimize(
            objective_function=test_objective,
            n_iterations=5
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], float)

def test_bayesian_optimizer_parameter_importance():
    """Test parameter importance calculation."""
    param_space = ParameterSpace()
    
    optimizer = BayesianOptimizer(
        param_bounds=param_space.get_param_bounds(),
        n_initial_points=3
    )
    
    def test_objective(params):
        return -sum(params.values())
    
    # Run optimization first
    optimizer.optimize(
        objective_function=test_objective,
        n_iterations=5
    )
    
    # Test parameter importance
    importance = optimizer.get_parameter_importance()
    assert isinstance(importance, dict)
    assert len(importance) == len(param_space.get_param_bounds())
    assert all(0 <= imp <= 1 for imp in importance.values())
    assert abs(sum(importance.values()) - 1.0) < 1e-6  # Should sum to 1 