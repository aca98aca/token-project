import pytest
import numpy as np
from token_sim.optimization.algorithms.bayesian_optimizer import BayesianOptimizer
from token_sim.optimization.algorithms.genetic_optimizer import GeneticOptimizer
from token_sim.optimization.utils.parameter_space import ParameterSpace

def test_optimization_workflow():
    """Test the complete optimization workflow."""
    # Initialize parameter space
    param_space = ParameterSpace()
    
    # Create test objective function
    def test_objective(params):
        # Simple test objective that has a known minimum
        return sum((params[param] - 1.0) ** 2 for param in params)
    
    # Test Bayesian optimization
    bayesian_optimizer = BayesianOptimizer(
        param_bounds=param_space.get_param_bounds(),
        n_initial_points=3
    )
    
    bayesian_result = bayesian_optimizer.optimize(
        objective_function=test_objective,
        n_iterations=5
    )
    
    assert isinstance(bayesian_result, tuple)
    assert len(bayesian_result) == 2
    assert isinstance(bayesian_result[0], dict)  # best parameters
    assert isinstance(bayesian_result[1], float)  # best score
    
    # Test Genetic optimization
    genetic_optimizer = GeneticOptimizer(
        param_bounds=param_space.get_param_bounds(),
        population_size=10,
        n_generations=5,
        mutation_rate=0.1
    )
    
    genetic_result = genetic_optimizer.optimize(
        objective_function=test_objective,
        n_iterations=5
    )
    
    assert isinstance(genetic_result, tuple)
    assert len(genetic_result) == 2
    assert isinstance(genetic_result[0], dict)  # best parameters
    assert isinstance(genetic_result[1], float)  # best score
    
    # Compare results
    assert bayesian_result[1] <= 1.0  # Should be close to 0
    assert genetic_result[1] <= 1.0  # Should be close to 0

def test_optimization_convergence():
    """Test optimization convergence behavior."""
    param_space = ParameterSpace()
    
    def test_objective(params):
        # Objective with a clear minimum at (1, 1, ..., 1)
        return sum((params[param] - 1.0) ** 2 for param in params)
    
    # Test Bayesian optimization convergence
    bayesian_optimizer = BayesianOptimizer(
        param_bounds=param_space.get_param_bounds(),
        n_initial_points=3
    )
    
    bayesian_result = bayesian_optimizer.optimize(
        objective_function=test_objective,
        n_iterations=10
    )
    
    # Check if the best parameters are close to the known minimum
    for param, value in bayesian_result[0].items():
        assert abs(value - 1.0) < 0.1
    
    # Test Genetic optimization convergence
    genetic_optimizer = GeneticOptimizer(
        param_bounds=param_space.get_param_bounds(),
        population_size=20,
        n_generations=10,
        mutation_rate=0.1
    )
    
    genetic_result = genetic_optimizer.optimize(
        objective_function=test_objective,
        n_iterations=10
    )
    
    # Check if the best parameters are close to the known minimum
    for param, value in genetic_result[0].items():
        assert abs(value - 1.0) < 0.1 