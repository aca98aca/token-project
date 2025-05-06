import pytest
import numpy as np
from token_sim.optimization.algorithms.genetic_optimizer import GeneticOptimizer, Individual
from token_sim.optimization.utils.parameter_space import ParameterSpace

def test_genetic_optimizer_initialization():
    """Test genetic optimizer initialization and basic functionality."""
    # Initialize parameter space
    param_space = ParameterSpace()
    
    # Create optimizer instance
    optimizer = GeneticOptimizer(
        param_bounds=param_space.get_param_bounds(),
        population_size=10,
        n_generations=5,
        mutation_rate=0.1
    )
    
    # Test individual creation
    individual = optimizer._create_random_individual()
    assert isinstance(individual, Individual)
    assert isinstance(individual.parameters, dict)
    assert len(individual.parameters) > 0
    
    # Test population initialization
    optimizer._initialize_population()
    assert len(optimizer.population) == 10
    
    # Test tournament selection
    selected = optimizer._tournament_selection()
    assert isinstance(selected, Individual)
    
    # Test crossover
    parent1 = optimizer._create_random_individual()
    parent2 = optimizer._create_random_individual()
    child = optimizer._crossover(parent1, parent2)
    assert isinstance(child, Individual)
    assert len(child.parameters) == len(parent1.parameters)
    
    # Test mutation
    original_params = child.parameters.copy()
    optimizer._mutate(child)
    assert any(original_params[param] != child.parameters[param] for param in child.parameters)
    
    # Test optimization
    def test_objective(params):
        return -sum(params.values())  # Simple test objective
    
    result = optimizer.optimize(
        objective=test_objective,
        n_iterations=5
    )
    
    assert isinstance(result, dict)
    assert 'best_params' in result
    assert 'best_score' in result
    assert 'history' in result 