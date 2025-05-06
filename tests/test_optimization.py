import pytest
import numpy as np
from token_sim.optimization.algorithms.bayesian_optimizer import BayesianOptimizer
from token_sim.optimization.algorithms.genetic_optimizer import GeneticOptimizer, Individual
from token_sim.optimization.utils.parameter_space import ParameterSpace
from token_sim.optimization.evaluation.metrics import TokenomicsMetrics
import copy

def test_parameter_space():
    """Test parameter space initialization and validation."""
    param_space = ParameterSpace()
    
    # Test getting all parameters
    all_params = param_space.get_all_params()
    assert len(all_params) > 0
    
    # Test parameter bounds
    bounds = param_space.get_param_bounds()
    assert 'block_reward' in bounds
    assert len(bounds['block_reward']) == 2
    
    # Test default parameters
    defaults = param_space.get_default_params()
    assert 'block_reward' in defaults
    assert defaults['block_reward'] == 50.0
    
    # Test parameter validation
    valid_params = {
        'block_reward': 50.0,
        'initial_supply': 1000000.0,
        'market_depth': 1000000.0
    }
    assert param_space.validate_params(valid_params)
    
    # Test invalid parameters
    invalid_params = {
        'block_reward': -1.0,  # Below minimum
        'initial_supply': 1000000.0,
        'market_depth': 1000000.0
    }
    assert not param_space.validate_params(invalid_params)

def test_bayesian_optimizer():
    """Test Bayesian optimizer initialization and basic functionality."""
    param_space = ParameterSpace()
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

def test_metrics():
    """Test metrics calculation and evaluation."""
    metrics = TokenomicsMetrics()
    
    # Test with sample simulation results
    simulation_results = {
        'consensus_type': 'pos',
        'num_participants': 500,
        'staking_ratio': 0.7,
        'price_history': [100.0, 101.0, 99.0, 100.5, 100.2],
        'token_balances': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        'market_depth': 500000.0,
        'trading_volume': 50000.0
    }
    
    # Test individual metric calculations
    security_score = metrics.calculate_network_security(simulation_results)
    assert 0 <= security_score <= 1
    
    stability_score = metrics.calculate_price_stability(simulation_results)
    assert 0 <= stability_score <= 1
    
    distribution_score = metrics.calculate_token_distribution(simulation_results)
    assert 0 <= distribution_score <= 1
    
    liquidity_score = metrics.calculate_market_liquidity(simulation_results)
    assert 0 <= liquidity_score <= 1
    
    # Test full evaluation
    results = metrics.evaluate(simulation_results)
    assert len(results) > 0
    
    # Test overall score calculation
    overall_score = metrics.calculate_overall_score(results)
    assert 0 <= overall_score <= 1

def test_optimization_workflow():
    """Test the complete optimization workflow."""
    param_space = ParameterSpace()
    optimizer = BayesianOptimizer(
        param_bounds=param_space.get_param_bounds(),
        n_initial_points=3
    )
    metrics = TokenomicsMetrics()
    
    def objective_function(params):
        """Simple objective function for testing."""
        simulation_results = {
            'consensus_type': 'pos',
            'num_participants': int(params.get('num_participants', 100)),
            'staking_ratio': 0.7,
            'price_history': [100.0, 101.0, 99.0, 100.5, 100.2],
            'token_balances': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            'market_depth': params.get('market_depth', 1000000.0),
            'trading_volume': 50000.0
        }
        
        results = metrics.evaluate(simulation_results)
        return metrics.calculate_overall_score(results)
    
    # Run optimization
    best_params, best_score = optimizer.optimize(objective_function, n_iterations=5)
    
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert 0 <= best_score <= 1

def test_genetic_optimizer():
    """Test genetic optimizer initialization and basic functionality."""
    param_space = ParameterSpace()
    optimizer = GeneticOptimizer(
        param_bounds=param_space.get_param_bounds(),
        population_size=10,
        n_generations=5,
        mutation_rate=1.0  # Set to 1.0 to ensure mutation happens
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
    
    # Test crossover with actual parameter names
    default_params = param_space.get_default_params()
    parent1 = Individual(parameters=default_params)
    parent2 = Individual(parameters={k: v * 1.5 for k, v in default_params.items()})
    child1, child2 = optimizer._crossover(parent1, parent2)
    assert isinstance(child1, Individual)
    assert isinstance(child2, Individual)
    
    # Test mutation
    original_params = copy.deepcopy(individual.parameters)
    optimizer._mutate(individual)
    assert any(original_params[k] != individual.parameters[k] for k in original_params)

def test_genetic_optimization_workflow():
    """Test the complete genetic optimization workflow."""
    param_space = ParameterSpace()
    optimizer = GeneticOptimizer(
        param_bounds=param_space.get_param_bounds(),
        population_size=10,
        n_generations=5
    )
    metrics = TokenomicsMetrics()
    
    def objective_function(params):
        """Simple objective function for testing."""
        simulation_results = {
            'consensus_type': 'pos',
            'num_participants': int(params.get('num_participants', 100)),
            'staking_ratio': 0.7,
            'price_history': [100.0, 101.0, 99.0, 100.5, 100.2],
            'token_balances': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            'market_depth': params.get('market_depth', 1000000.0),
            'trading_volume': 50000.0
        }
        
        results = metrics.evaluate(simulation_results)
        return metrics.calculate_overall_score(results)
    
    # Run optimization
    best_params, best_score = optimizer.optimize(objective_function)
    
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert 0 <= best_score <= 1
    
    # Test population statistics
    stats = optimizer.get_population_statistics()
    assert 'mean_fitness' in stats
    assert 'std_fitness' in stats
    assert 'max_fitness' in stats
    assert 'min_fitness' in stats

def test_optimizer_comparison():
    """Compare Bayesian and Genetic optimizers on the same problem."""
    param_space = ParameterSpace()
    bayesian_optimizer = BayesianOptimizer(
        param_bounds=param_space.get_param_bounds(),
        n_initial_points=3
    )
    genetic_optimizer = GeneticOptimizer(
        param_bounds=param_space.get_param_bounds(),
        population_size=10,
        n_generations=5
    )
    metrics = TokenomicsMetrics()
    
    def objective_function(params):
        """Simple objective function for testing."""
        simulation_results = {
            'consensus_type': 'pos',
            'num_participants': int(params.get('num_participants', 100)),
            'staking_ratio': 0.7,
            'price_history': [100.0, 101.0, 99.0, 100.5, 100.2],
            'token_balances': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            'market_depth': params.get('market_depth', 1000000.0),
            'trading_volume': 50000.0
        }
        
        results = metrics.evaluate(simulation_results)
        return metrics.calculate_overall_score(results)
    
    # Run both optimizers
    bayesian_params, bayesian_score = bayesian_optimizer.optimize(objective_function, n_iterations=5)
    genetic_params, genetic_score = genetic_optimizer.optimize(objective_function)
    
    # Compare results
    assert isinstance(bayesian_params, dict)
    assert isinstance(genetic_params, dict)
    assert isinstance(bayesian_score, float)
    assert isinstance(genetic_score, float)
    assert 0 <= bayesian_score <= 1
    assert 0 <= genetic_score <= 1 