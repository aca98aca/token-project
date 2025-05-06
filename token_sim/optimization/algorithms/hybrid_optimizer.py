from typing import Dict, List, Tuple, Any, Callable
import numpy as np
from .bayesian_optimizer import BayesianOptimizer
from .genetic_optimizer import GeneticOptimizer
import logging

class HybridOptimizer:
    """Hybrid optimization combining Bayesian and Genetic approaches."""
    
    def __init__(self,
                 param_bounds: Dict[str, Tuple[float, float]],
                 n_initial_points: int = 5,
                 n_iterations: int = 50,
                 population_size: int = 50,
                 n_generations: int = 100,
                 exploration_weight: float = 0.1,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 tournament_size: int = 3):
        """
        Initialize the Hybrid Optimizer.
        
        Args:
            param_bounds: Dictionary of parameter names and their bounds (min, max)
            n_initial_points: Number of random points for Bayesian optimization
            n_iterations: Number of Bayesian optimization iterations
            population_size: Size of genetic algorithm population
            n_generations: Number of genetic algorithm generations
            exploration_weight: Weight for exploration vs exploitation in Bayesian optimization
            mutation_rate: Probability of mutation in genetic algorithm
            crossover_rate: Probability of crossover in genetic algorithm
            tournament_size: Size of tournament for selection in genetic algorithm
        """
        self.param_bounds = param_bounds
        
        # Initialize Bayesian optimizer
        self.bayesian_optimizer = BayesianOptimizer(
            param_bounds=param_bounds,
            n_initial_points=n_initial_points,
            n_iterations=n_iterations,
            exploration_weight=exploration_weight
        )
        
        # Initialize Genetic optimizer
        self.genetic_optimizer = GeneticOptimizer(
            param_bounds=param_bounds,
            population_size=population_size,
            n_generations=n_generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            tournament_size=tournament_size
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Storage for best solutions
        self.best_bayesian_params = None
        self.best_bayesian_score = float('-inf')
        self.best_genetic_params = None
        self.best_genetic_score = float('-inf')
        self.best_overall_params = None
        self.best_overall_score = float('-inf')
    
    def _update_best_solutions(self,
                             bayesian_params: Dict[str, float],
                             bayesian_score: float,
                             genetic_params: Dict[str, float],
                             genetic_score: float):
        """Update the best solutions found by each optimizer."""
        # Update Bayesian best
        if bayesian_score > self.best_bayesian_score:
            self.best_bayesian_params = bayesian_params
            self.best_bayesian_score = bayesian_score
        
        # Update Genetic best
        if genetic_score > self.best_genetic_score:
            self.best_genetic_params = genetic_params
            self.best_genetic_score = genetic_score
        
        # Update overall best
        if bayesian_score > self.best_overall_score:
            self.best_overall_params = bayesian_params
            self.best_overall_score = bayesian_score
        if genetic_score > self.best_overall_score:
            self.best_overall_params = genetic_params
            self.best_overall_score = genetic_score
    
    def optimize(self, objective_function: Callable) -> Tuple[Dict[str, float], float]:
        """
        Run the hybrid optimization process.
        
        Args:
            objective_function: Function that takes parameter dictionary and returns objective value
            
        Returns:
            Tuple of (best parameters, best objective value)
        """
        self.logger.info("Starting hybrid optimization...")
        
        # Run Bayesian optimization
        self.logger.info("Running Bayesian optimization...")
        bayesian_params, bayesian_score = self.bayesian_optimizer.optimize(objective_function)
        
        # Run Genetic optimization
        self.logger.info("Running Genetic optimization...")
        genetic_params, genetic_score = self.genetic_optimizer.optimize(objective_function)
        
        # Update best solutions
        self._update_best_solutions(
            bayesian_params, bayesian_score,
            genetic_params, genetic_score
        )
        
        # Log results
        self.logger.info("Optimization complete.")
        self.logger.info(f"Best Bayesian parameters: {self.best_bayesian_params}")
        self.logger.info(f"Best Bayesian score: {self.best_bayesian_score}")
        self.logger.info(f"Best Genetic parameters: {self.best_genetic_params}")
        self.logger.info(f"Best Genetic score: {self.best_genetic_score}")
        self.logger.info(f"Best overall parameters: {self.best_overall_params}")
        self.logger.info(f"Best overall score: {self.best_overall_score}")
        
        return self.best_overall_params, self.best_overall_score
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get statistics about the optimization process."""
        return {
            'bayesian': {
                'best_params': self.best_bayesian_params,
                'best_score': self.best_bayesian_score,
                'parameter_importance': self.bayesian_optimizer.get_parameter_importance()
            },
            'genetic': {
                'best_params': self.best_genetic_params,
                'best_score': self.best_genetic_score,
                'population_stats': self.genetic_optimizer.get_population_statistics()
            },
            'overall': {
                'best_params': self.best_overall_params,
                'best_score': self.best_overall_score
            }
        } 