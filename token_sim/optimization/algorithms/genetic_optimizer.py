import numpy as np
from typing import Dict, List, Tuple, Any, Callable
import random
from dataclasses import dataclass
import logging

@dataclass
class Individual:
    """Represents a single solution in the genetic algorithm."""
    parameters: Dict[str, float]
    fitness: float = 0.0
    objectives: Dict[str, float] = None

class GeneticOptimizer:
    """Genetic Algorithm for tokenomics parameter optimization."""
    
    def __init__(self,
                 param_bounds: Dict[str, Tuple[float, float]],
                 population_size: int = 50,
                 n_generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 tournament_size: int = 3):
        """
        Initialize the Genetic Optimizer.
        
        Args:
            param_bounds: Dictionary of parameter names and their bounds (min, max)
            population_size: Size of the population
            n_generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament for selection
        """
        self.param_bounds = param_bounds
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize population
        self.population: List[Individual] = []
        self.best_solution: Individual = None
    
    def _create_random_individual(self) -> Individual:
        """Create a random individual within parameter bounds."""
        parameters = {}
        for param, (min_val, max_val) in self.param_bounds.items():
            parameters[param] = np.random.uniform(min_val, max_val)
        return Individual(parameters=parameters)
    
    def _initialize_population(self):
        """Initialize the population with random individuals."""
        self.population = [self._create_random_individual() for _ in range(self.population_size)]
    
    def _tournament_selection(self) -> Individual:
        """Select an individual using tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        child1_params = {}
        child2_params = {}
        
        for param in self.param_bounds.keys():
            if random.random() < 0.5:
                child1_params[param] = parent1.parameters[param]
                child2_params[param] = parent2.parameters[param]
            else:
                child1_params[param] = parent2.parameters[param]
                child2_params[param] = parent1.parameters[param]
        
        return Individual(parameters=child1_params), Individual(parameters=child2_params)
    
    def _mutate(self, individual: Individual):
        """Apply mutation to an individual."""
        for param, (min_val, max_val) in self.param_bounds.items():
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation = np.random.normal(0, (max_val - min_val) * 0.1)
                individual.parameters[param] = np.clip(
                    individual.parameters[param] + mutation,
                    min_val,
                    max_val
                )
    
    def _evaluate_population(self, objective_function: Callable):
        """Evaluate the fitness of all individuals in the population."""
        for individual in self.population:
            if individual.fitness == 0.0:  # Only evaluate if not already evaluated
                individual.fitness = objective_function(individual.parameters)
    
    def optimize(self, objective_function: Callable) -> Tuple[Dict[str, float], float]:
        """
        Run the genetic optimization process.
        
        Args:
            objective_function: Function that takes parameter dictionary and returns objective value
            
        Returns:
            Tuple of (best parameters, best objective value)
        """
        self.logger.info("Starting genetic optimization...")
        
        # Initialize population
        self._initialize_population()
        
        # Main evolution loop
        for generation in range(self.n_generations):
            self.logger.info(f"Generation {generation + 1}/{self.n_generations}")
            
            # Evaluate population
            self._evaluate_population(objective_function)
            
            # Update best solution
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.best_solution is None or current_best.fitness > self.best_solution.fitness:
                self.best_solution = current_best
            
            # Create new population
            new_population = []
            
            # Elitism: Keep the best individual
            new_population.append(self.best_solution)
            
            # Generate rest of new population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                self._mutate(child1)
                self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Update population
            self.population = new_population[:self.population_size]
            
            self.logger.info(f"Best fitness: {self.best_solution.fitness}")
        
        self.logger.info("Optimization complete.")
        self.logger.info(f"Best parameters: {self.best_solution.parameters}")
        self.logger.info(f"Best fitness: {self.best_solution.fitness}")
        
        return self.best_solution.parameters, self.best_solution.fitness
    
    def get_population_statistics(self) -> Dict[str, float]:
        """Get statistics about the current population."""
        if not self.population:
            return {}
        
        fitnesses = [ind.fitness for ind in self.population]
        return {
            'mean_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses)
        } 