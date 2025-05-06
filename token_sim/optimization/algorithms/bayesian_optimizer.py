import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import logging

class BayesianOptimizer:
    """Bayesian Optimization for core tokenomics parameters."""
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_initial_points: int = 5,
        acquisition_function: str = 'ei',
        random_state: Optional[int] = None
    ):
        """
        Initialize the Bayesian Optimizer for tokenomics parameters.
        
        Args:
            param_bounds: Dictionary of parameter names and their bounds (min, max)
            n_initial_points: Number of random points to sample initially
            acquisition_function: Type of acquisition function ('ei', 'ucb', or 'pi')
            random_state: Random seed for reproducibility
        """
        self.param_bounds = param_bounds
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        self.random_state = random_state
        
        # Initialize Gaussian Process
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            random_state=random_state
        )
        
        # Storage for observations
        self.X_observed = []
        self.y_observed = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _random_sample(self) -> Dict[str, float]:
        """Generate a random sample within parameter bounds."""
        return {
            param: np.random.uniform(bounds[0], bounds[1])
            for param, bounds in self.param_bounds.items()
        }

    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Calculate Expected Improvement acquisition function."""
        mean, std = self.gp.predict(X, return_std=True)
        best_f = np.max(self.y_observed)
        
        # Calculate improvement
        improvement = mean - best_f - xi
        Z = improvement / (std + 1e-9)
        
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        return ei

    def _upper_confidence_bound(self, X: np.ndarray, kappa: float = 2.0) -> np.ndarray:
        """Calculate Upper Confidence Bound acquisition function."""
        mean, std = self.gp.predict(X, return_std=True)
        return mean + kappa * std

    def _probability_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Calculate Probability of Improvement acquisition function."""
        mean, std = self.gp.predict(X, return_std=True)
        best_f = np.max(self.y_observed)
        
        Z = (mean - best_f - xi) / (std + 1e-9)
        return norm.cdf(Z)

    def suggest_next_point(self, n_candidates: int = 1000) -> Dict[str, float]:
        """Suggest the next point to evaluate."""
        if len(self.X_observed) < self.n_initial_points:
            return self._random_sample()
        
        # Generate candidate points
        candidates = []
        for _ in range(n_candidates):
            candidates.append(self._random_sample())
        
        # Convert to array format
        X_candidates = np.array([[c[param] for param in self.param_bounds.keys()]
                               for c in candidates])
        
        # Calculate acquisition function values
        if self.acquisition_function == 'ei':
            acq_values = self._expected_improvement(X_candidates)
        elif self.acquisition_function == 'ucb':
            acq_values = self._upper_confidence_bound(X_candidates)
        elif self.acquisition_function == 'pi':
            acq_values = self._probability_improvement(X_candidates)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
        
        # Select best candidate
        best_idx = np.argmax(acq_values)
        return candidates[best_idx]

    def update(self, X: Dict[str, float], y: float):
        """Update the model with new observation."""
        # Convert X to array format
        X_array = np.array([[X[param] for param in self.param_bounds.keys()]])
        
        # Update observations
        self.X_observed.append(X_array[0])
        self.y_observed.append(y)
        
        # Update GP model
        self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
        
        self.logger.info(f"Updated model with observation: {y:.4f}")

    def get_best_parameters(self) -> Tuple[Dict[str, float], float]:
        """Get the best parameters found so far."""
        if not self.y_observed:
            raise ValueError("No observations available")
        
        best_idx = np.argmax(self.y_observed)
        best_params = {
            param: self.X_observed[best_idx][i]
            for i, param in enumerate(self.param_bounds.keys())
        }
        return best_params, self.y_observed[best_idx]

    def optimize(
        self,
        objective_function,
        n_iterations: int,
        callback: Optional[callable] = None
    ) -> Tuple[Dict[str, float], float]:
        """
        Run the optimization process.
        
        Args:
            objective_function: Function that takes parameters and returns objective value
            n_iterations: Number of optimization iterations
            callback: Optional callback function called after each iteration
        
        Returns:
            Tuple of (best parameters, best objective value)
        """
        for i in range(n_iterations):
            self.logger.info(f"Starting iteration {i+1}/{n_iterations}")
            
            # Get next point to evaluate
            next_point = self.suggest_next_point()
            
            # Evaluate objective
            objective_value = objective_function(next_point)
            
            # Update model
            self.update(next_point, objective_value)
            
            # Call callback if provided
            if callback:
                callback(i, next_point, objective_value)
            
            self.logger.info(f"Iteration {i+1} completed. Objective value: {objective_value:.4f}")
        
        return self.get_best_parameters()
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """Calculate parameter importance based on GP kernel length scales."""
        if not hasattr(self.gp, 'kernel_'):
            raise ValueError("GP not fitted yet. Run optimize() first.")
        
        length_scales = self.gp.kernel_.length_scale
        importance = 1.0 / length_scales
        importance = importance / np.sum(importance)  # Normalize
        
        return {param: imp for param, imp in zip(self.param_bounds.keys(), importance)} 