import numpy as np
from typing import Dict, List, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim

class ConsensusOptimizer:
    """Bayesian Optimization for consensus parameters."""
    
    def __init__(self, simulation, param_bounds: Dict[str, tuple]):
        self.simulation = simulation
        self.param_bounds = param_bounds
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=10
        )
        self.X = []  # Parameter combinations
        self.y = []  # Performance scores
    
    def optimize(self, n_iterations: int = 50) -> Dict[str, float]:
        """Optimize consensus parameters using Bayesian Optimization."""
        for i in range(n_iterations):
            # Get next parameters to try
            if len(self.X) == 0:
                params = self._random_params()
            else:
                params = self._get_next_params()
            
            # Run simulation with parameters
            score = self._evaluate_params(params)
            
            # Update GP
            self.X.append(list(params.values()))
            self.y.append(score)
            self.gp.fit(np.array(self.X), np.array(self.y))
            
            print(f"Iteration {i+1}/{n_iterations}")
            print(f"Parameters: {params}")
            print(f"Score: {score:.4f}")
        
        # Return best parameters
        best_idx = np.argmax(self.y)
        return dict(zip(self.param_bounds.keys(), self.X[best_idx]))
    
    def _random_params(self) -> Dict[str, float]:
        """Generate random parameters within bounds."""
        return {
            name: np.random.uniform(bounds[0], bounds[1])
            for name, bounds in self.param_bounds.items()
        }
    
    def _get_next_params(self) -> Dict[str, float]:
        """Get next parameters using Upper Confidence Bound (UCB)."""
        # Generate random candidates
        n_candidates = 1000
        candidates = np.array([
            self._random_params()
            for _ in range(n_candidates)
        ])
        
        # Predict mean and std for each candidate
        mean, std = self.gp.predict(candidates, return_std=True)
        
        # Calculate UCB
        ucb = mean + 2 * std
        
        # Select best candidate
        best_idx = np.argmax(ucb)
        return dict(zip(self.param_bounds.keys(), candidates[best_idx]))
    
    def _evaluate_params(self, params: Dict[str, float]) -> float:
        """Evaluate parameters by running simulation."""
        # Update simulation parameters
        self.simulation.update_params(params)
        
        # Run simulation
        history = self.simulation.run()
        
        # Calculate performance score
        score = self._calculate_score(history)
        
        return score
    
    def _calculate_score(self, history: Dict[str, List]) -> float:
        """Calculate performance score from simulation history."""
        # Price stability
        price_volatility = np.std(history['price']) / np.mean(history['price'])
        price_stability = 1 / (1 + price_volatility)
        
        # Network security
        security_score = np.mean(history['security_score'])
        
        # Trading volume
        volume_score = np.mean(history['volume']) / np.max(history['volume'])
        
        # Token distribution
        distribution_score = 1 - np.mean(history['gini_coefficient'])
        
        # Combine scores
        return (0.3 * price_stability + 
                0.3 * security_score + 
                0.2 * volume_score + 
                0.2 * distribution_score)

class NeuralConsensus:
    """Neural network for consensus parameter prediction."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64):
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
    
    def train(self, X: torch.Tensor, y: torch.Tensor):
        """Train the model."""
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict(self, X: torch.Tensor):
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            return self.model(X)
    
    def save(self, path: str):
        """Save the model."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        """Load the model."""
        self.model.load_state_dict(torch.load(path)) 