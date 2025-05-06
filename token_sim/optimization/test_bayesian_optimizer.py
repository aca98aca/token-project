import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_sim.optimization.algorithms.bayesian_optimizer import BayesianOptimizer
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

def test_simple_function():
    """Test the Bayesian optimizer on a simple function."""
    # Define a simple 2D test function (quadratic)
    def objective_function(params: Dict[str, float]) -> float:
        x = params['x']
        y = params['y']
        return -(x - 2)**2 - (y - 3)**2 + 5  # Maximum at (2, 3) with value 5
    
    # Define parameter bounds
    param_bounds = {
        'x': (-5.0, 5.0),
        'y': (-5.0, 5.0)
    }
    
    # Initialize optimizer
    optimizer = BayesianOptimizer(
        param_bounds=param_bounds,
        n_initial_points=5,
        acquisition_function='ei',
        random_state=42
    )
    
    # Run optimization
    best_params, best_value = optimizer.optimize(
        objective_function=objective_function,
        n_iterations=20
    )
    
    print("\nTest Simple Function Results:")
    print(f"True optimum: x=2, y=3, value=5")
    print(f"Found optimum: x={best_params['x']:.4f}, y={best_params['y']:.4f}, value={best_value:.4f}")
    
    # Calculate error
    error_x = abs(best_params['x'] - 2)
    error_y = abs(best_params['y'] - 3)
    error_value = abs(best_value - 5)
    
    print(f"Errors: x={error_x:.4f}, y={error_y:.4f}, value={error_value:.4f}")
    
    return optimizer

def plot_optimization_progress(optimizer: BayesianOptimizer):
    """Plot the optimization progress."""
    try:
        # Convert observed points to numpy arrays
        X = np.array(optimizer.X_observed)
        y = np.array(optimizer.y_observed)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot points in parameter space
        scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Sampled Points in Parameter Space')
        fig.colorbar(scatter, ax=ax1, label='Objective Value')
        
        # Mark the best point
        best_idx = np.argmax(y)
        ax1.scatter([X[best_idx, 0]], [X[best_idx, 1]], c='red', s=100, marker='*', 
                    label=f'Best: ({X[best_idx, 0]:.2f}, {X[best_idx, 1]:.2f})')
        ax1.legend()
        
        # Plot objective value over iterations
        iterations = range(len(y))
        ax2.plot(iterations, y, 'o-', label='Objective Value')
        ax2.plot(iterations, np.maximum.accumulate(y), 'r--', label='Best Value')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Objective Value')
        ax2.set_title('Optimization Progress')
        ax2.legend()
        
        # Improve layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig('bayesian_optimization_test.png')
        print("Plot saved to bayesian_optimization_test.png")
        
    except Exception as e:
        print(f"Error creating plot: {str(e)}")

if __name__ == "__main__":
    print("Testing Bayesian Optimizer...")
    optimizer = test_simple_function()
    plot_optimization_progress(optimizer)
    
    print("\nDone!") 