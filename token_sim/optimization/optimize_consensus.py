import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_sim.optimization.optimize_core_params import TokenomicsAdapter, CoreParameters
from token_sim.optimization.algorithms.bayesian_optimizer import BayesianOptimizer
from typing import Dict, Tuple, List, Any
import numpy as np
import logging
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class ConsensusSpecificParameters:
    """Parameters specific to each consensus type."""
    pow_params: Dict[str, Tuple[float, float]] = None
    pos_params: Dict[str, Tuple[float, float]] = None
    dpos_params: Dict[str, Tuple[float, float]] = None

    def __post_init__(self):
        if self.pow_params is None:
            self.pow_params = {
                'difficulty_adjustment_blocks': (1000, 3000),
                'target_block_time': (300, 900)  # 5-15 minutes
            }
        if self.pos_params is None:
            self.pos_params = {
                'min_stake': (50, 500),
                'staking_apy': (0.01, 0.2)  # 1-20% APY
            }
        if self.dpos_params is None:
            self.dpos_params = {
                'num_delegates': (11, 101),
                'min_stake': (50, 500),
                'staking_apy': (0.01, 0.2)  # 1-20% APY
            }

def get_consensus_specific_params(consensus_type: str) -> Dict[str, Tuple[float, float]]:
    """Get the parameter bounds for a specific consensus type."""
    params = ConsensusSpecificParameters()
    if consensus_type == 'pow':
        return params.pow_params
    elif consensus_type == 'pos':
        return params.pos_params
    elif consensus_type == 'dpos':
        return params.dpos_params
    else:
        raise ValueError(f"Unsupported consensus type: {consensus_type}")

def calculate_decentralization(results: Dict[str, Any]) -> float:
    """Calculate decentralization score based on participant distribution."""
    if 'participant_stats' not in results:
        return 0.0
    
    stats = results['participant_stats']
    if not stats:
        return 0.0
    
    # Calculate Gini coefficient for participant resources
    resources = []
    for participant in stats.values():
        if 'hashrate' in participant:
            resources.append(participant['hashrate'])
        elif 'stake' in participant:
            resources.append(participant['stake'])
    
    if not resources:
        return 0.0
    
    # Sort resources
    resources = sorted(resources)
    n = len(resources)
    
    # Calculate Gini coefficient
    index = np.arange(1, n + 1)
    return ((2 * index - n - 1) * resources).sum() / (n * sum(resources))

def optimize_consensus_type(
    consensus_type: str,
    n_iterations: int = 30,
    n_initial_points: int = 5
) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    """
    Optimize parameters for a specific consensus type.
    
    Args:
        consensus_type: Type of consensus mechanism ('pow', 'pos', 'dpos')
        n_iterations: Number of optimization iterations
        n_initial_points: Number of random initial points
    
    Returns:
        Tuple of (best parameters, best objective value, detailed results)
    """
    # Initialize adapter
    adapter = TokenomicsAdapter(consensus_type=consensus_type)
    
    # Get consensus-specific parameters
    consensus_params = get_consensus_specific_params(consensus_type)
    
    # Define parameter bounds
    param_bounds = {
        'block_reward': (0.1, 10.0),
        'initial_supply': (1e6, 1e8),
        'market_depth': (1e5, 1e7),
        'initial_price': (0.1, 10.0)
    }
    
    # Add consensus-specific parameters
    param_bounds.update(consensus_params)
    
    # Initialize optimizer
    optimizer = BayesianOptimizer(
        param_bounds=param_bounds,
        n_initial_points=n_initial_points,
        acquisition_function='ei',
        random_state=42
    )
    
    # Create objective function
    def objective(params: Dict[str, float]) -> float:
        # Convert parameters to simulation format
        core_params = CoreParameters(
            block_reward=params['block_reward'],
            initial_supply=params['initial_supply'],
            market_depth=params['market_depth'],
            initial_price=params['initial_price']
        )
        
        # Run simulation
        results = adapter.run_with_parameters(core_params)
        
        # Calculate objective value
        price_stability = -abs(results['price_volatility'])
        network_security = results['network_security_score']
        market_liquidity = results['market_liquidity']
        energy_efficiency = results['energy_efficiency']
        
        # Calculate decentralization score
        decentralization = calculate_decentralization(results)
        
        # Combine metrics with weights
        objective_value = (
            0.25 * price_stability +      # 25% weight on price stability
            0.25 * network_security +     # 25% weight on network security
            0.20 * market_liquidity +     # 20% weight on market liquidity
            0.15 * energy_efficiency +    # 15% weight on energy efficiency
            0.15 * decentralization       # 15% weight on decentralization
        )
        
        return objective_value
    
    # Run optimization
    best_params, best_value = optimizer.optimize(
        objective_function=objective,
        n_iterations=n_iterations
    )
    
    # Get detailed results for the best parameters
    core_params = CoreParameters(
        block_reward=best_params['block_reward'],
        initial_supply=best_params['initial_supply'],
        market_depth=best_params['market_depth'],
        initial_price=best_params['initial_price']
    )
    detailed_results = adapter.run_with_parameters(core_params)
    
    return best_params, best_value, detailed_results

def optimize_across_consensus_types(
    n_iterations: int = 30,
    n_initial_points: int = 5
) -> Dict[str, Dict]:
    """
    Optimize parameters across all consensus types and compare results.
    
    Args:
        n_iterations: Number of optimization iterations per consensus type
        n_initial_points: Number of random initial points per consensus type
    
    Returns:
        Dictionary containing results for each consensus type
    """
    results = {}
    consensus_types = ['pow', 'pos', 'dpos']
    
    for consensus_type in consensus_types:
        print(f"\nOptimizing {consensus_type.upper()} consensus...")
        best_params, best_value, detailed_results = optimize_consensus_type(
            consensus_type=consensus_type,
            n_iterations=n_iterations,
            n_initial_points=n_initial_points
        )
        
        results[consensus_type] = {
            'parameters': best_params,
            'objective_value': best_value,
            'detailed_results': detailed_results
        }
        
        print(f"\n{consensus_type.upper()} Results:")
        print(f"Best Parameters: {best_params}")
        print(f"Objective Value: {best_value:.4f}")
        print(f"Price Volatility: {detailed_results['price_volatility']:.2f}%")
        print(f"Network Security: {detailed_results['network_security_score']:.2f}")
        print(f"Market Liquidity: {detailed_results['market_liquidity']:.2f}")
        print(f"Energy Efficiency: {detailed_results['energy_efficiency']:.2f}")
    
    return results

def save_results(results: Dict[str, Dict], filename: str = None):
    """Save optimization results to a JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"consensus_optimization_results_{timestamp}.json"
    
    # Convert numpy types to Python native types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results = convert_numpy(results)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Optimize tokenomics parameters across consensus types')
    parser.add_argument('--iterations', type=int, default=30,
                       help='Number of optimization iterations per consensus type')
    parser.add_argument('--initial-points', type=int, default=5,
                       help='Number of random initial points per consensus type')
    parser.add_argument('--output', type=str,
                       help='Output file for results (optional)')
    args = parser.parse_args()
    
    # Run optimization across all consensus types
    results = optimize_across_consensus_types(
        n_iterations=args.iterations,
        n_initial_points=args.initial_points
    )
    
    # Save results
    save_results(results, args.output)
    
    # Print comparison
    print("\nConsensus Type Comparison:")
    print("-" * 50)
    for consensus_type, data in results.items():
        print(f"\n{consensus_type.upper()}:")
        print(f"Objective Value: {data['objective_value']:.4f}")
        print(f"Price Volatility: {data['detailed_results']['price_volatility']:.2f}%")
        print(f"Network Security: {data['detailed_results']['network_security_score']:.2f}")
        print(f"Market Liquidity: {data['detailed_results']['market_liquidity']:.2f}")
        print(f"Energy Efficiency: {data['detailed_results']['energy_efficiency']:.2f}")
    
    # Find best consensus type
    best_consensus = max(results.items(), key=lambda x: x[1]['objective_value'])
    print(f"\nBest Consensus Type: {best_consensus[0].upper()}")
    print(f"Objective Value: {best_consensus[1]['objective_value']:.4f}") 