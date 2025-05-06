import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_sim.optimization.algorithms.bayesian_optimizer import BayesianOptimizer
from typing import Dict, Tuple, List, Any
import numpy as np
import logging
from dataclasses import dataclass
from token_sim.simulation import TokenSimulation
from token_sim.consensus import ConsensusMechanism 
from token_sim.market.price_discovery import PriceDiscovery, SimplePriceDiscovery
from token_sim.agents import Agent
import math

# Create an adapter class to handle our core parameters
class TokenomicsAdapter:
    """Adapter class to interface with TokenSimulation using our core parameters."""
    
    def __init__(self, consensus_type: str = 'pow'):
        """
        Initialize the adapter.
        
        Args:
            consensus_type: Type of consensus mechanism ('pow', 'pos', 'dpos')
        """
        self.consensus_type = consensus_type
        self.simulation = None
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing TokenomicsAdapter with consensus type: {consensus_type}")
        
    def _create_simulation(self, core_params: 'CoreParameters') -> TokenSimulation:
        """Create a TokenSimulation instance with the given parameters."""
        from token_sim.consensus.pow import ProofOfWork
        from token_sim.consensus.pos import ProofOfStake
        from token_sim.consensus.dpos import DelegatedProofOfStake
        from token_sim.agents.miner import Miner
        from token_sim.agents.holder import Holder
        
        self.logger.info(f"Creating simulation with parameters: {core_params}")
        
        # Create consensus mechanism
        if self.consensus_type == 'pow':
            consensus = ProofOfWork(
                block_reward=core_params.block_reward,
                difficulty_adjustment_blocks=2016,
                target_block_time=600.0
            )
        elif self.consensus_type == 'pos':
            consensus = ProofOfStake(
                block_reward=core_params.block_reward,
                min_stake=100.0,
                staking_apy=0.05
            )
        elif self.consensus_type == 'dpos':
            consensus = DelegatedProofOfStake(
                block_reward=core_params.block_reward,
                min_stake=100.0,
                num_delegates=21,
                staking_apy=0.05
            )
        else:
            raise ValueError(f"Unsupported consensus type: {self.consensus_type}")
        
        # Create price discovery mechanism
        price_discovery = SimplePriceDiscovery(
            initial_price=core_params.initial_price,
            market_depth=core_params.market_depth,
            volatility=0.05
        )
        
        # Create agents
        agents = self._create_agents(core_params, consensus)
        
        # Create simulation
        simulation = TokenSimulation(
            consensus=consensus,
            price_discovery=price_discovery,
            agents=agents,
            initial_supply=core_params.initial_supply,
            time_steps=500  # Reduced for optimization speed
        )
        
        return simulation
    
    def _create_agents(self, core_params: 'CoreParameters', consensus: ConsensusMechanism) -> List[Agent]:
        """Create agents for the simulation."""
        from token_sim.agents.miner import Miner
        from token_sim.agents.holder import Holder
        
        agents = []
        
        # Create miners or validators
        num_miners = 25  # Reduced for optimization speed
        for i in range(num_miners):
            if self.consensus_type == 'pow':
                agent = Miner(
                    agent_id=f"miner_{i}",
                    initial_balance=1000.0,
                    initial_hashrate=100 + np.random.randint(0, 200),
                    strategy="opportunistic"
                )
            else:
                agent = Holder(
                    agent_id=f"validator_{i}",
                    initial_balance=1000.0,
                    strategy="long_term"
                )
            agents.append(agent)
        
        # Create token holders
        num_holders = 50  # Reduced for optimization speed
        for i in range(num_holders):
            agent = Holder(
                agent_id=f"holder_{i}",
                initial_balance=500.0,
                strategy=np.random.choice(["long_term", "medium_term", "short_term"])
            )
            agents.append(agent)
        
        self.logger.info(f"Created {len(agents)} agents: {num_miners} miners/validators, {num_holders} holders")
        return agents
    
    def run_with_parameters(self, core_params: 'CoreParameters') -> Dict[str, Any]:
        """Run the simulation with the given core parameters."""
        try:
            self.simulation = self._create_simulation(core_params)
            
            # Run the simulation
            history = self.simulation.run()
            
            # Calculate metrics
            price_history = np.array(history['price'])
            volumes = np.array(history.get('volume', []))
            
            if len(price_history) > 1 and len(volumes) > 0:
                # Calculate returns
                returns = np.diff(price_history) / price_history[:-1]
                
                # Calculate volume-weighted price volatility
                total_volume = np.sum(volumes[1:])
                if total_volume > 0:
                    weights = volumes[1:] / total_volume
                    weighted_variance = np.sum(weights * (returns - np.mean(returns))**2)
                    price_volatility = np.sqrt(weighted_variance) * 100
                else:
                    # Fallback to standard volatility if no volume
                    price_volatility = np.std(returns) * 100
                
                # Apply dampening based on market depth
                depth_factor = min(core_params.market_depth / core_params.initial_supply, 1.0)
                price_volatility *= (1.0 - 0.5 * depth_factor)  # Reduce volatility by up to 50% based on depth
            else:
                price_volatility = 0.0
            
            # Security score (based on active miners/validators and their resources)
            network_security = np.array(history.get('network_security', []))
            
            if len(network_security) > 0:
                network_security_score = np.mean(network_security)
            else:
                network_security_score = 0.0
            
            # Liquidity (based on trading volume, depth, and spreads)
            trades = history.get('trades', [])
            
            if len(volumes) > 0 and trades:
                # Volume score (0-1)
                volume_score = min(np.mean(volumes) / (core_params.market_depth * 0.1), 1.0)
                
                # Market depth score
                depth_ratio = core_params.market_depth / core_params.initial_supply
                depth_score = min(depth_ratio * 2, 1.0)  # Target 50% of supply as depth
                
                # Bid-ask spread score from trades
                spreads = []
                for trade_list in trades:
                    if trade_list:  # Check if there are trades in this step
                        prices = [t['price'] for t in trade_list]
                        if prices:
                            spread = (max(prices) - min(prices)) / np.mean(prices)
                            spreads.append(spread)
                
                if spreads:
                    spread_score = 1.0 - min(np.mean(spreads), 1.0)
                else:
                    spread_score = 0.0
                
                # Combine scores with weights
                market_liquidity = (
                    0.4 * volume_score +
                    0.4 * depth_score +
                    0.2 * spread_score
                )
            else:
                market_liquidity = 0.0
            
            # Energy efficiency (only relevant for PoW)
            if self.consensus_type == 'pow':
                energy_efficiency = 0.2  # Lower efficiency for PoW
            else:
                energy_efficiency = 0.8  # Higher efficiency for PoS/DPoS
            
            # Calculate objective value
            objective_value = -(
                price_volatility * 0.3 +  # 30% weight on price stability
                (1.0 - network_security_score) * 0.4 +  # 40% weight on network security
                (1.0 - market_liquidity) * 0.2 +  # 20% weight on liquidity
                (1.0 - energy_efficiency) * 0.1  # 10% weight on energy efficiency
            )
            
            results = {
                'price_volatility': price_volatility,
                'network_security_score': network_security_score,
                'market_liquidity': market_liquidity,
                'energy_efficiency': energy_efficiency,
                'final_price': price_history[-1] if len(price_history) > 0 else core_params.initial_price
            }
            
            self.logger.info(f"Simulation completed. Results: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running simulation: {str(e)}")
            # Return default values in case of error
            return {
                'price_volatility': 100.0,  # High volatility (bad)
                'network_security_score': 0.1,  # Low security (bad)
                'market_liquidity': 0.1,  # Low liquidity (bad)
                'energy_efficiency': 0.1,  # Low efficiency (bad)
                'final_price': 0.0
            }

@dataclass
class CoreParameters:
    block_reward: float
    initial_supply: float
    market_depth: float
    initial_price: float
    
    def __str__(self):
        return (f"CoreParameters(block_reward={self.block_reward:.2f}, "
                f"initial_supply={self.initial_supply:.0f}, "
                f"market_depth={self.market_depth:.0f}, "
                f"initial_price={self.initial_price:.2f})")

def create_objective_function(simulation_adapter: TokenomicsAdapter):
    """Create an objective function that evaluates the simulation with given parameters."""
    def objective(params: Dict[str, float]) -> float:
        # Convert parameters to simulation format
        core_params = CoreParameters(
            block_reward=params['block_reward'],
            initial_supply=params['initial_supply'],
            market_depth=params['market_depth'],
            initial_price=params['initial_price']
        )
        
        # Run simulation
        results = simulation_adapter.run_with_parameters(core_params)
        
        # Calculate objective value (weighted combination of metrics)
        price_stability = -abs(results['price_volatility'])  # Negative because we want to minimize volatility
        network_security = results['network_security_score']
        market_liquidity = results['market_liquidity']
        energy_efficiency = results['energy_efficiency']
        
        # Combine metrics with weights
        objective_value = (
            0.3 * price_stability +      # 30% weight on price stability
            0.3 * network_security +     # 30% weight on network security
            0.2 * market_liquidity +     # 20% weight on market liquidity
            0.2 * energy_efficiency      # 20% weight on energy efficiency
        )
        
        return objective_value
    
    return objective

def optimize_core_parameters(
    simulation_adapter: TokenomicsAdapter,
    n_iterations: int = 50,
    n_initial_points: int = 5
) -> Tuple[Dict[str, float], float]:
    """
    Optimize core tokenomics parameters using Bayesian optimization.
    
    Args:
        simulation_adapter: TokenomicsAdapter instance
        n_iterations: Number of optimization iterations
        n_initial_points: Number of random initial points
    
    Returns:
        Tuple of (best parameters, best objective value)
    """
    # Define parameter bounds
    param_bounds = {
        'block_reward': (0.1, 10.0),        # Block reward in tokens
        'initial_supply': (1e6, 1e8),       # Initial token supply (reduced upper bound for speed)
        'market_depth': (1e5, 1e7),         # Market depth in USD
        'initial_price': (0.1, 10.0)        # Initial token price in USD (reduced upper bound for reasonability)
    }
    
    # Initialize optimizer
    optimizer = BayesianOptimizer(
        param_bounds=param_bounds,
        n_initial_points=n_initial_points,
        acquisition_function='ei',  # Expected Improvement
        random_state=42
    )
    
    # Create objective function
    objective = create_objective_function(simulation_adapter)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def callback(iteration: int, params: Dict[str, float], value: float):
        """Callback function to log progress."""
        logger.info(f"Iteration {iteration + 1}:")
        logger.info(f"Parameters: {params}")
        logger.info(f"Objective value: {value:.4f}")
    
    # Run optimization
    best_params, best_value = optimizer.optimize(
        objective_function=objective,
        n_iterations=n_iterations,
        callback=callback
    )
    
    return best_params, best_value

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Optimize core tokenomics parameters')
    parser.add_argument('--consensus', type=str, default='pow', choices=['pow', 'pos', 'dpos'],
                        help='Consensus mechanism to use')
    parser.add_argument('--iterations', type=int, default=30,
                        help='Number of optimization iterations')
    parser.add_argument('--initial-points', type=int, default=5,
                        help='Number of random initial points')
    args = parser.parse_args()
    
    # Initialize simulation adapter
    simulation_adapter = TokenomicsAdapter(consensus_type=args.consensus)
    
    # Run optimization
    best_params, best_value = optimize_core_parameters(
        simulation_adapter=simulation_adapter,
        n_iterations=args.iterations,
        n_initial_points=args.initial_points
    )
    
    print("\nOptimization Results:")
    print(f"Consensus type: {args.consensus}")
    print("Best Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value:.4f}")
    print(f"Best Objective Value: {best_value:.4f}") 