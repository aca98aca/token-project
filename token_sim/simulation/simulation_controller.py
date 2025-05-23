from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta

from token_sim.network.enhanced_network_model import EnhancedNetworkModel, ConsensusType
from token_sim.market.enhanced_market_model import EnhancedMarketModel

@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    # Time parameters
    start_time: datetime
    end_time: datetime
    time_step: timedelta
    
    # Market parameters
    initial_price: float
    market_depth: float
    volatility: float
    
    # Network parameters
    consensus_type: ConsensusType
    num_miners: int
    base_hashrate: float
    block_time: int
    difficulty: float
    min_stake: float
    
    # Simulation parameters
    random_seed: Optional[int] = None

class SimulationController:
    """Controller class for coordinating market and network simulations."""
    
    def __init__(self, config: SimulationConfig):
        """Initialize the simulation controller.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        
        # Set random seed if provided
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # Initialize models
        self.market_model = EnhancedMarketModel(
            initial_price=config.initial_price,
            market_depth=config.market_depth,
            volatility=config.volatility
        )
        
        self.network_model = EnhancedNetworkModel(
            consensus_type=config.consensus_type,
            initial_params={
                'num_miners': config.num_miners,
                'base_hashrate': config.base_hashrate,
                'block_time': config.block_time,
                'difficulty': config.difficulty,
                'min_stake': config.min_stake
            }
        )
        
        # Initialize results storage
        self.results = {
            'market': [],
            'network': [],
            'interactions': []
        }
    
    def run_simulation(self) -> Dict[str, pd.DataFrame]:
        """Run the simulation and collect results.
        
        Returns:
            Dictionary containing DataFrames with market, network, and interaction results
        """
        current_time = self.config.start_time
        time_step = 0
        
        while current_time <= self.config.end_time:
            # Get current market conditions
            market_metrics = self.market_model.get_market_metrics()
            market_conditions = {
                'price': market_metrics['price'],
                'volatility': self.config.volatility,
                'volume': market_metrics['volume'],
                'spread': market_metrics['spread']
            }
            
            # Update network with market conditions
            self.network_model.update(time_step, market_conditions)
            
            # Get network metrics
            network_metrics = self.network_model.get_network_metrics()
            
            # Update market with network conditions
            self.market_model.update(time_step, {
                'hashrate': network_metrics['total_hashrate'],
                'block_time': network_metrics['block_time'],
                'participant_count': network_metrics['participant_count']
            })
            
            # Record results
            self._record_results(current_time, market_metrics, network_metrics)
            
            # Update time
            current_time += self.config.time_step
            time_step += 1
        
        return self._process_results()
    
    def _record_results(self, timestamp: datetime, market_metrics: Dict[str, Any], network_metrics: Dict[str, Any]):
        """Record simulation results for the current time step."""
        # Record market metrics
        market_record = {
            'timestamp': timestamp,
            'price': market_metrics['price'],
            'volume': market_metrics['volume'],
            'spread': market_metrics['spread'],
            'depth': market_metrics['depth']
        }
        self.results['market'].append(market_record)
        
        # Record network metrics
        network_record = {
            'timestamp': timestamp,
            'hashrate': network_metrics['total_hashrate'],
            'block_time': network_metrics['block_time'],
            'participant_count': network_metrics['participant_count'],
            'total_stake': network_metrics['total_stake'],
            'total_voting_power': network_metrics['total_voting_power']
        }
        
        # Add consensus-specific metrics
        if self.config.consensus_type == ConsensusType.POW:
            network_record.update({
                'difficulty': network_metrics['difficulty'],
                'miner_gini': network_metrics['miner_distribution']['gini_coefficient']
            })
        elif self.config.consensus_type == ConsensusType.POS:
            network_record.update({
                'staking_ratio': network_metrics['staking_ratio'],
                'validator_gini': network_metrics['validator_distribution']['gini_coefficient']
            })
        else:  # DPoS
            network_record.update({
                'delegate_gini': network_metrics['delegate_distribution']['gini_coefficient'],
                'voting_power_gini': network_metrics['voting_power_distribution']['gini_coefficient']
            })
        
        self.results['network'].append(network_record)
        
        # Record interactions
        interaction_record = {
            'timestamp': timestamp,
            'price_impact': market_metrics['price'] / self.config.initial_price - 1,
            'volume_impact': market_metrics['volume'] / self.config.market_depth,
            'network_growth': network_metrics['participant_count'] / self.config.num_miners - 1
        }
        self.results['interactions'].append(interaction_record)
    
    def _process_results(self) -> Dict[str, pd.DataFrame]:
        """Process recorded results into DataFrames."""
        return {
            'market': pd.DataFrame(self.results['market']),
            'network': pd.DataFrame(self.results['network']),
            'interactions': pd.DataFrame(self.results['interactions'])
        }
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for the simulation results."""
        market_df = pd.DataFrame(self.results['market'])
        network_df = pd.DataFrame(self.results['network'])
        interactions_df = pd.DataFrame(self.results['interactions'])
        
        return {
            'market': {
                'avg_price': market_df['price'].mean(),
                'price_volatility': market_df['price'].std(),
                'avg_volume': market_df['volume'].mean(),
                'avg_spread': market_df['spread'].mean()
            },
            'network': {
                'avg_hashrate': network_df['hashrate'].mean(),
                'avg_block_time': network_df['block_time'].mean(),
                'final_participants': network_df['participant_count'].iloc[-1],
                'total_stake': network_df['total_stake'].iloc[-1]
            },
            'interactions': {
                'max_price_impact': interactions_df['price_impact'].max(),
                'max_volume_impact': interactions_df['volume_impact'].max(),
                'max_network_growth': interactions_df['network_growth'].max()
            }
        }
    
    def save_results(self, output_dir: str):
        """Save simulation results to CSV files.
        
        Args:
            output_dir: Directory to save results
        """
        results = self._process_results()
        
        # Save each DataFrame
        for name, df in results.items():
            filename = f"{output_dir}/{name}_results.csv"
            df.to_csv(filename, index=False)
        
        # Save summary statistics
        summary = self.get_summary_statistics()
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(f"{output_dir}/summary_statistics.csv") 