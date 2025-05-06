from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

class MetricType(Enum):
    SECURITY = "security"
    STABILITY = "stability"
    PERFORMANCE = "performance"
    ECONOMIC = "economic"

@dataclass
class MetricResult:
    """Container for metric evaluation results."""
    value: float
    weight: float
    type: MetricType
    description: str

class TokenomicsMetrics:
    """Evaluation metrics for tokenomics optimization."""
    
    def __init__(self):
        self.metrics = {
            # Security Metrics
            'network_security': {
                'type': MetricType.SECURITY,
                'weight': 0.3,
                'description': 'Network security score based on consensus and participation'
            },
            'attack_resistance': {
                'type': MetricType.SECURITY,
                'weight': 0.2,
                'description': 'Resistance to various attack vectors'
            },
            
            # Stability Metrics
            'price_stability': {
                'type': MetricType.STABILITY,
                'weight': 0.25,
                'description': 'Price stability over time'
            },
            'volatility': {
                'type': MetricType.STABILITY,
                'weight': 0.15,
                'description': 'Price volatility measure'
            },
            
            # Performance Metrics
            'transaction_throughput': {
                'type': MetricType.PERFORMANCE,
                'weight': 0.2,
                'description': 'Transactions per second'
            },
            'block_time_consistency': {
                'type': MetricType.PERFORMANCE,
                'weight': 0.15,
                'description': 'Consistency of block times'
            },
            
            # Economic Metrics
            'token_distribution': {
                'type': MetricType.ECONOMIC,
                'weight': 0.2,
                'description': 'Gini coefficient of token distribution'
            },
            'market_liquidity': {
                'type': MetricType.ECONOMIC,
                'weight': 0.2,
                'description': 'Market liquidity score'
            },
            'staking_participation': {
                'type': MetricType.ECONOMIC,
                'weight': 0.15,
                'description': 'Percentage of tokens staked'
            }
        }
    
    def calculate_network_security(self, simulation_results: Dict[str, Any]) -> float:
        """Calculate network security score."""
        # Example implementation - replace with actual logic
        consensus_type = simulation_results.get('consensus_type', 'pow')
        num_participants = simulation_results.get('num_participants', 0)
        staking_ratio = simulation_results.get('staking_ratio', 0)
        
        base_score = 0.0
        if consensus_type == 'pow':
            base_score = 0.7
        elif consensus_type == 'pos':
            base_score = 0.8
        elif consensus_type == 'dpos':
            base_score = 0.9
        
        participation_score = min(num_participants / 1000, 1.0)
        staking_score = staking_ratio if consensus_type in ['pos', 'dpos'] else 1.0
        
        return base_score * 0.4 + participation_score * 0.3 + staking_score * 0.3
    
    def calculate_price_stability(self, simulation_results: Dict[str, Any]) -> float:
        """Calculate price stability score."""
        # Example implementation - replace with actual logic
        price_history = simulation_results.get('price_history', [])
        if not price_history:
            return 0.0
        
        returns = np.diff(np.log(price_history))
        volatility = np.std(returns)
        stability = 1.0 / (1.0 + volatility)
        
        return min(max(stability, 0.0), 1.0)
    
    def calculate_token_distribution(self, simulation_results: Dict[str, Any]) -> float:
        """Calculate token distribution score using Gini coefficient."""
        # Example implementation - replace with actual logic
        balances = simulation_results.get('token_balances', [])
        if not balances:
            return 0.0
        
        # Calculate Gini coefficient
        sorted_balances = np.sort(balances)
        n = len(sorted_balances)
        index = np.arange(1, n + 1)
        gini = ((2 * index - n - 1) * sorted_balances).sum() / (n * sorted_balances.sum())
        
        # Convert to a score where 1 is perfect equality
        return 1.0 - gini
    
    def calculate_market_liquidity(self, simulation_results: Dict[str, Any]) -> float:
        """Calculate market liquidity score."""
        # Example implementation - replace with actual logic
        market_depth = simulation_results.get('market_depth', 0)
        trading_volume = simulation_results.get('trading_volume', 0)
        
        if market_depth == 0 or trading_volume == 0:
            return 0.0
        
        # Simple liquidity score based on market depth and trading volume
        depth_score = min(market_depth / 1000000, 1.0)
        volume_score = min(trading_volume / 100000, 1.0)
        
        return 0.6 * depth_score + 0.4 * volume_score
    
    def evaluate(self, simulation_results: Dict[str, Any]) -> Dict[str, MetricResult]:
        """Evaluate all metrics for the simulation results."""
        results = {}
        
        # Calculate each metric
        results['network_security'] = MetricResult(
            value=self.calculate_network_security(simulation_results),
            weight=self.metrics['network_security']['weight'],
            type=self.metrics['network_security']['type'],
            description=self.metrics['network_security']['description']
        )
        
        results['price_stability'] = MetricResult(
            value=self.calculate_price_stability(simulation_results),
            weight=self.metrics['price_stability']['weight'],
            type=self.metrics['price_stability']['type'],
            description=self.metrics['price_stability']['description']
        )
        
        results['token_distribution'] = MetricResult(
            value=self.calculate_token_distribution(simulation_results),
            weight=self.metrics['token_distribution']['weight'],
            type=self.metrics['token_distribution']['type'],
            description=self.metrics['token_distribution']['description']
        )
        
        results['market_liquidity'] = MetricResult(
            value=self.calculate_market_liquidity(simulation_results),
            weight=self.metrics['market_liquidity']['weight'],
            type=self.metrics['market_liquidity']['type'],
            description=self.metrics['market_liquidity']['description']
        )
        
        return results
    
    def calculate_overall_score(self, metric_results: Dict[str, MetricResult]) -> float:
        """Calculate overall score from individual metric results."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric in metric_results.values():
            weighted_sum += metric.value * metric.weight
            total_weight += metric.weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight 