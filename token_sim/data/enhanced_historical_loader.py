from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .historical_loader import HistoricalDataLoader
from ..market.metrics import MarketMetricsCalculator
from ..network.metrics import NetworkMetricsCalculator

class EnhancedHistoricalLoader:
    """Enhanced historical data loader with comprehensive metrics calculation."""
    
    def __init__(self, symbol: str = "BTC", api_key: Optional[str] = None):
        """Initialize the enhanced historical loader.
        
        Args:
            symbol: The cryptocurrency symbol to load data for
            api_key: Optional API key for data provider
        """
        self.base_loader = HistoricalDataLoader(symbol=symbol, api_key=api_key)
        self.market_metrics = MarketMetricsCalculator()
        self.network_metrics = NetworkMetricsCalculator()
        self.symbol = symbol
        
    def get_t0_data(self) -> Dict[str, Any]:
        """Get comprehensive T0 data points for simulation initialization.
        
        Returns:
            Dictionary containing all required T0 data points
        """
        # Fetch base historical data
        historical_data = self.base_loader.fetch_data()
        base_metrics = self.base_loader.get_market_metrics()
        
        # Calculate enhanced market metrics
        market_metrics = self.market_metrics.calculate_metrics(historical_data)
        
        # Calculate network metrics
        network_metrics = self.network_metrics.calculate_metrics(historical_data)
        
        # Combine all metrics into T0 data
        t0_data = {
            # Price data
            'price': {
                'current': base_metrics['initial_price'],
                'ohlcv': historical_data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1].to_dict(),
                'volatility': base_metrics['volatility'],
                'price_range': base_metrics['price_range']
            },
            
            # Volume metrics
            'volume': {
                'current': base_metrics['avg_daily_volume'],
                'profile': market_metrics['volume_profile'],
                'distribution': market_metrics['volume_distribution']
            },
            
            # Market microstructure
            'market_microstructure': {
                'bid_ask_spread': market_metrics['bid_ask_spread'],
                'market_depth': market_metrics['market_depth'],
                'liquidity': market_metrics['liquidity_metrics'],
                'order_book': market_metrics['order_book_metrics']
            },
            
            # Technical indicators
            'technical_indicators': {
                'rsi': market_metrics['rsi'],
                'macd': market_metrics['macd'],
                'bollinger_bands': market_metrics['bollinger_bands'],
                'moving_averages': market_metrics['moving_averages']
            },
            
            # Network metrics
            'network': {
                'consensus': self.base_loader.consensus,
                'hashrate': network_metrics['hashrate'],
                'difficulty': network_metrics['difficulty'],
                'block_time': network_metrics['block_time'],
                'active_addresses': network_metrics['active_addresses'],
                'transaction_count': network_metrics['transaction_count']
            },
            
            # Risk metrics
            'risk_metrics': {
                'var_95': market_metrics['var_95'],
                'expected_shortfall': market_metrics['expected_shortfall'],
                'max_drawdown': market_metrics['max_drawdown'],
                'volatility_clustering': market_metrics['volatility_clustering']
            },
            
            # Consensus-specific metrics
            'consensus_metrics': self._get_consensus_metrics(network_metrics)
        }
        
        return t0_data
    
    def _get_consensus_metrics(self, network_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get consensus-specific metrics based on the cryptocurrency's consensus mechanism.
        
        Args:
            network_metrics: Dictionary of network metrics
            
        Returns:
            Dictionary of consensus-specific metrics
        """
        consensus = self.base_loader.consensus
        
        if consensus == 'PoW':
            return {
                'hashrate': network_metrics['hashrate'],
                'difficulty': network_metrics['difficulty'],
                'block_time': network_metrics['block_time'],
                'mining_power_distribution': network_metrics['mining_power_distribution']
            }
        elif consensus == 'PoS':
            return {
                'staking_ratio': network_metrics['staking_ratio'],
                'validator_count': network_metrics['validator_count'],
                'delegation_rate': network_metrics['delegation_rate'],
                'reward_rate': network_metrics['reward_rate']
            }
        elif consensus == 'DPoS':
            return {
                'delegate_count': network_metrics['delegate_count'],
                'voting_power': network_metrics['voting_power'],
                'reward_distribution': network_metrics['reward_distribution'],
                'consensus_participation': network_metrics['consensus_participation']
            }
        else:
            return {} 