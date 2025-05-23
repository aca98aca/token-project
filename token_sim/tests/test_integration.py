import pytest
import numpy as np
from token_sim.network.enhanced_network_model import EnhancedNetworkModel, ConsensusType
from token_sim.market.enhanced_market_model import EnhancedMarketModel

@pytest.fixture
def simulation_params():
    return {
        # Market parameters
        'initial_price': 50000.0,
        'market_depth': 1000000.0,
        'volatility': 0.02,
        
        # Network parameters
        'num_miners': 100,
        'base_hashrate': 1000.0,
        'block_time': 600,
        'difficulty': 1000000,
        'min_stake': 1000.0
    }

def test_market_network_interaction(simulation_params):
    """Test the interaction between market and network models."""
    # Initialize models
    market_model = EnhancedMarketModel(
        initial_price=simulation_params['initial_price'],
        market_depth=simulation_params['market_depth'],
        volatility=simulation_params['volatility']
    )
    
    network_model = EnhancedNetworkModel(
        consensus_type=ConsensusType.POW,
        initial_params=simulation_params
    )
    
    # Run simulation for multiple time steps
    for time_step in range(1, 11):
        # Get current market conditions
        market_metrics = market_model.get_market_metrics()
        market_conditions = {
            'price': market_metrics['price'],
            'volatility': simulation_params['volatility'],
            'volume': market_metrics['volume'],
            'spread': market_metrics['spread']
        }
        
        # Update network with market conditions
        network_model.update(time_step, market_conditions)
        
        # Get network metrics
        network_metrics = network_model.get_network_metrics()
        
        # Update market with network conditions
        market_model.update(time_step, {
            'hashrate': network_metrics['total_hashrate'],
            'block_time': network_metrics['block_time'],
            'participant_count': network_metrics['participant_count']
        })
        
        # Verify market metrics
        assert market_model.order_book.get_mid_price() > 0
        assert market_model.order_book.volume >= 0
        assert market_model.order_book.get_spread() > 0
        
        # Verify network metrics
        assert network_model.total_hashrate > 0
        assert network_model.block_time > 0
        assert len(network_model.participants) > 0
        
        # Check that rewards are being distributed
        miners = [p for p in network_model.participants.values() if p.type == 'miner']
        assert all(m.rewards >= 0 for m in miners)

def test_price_impact_on_network(simulation_params):
    """Test how price changes impact network behavior."""
    # Initialize models
    market_model = EnhancedMarketModel(
        initial_price=simulation_params['initial_price'],
        market_depth=simulation_params['market_depth'],
        volatility=simulation_params['volatility']
    )
    
    network_model = EnhancedNetworkModel(
        consensus_type=ConsensusType.POW,
        initial_params=simulation_params
    )
    
    # Record initial state
    initial_hashrate = network_model.total_hashrate
    initial_participants = len(network_model.participants)
    
    # Simulate price increase
    for time_step in range(1, 6):
        market_conditions = {
            'price': simulation_params['initial_price'] * (1 + 0.1 * time_step),  # 10% increase each step
            'volatility': simulation_params['volatility'],
            'volume': 1000000.0,
            'spread': 100.0
        }
        network_model.update(time_step, market_conditions)
    
    # Verify network response to price increase
    assert network_model.total_hashrate > initial_hashrate  # More mining power
    assert len(network_model.participants) >= initial_participants  # More participants

def test_network_impact_on_market(simulation_params):
    """Test how network changes impact market behavior."""
    # Initialize models
    market_model = EnhancedMarketModel(
        initial_price=simulation_params['initial_price'],
        market_depth=simulation_params['market_depth'],
        volatility=simulation_params['volatility']
    )
    
    network_model = EnhancedNetworkModel(
        consensus_type=ConsensusType.POW,
        initial_params=simulation_params
    )
    
    # Record initial state
    initial_price = market_model.order_book.get_mid_price()
    initial_volume = market_model.order_book.volume
    
    # Simulate network growth
    for time_step in range(1, 6):
        network_conditions = {
            'hashrate': simulation_params['base_hashrate'] * (1 + 0.2 * time_step),  # 20% increase each step
            'block_time': simulation_params['block_time'] * (1 - 0.05 * time_step),  # 5% decrease each step
            'participant_count': simulation_params['num_miners'] + time_step * 10  # 10 new participants each step
        }
        market_model.update(time_step, network_conditions)
    
    # Verify market response to network growth
    assert market_model.order_book.get_mid_price() != initial_price  # Price should change
    assert market_model.order_book.volume > initial_volume  # Volume should increase

def test_consensus_impact(simulation_params):
    """Test how different consensus mechanisms impact market behavior."""
    consensus_types = [ConsensusType.POW, ConsensusType.POS, ConsensusType.DPOS]
    
    for consensus_type in consensus_types:
        # Initialize models
        market_model = EnhancedMarketModel(
            initial_price=simulation_params['initial_price'],
            market_depth=simulation_params['market_depth'],
            volatility=simulation_params['volatility']
        )
        
        network_model = EnhancedNetworkModel(
            consensus_type=consensus_type,
            initial_params=simulation_params
        )
        
        # Run simulation
        for time_step in range(1, 6):
            # Update both models
            market_metrics = market_model.get_market_metrics()
            network_model.update(time_step, {
                'price': market_metrics['price'],
                'volatility': simulation_params['volatility'],
                'volume': market_metrics['volume']
            })
            
            network_metrics = network_model.get_network_metrics()
            market_model.update(time_step, {
                'hashrate': network_metrics['total_hashrate'],
                'block_time': network_metrics['block_time'],
                'participant_count': network_metrics['participant_count']
            })
        
        # Verify consensus-specific behavior
        if consensus_type == ConsensusType.POW:
            assert network_model.total_hashrate > 0
            assert network_model.difficulty > 0
        elif consensus_type == ConsensusType.POS:
            assert network_model.total_stake > 0
            assert network_model.total_voting_power > 0
        else:  # DPoS
            assert network_model.num_delegates > 0
            assert network_model.total_voting_power > 0
        
        # Verify market metrics are reasonable
        assert market_model.order_book.get_mid_price() > 0
        assert market_model.order_book.volume >= 0
        assert market_model.order_book.get_spread() > 0 