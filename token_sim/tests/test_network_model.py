import pytest
import numpy as np
from token_sim.network.enhanced_network_model import EnhancedNetworkModel, ConsensusType, NetworkParticipant

@pytest.fixture
def pow_params():
    return {
        'num_miners': 100,
        'base_hashrate': 1000.0,
        'block_time': 600,
        'difficulty': 1000000
    }

@pytest.fixture
def pos_params():
    return {
        'num_validators': 100,
        'base_stake': 10000.0,
        'block_time': 600,
        'min_stake': 1000.0
    }

@pytest.fixture
def dpos_params():
    return {
        'num_delegates': 21,
        'num_voters': 1000,
        'base_stake': 1000.0,
        'block_time': 600,
        'min_stake': 100.0
    }

def test_pow_initialization(pow_params):
    model = EnhancedNetworkModel(ConsensusType.POW, pow_params)
    
    # Check basic initialization
    assert model.consensus_type == ConsensusType.POW
    assert len(model.participants) == pow_params['num_miners']
    assert model.total_hashrate > 0
    assert model.block_time == pow_params['block_time']
    
    # Check miner distribution
    miners = [p for p in model.participants.values() if p.type == 'miner']
    assert len(miners) == pow_params['num_miners']
    assert all(m.hashrate > 0 for m in miners)
    assert all(m.stake == 0 for m in miners)  # PoW miners don't stake

def test_pos_initialization(pos_params):
    model = EnhancedNetworkModel(ConsensusType.POS, pos_params)
    
    # Check basic initialization
    assert model.consensus_type == ConsensusType.POS
    assert len(model.participants) == pos_params['num_validators']
    assert model.total_stake > 0
    assert model.block_time == pos_params['block_time']
    
    # Check validator distribution
    validators = [p for p in model.participants.values() if p.type == 'validator']
    assert len(validators) == pos_params['num_validators']
    assert all(v.stake >= pos_params['min_stake'] for v in validators)
    assert all(v.voting_power == v.stake for v in validators)

def test_dpos_initialization(dpos_params):
    model = EnhancedNetworkModel(ConsensusType.DPOS, dpos_params)
    
    # Check basic initialization
    assert model.consensus_type == ConsensusType.DPOS
    assert len(model.participants) == dpos_params['num_delegates'] + dpos_params['num_voters']
    assert model.total_stake > 0
    assert model.block_time == dpos_params['block_time']
    
    # Check delegate and voter distribution
    delegates = [p for p in model.participants.values() if p.type == 'delegate']
    voters = [p for p in model.participants.values() if p.type == 'voter']
    
    assert len(delegates) == dpos_params['num_delegates']
    assert len(voters) == dpos_params['num_voters']
    assert all(d.stake >= dpos_params['min_stake'] * 10 for d in delegates)
    assert all(v.stake >= dpos_params['min_stake'] for v in voters)

def test_pow_updates(pow_params):
    model = EnhancedNetworkModel(ConsensusType.POW, pow_params)
    initial_hashrate = model.total_hashrate
    
    # Simulate network updates
    for time_step in range(1, 11):
        model.update(time_step, {'price': 50000.0})
        
        # Check that hashrate history is updated
        assert len(model.hashrate_history) == time_step + 1
        assert model.hashrate_history[-1] == model.total_hashrate
        
        # Check that block time is reasonable
        assert 300 <= model.block_time <= 900  # 5-15 minutes
    
    # Check that difficulty has been adjusted
    assert model.difficulty != pow_params['difficulty']

def test_pos_updates(pos_params):
    model = EnhancedNetworkModel(ConsensusType.POS, pos_params)
    initial_stake = model.total_stake
    
    # Simulate network updates
    for time_step in range(1, 11):
        model.update(time_step, {'price': 50000.0})
        
        # Check that stake history is updated
        assert len(model.stake_history) == time_step + 1
        assert model.stake_history[-1] == model.total_stake
        
        # Check that block time is reasonable
        assert 300 <= model.block_time <= 900  # 5-15 minutes
    
    # Check that rewards have been distributed
    validators = [p for p in model.participants.values() if p.type == 'validator']
    assert all(v.rewards > 0 for v in validators)

def test_dpos_updates(dpos_params):
    model = EnhancedNetworkModel(ConsensusType.DPOS, dpos_params)
    initial_stake = model.total_stake
    
    # Simulate network updates
    for time_step in range(1, 11):
        model.update(time_step, {'price': 50000.0})
        
        # Check that stake history is updated
        assert len(model.stake_history) == time_step + 1
        assert model.stake_history[-1] == model.total_stake
        
        # Check that block time is reasonable
        assert 20 <= model.block_time <= 40  # DPoS should be faster
    
    # Check that rewards have been distributed
    delegates = [p for p in model.participants.values() if p.type == 'delegate']
    assert all(d.rewards > 0 for d in delegates)

def test_network_metrics(pow_params):
    model = EnhancedNetworkModel(ConsensusType.POW, pow_params)
    
    # Get initial metrics
    metrics = model.get_network_metrics()
    
    # Check basic metrics
    assert metrics['consensus_type'] == ConsensusType.POW.value
    assert metrics['block_height'] == 0
    assert metrics['block_time'] == pow_params['block_time']
    assert metrics['total_hashrate'] == model.total_hashrate
    assert metrics['participant_count'] == len(model.participants)
    
    # Check PoW-specific metrics
    assert 'difficulty' in metrics
    assert 'miner_distribution' in metrics
    assert 'gini_coefficient' in metrics['miner_distribution']
    assert 'top_10_percent' in metrics['miner_distribution']
    
    # Check history metrics
    assert len(metrics['hashrate_history']) == 1
    assert len(metrics['block_time_history']) == 1
    assert len(metrics['participant_count_history']) == 1

def test_gini_calculation():
    model = EnhancedNetworkModel(ConsensusType.POW, {'num_miners': 100})
    
    # Test with uniform distribution
    uniform = [1.0] * 100
    assert abs(model._calculate_gini(uniform)) < 0.01
    
    # Test with perfect inequality
    perfect_inequality = [0.0] * 99 + [1.0]
    assert abs(model._calculate_gini(perfect_inequality) - 1.0) < 0.01
    
    # Test with normal distribution
    normal = np.random.normal(1000, 100, 100)
    gini = model._calculate_gini(normal)
    assert 0 <= gini <= 1 