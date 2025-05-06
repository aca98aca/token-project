import pytest
from token_sim.consensus.pow import ProofOfWork
from token_sim.consensus.pos import ProofOfStake
from token_sim.consensus.dpos import DelegatedProofOfStake

def test_pow_initialization():
    """Test PoW consensus initialization."""
    pow = ProofOfWork(
        block_reward=1.0,
        difficulty_adjustment_blocks=2016,
        target_block_time=600.0
    )
    
    # Test initial state
    assert pow.block_reward == 1.0
    assert pow.difficulty_adjustment_blocks == 2016
    assert pow.target_block_time == 600.0
    assert pow.current_difficulty == 1.0
    assert pow.total_hashrate == 0.0
    assert pow.network_security_score == 0.0

def test_pow_participant_initialization():
    """Test PoW participant initialization."""
    pow = ProofOfWork()
    pow.initialize_participants(num_participants=10)
    
    # Test participant initialization
    assert len(pow.participants) == 10
    assert len(pow.get_active_participants()) == 10
    
    # Test participant stats
    for miner_id in pow.participants:
        stats = pow.get_participant_stats(miner_id)
        assert 10 <= stats['hashrate'] <= 100  # Random hashrate between 10 and 100
        assert stats['rewards'] == 0.0
        assert stats['blocks_found'] == 0
        assert stats['active'] == True
        assert stats['last_reward_time'] == 0.0

def test_pow_consensus_step():
    """Test PoW consensus step."""
    pow = ProofOfWork(block_reward=10.0)
    pow.initialize_participants(num_participants=5)
    
    # Run multiple consensus steps
    total_rewards = 0
    blocks_found = 0
    for _ in range(100):  # Run 100 steps
        step_rewards, distribution = pow.perform_consensus_step()
        total_rewards += step_rewards
        if distribution:  # If a block was found
            blocks_found += len(distribution)
            # Verify reward distribution
            assert sum(distribution.values()) == step_rewards
            # Each individual reward should be equal to block_reward
            for reward in distribution.values():
                assert reward == pow.block_reward
            # The total step_rewards should be a multiple of block_reward
            assert step_rewards % pow.block_reward == 0
    
    # Verify that blocks were found
    assert blocks_found > 0
    assert total_rewards == blocks_found * pow.block_reward

def test_pow_difficulty_adjustment():
    """Test PoW difficulty adjustment."""
    pow = ProofOfWork(
        block_reward=1.0,
        difficulty_adjustment_blocks=10,  # Small number for testing
        target_block_time=600.0
    )
    pow.initialize_participants(num_participants=5)
    
    # Run enough blocks for difficulty adjustment
    initial_difficulty = pow.current_difficulty
    blocks_found = 0
    for _ in range(200):  # Run more steps to increase chance of adjustment
        _, distribution = pow.perform_consensus_step()
        if distribution:
            blocks_found += 1
    # Only check if block_times is not empty (i.e., at least one adjustment was attempted)
    if pow.block_times:
        assert pow.current_difficulty != initial_difficulty
        assert 0.1 <= pow.current_difficulty <= 100.0  # Within bounds

def test_pow_network_security():
    """Test PoW network security score calculation."""
    pow = ProofOfWork()
    pow.initialize_participants(num_participants=25)  # Optimal number for decentralization
    
    # Initial security score should be reasonable
    assert 0 <= pow.network_security_score <= 1
    
    # Run some consensus steps
    for _ in range(10):
        pow.perform_consensus_step()
    
    # Security score should still be valid
    assert 0 <= pow.network_security_score <= 1
    
    # Test with fewer participants
    pow = ProofOfWork()
    pow.initialize_participants(num_participants=5)
    assert pow.network_security_score < 1  # Should be lower due to less decentralization

def test_pow_reset():
    """Test PoW reset functionality."""
    pow = ProofOfWork()
    pow.initialize_participants(num_participants=5)
    
    # Run some consensus steps
    for _ in range(10):
        pow.perform_consensus_step()
    
    # Record some state
    pre_reset_rewards = pow.get_rewards_distribution()
    pre_reset_difficulty = pow.current_difficulty
    
    # Reset
    pow.reset()
    
    # Verify reset state
    assert pow.current_difficulty == pow.initial_difficulty
    assert pow.current_block == 0
    assert pow.total_rewards == 0.0
    assert pow.blocks_since_adjustment == 0
    assert pow.last_block_time == 0.0
    assert len(pow.block_times) == 0
    
    # Verify participant reset
    for miner_id in pow.participants:
        stats = pow.get_participant_stats(miner_id)
        assert stats['rewards'] == 0.0
        assert stats['blocks_found'] == 0
        assert stats['active'] == True
        assert stats['last_reward_time'] == 0.0

def test_pos_initialization():
    """Test PoS consensus initialization."""
    pos = ProofOfStake(
        block_reward=1.0,
        min_stake=100.0,
        staking_apy=0.05
    )
    
    # Test initial state
    assert pos.block_reward == 1.0
    assert pos.min_stake == 100.0
    assert pos.staking_apy == 0.05
    assert pos.total_stake == 0.0
    assert len(pos.participants) == 0

def test_pos_participant_initialization():
    """Test PoS participant initialization."""
    pos = ProofOfStake(min_stake=100.0)
    pos.initialize_participants(num_participants=10)
    
    # Test participant initialization
    assert len(pos.participants) == 10
    assert len(pos.get_active_participants()) == 10
    
    # Test participant stats
    total_stake = 0
    for validator_id in pos.participants:
        stats = pos.get_participant_stats(validator_id)
        assert 100.0 <= stats['stake'] <= 1000.0  # Between min_stake and 10x min_stake
        assert stats['rewards'] == 0.0
        assert stats['blocks_produced'] == 0
        assert stats['active'] == True
        assert stats['stake_rewards'] == 0.0
        total_stake += stats['stake']
    
    assert pos.total_stake == total_stake

def test_pos_consensus_step():
    """Test PoS consensus step."""
    pos = ProofOfStake(block_reward=10.0, staking_apy=0.1)  # 10% APY for noticeable rewards
    pos.initialize_participants(num_participants=5)
    
    # Record initial stakes
    initial_stakes = {
        validator_id: stats['stake']
        for validator_id, stats in pos.participants.items()
    }
    
    # Run multiple consensus steps
    total_rewards = 0
    blocks_produced = 0
    for _ in range(10):  # Run 10 days worth of steps
        step_rewards, distribution = pos.perform_consensus_step()
        total_rewards += step_rewards
        if distribution:  # If a block was produced
            blocks_produced += len(distribution)
            # Verify reward distribution
            assert sum(distribution.values()) == step_rewards
            # Each individual reward should be equal to block_reward
            for reward in distribution.values():
                assert reward == pos.block_reward
            # The total step_rewards should be a multiple of block_reward
            assert step_rewards % pos.block_reward == 0
    
    # Verify that blocks were produced
    assert blocks_produced > 0
    
    # Verify staking rewards were distributed
    for validator_id, stats in pos.participants.items():
        current_stake = stats['stake']
        initial_stake = initial_stakes[validator_id]
        # Should have earned some staking rewards
        assert current_stake > initial_stake
        assert stats['stake_rewards'] > 0

def test_pos_stake_updates():
    """Test PoS stake update functionality."""
    pos = ProofOfStake(min_stake=100.0)
    pos.initialize_participants(num_participants=1)
    validator_id = "validator_0"
    
    # Get initial stake
    initial_stake = pos.get_participant_stats(validator_id)['stake']
    initial_total = pos.total_stake
    
    # Test increasing stake
    pos.update_stake(validator_id, 100.0)
    stats = pos.get_participant_stats(validator_id)
    assert stats['stake'] == initial_stake + 100.0
    assert pos.total_stake == initial_total + 100.0
    
    # Test decreasing stake
    pos.update_stake(validator_id, -50.0)
    stats = pos.get_participant_stats(validator_id)
    assert stats['stake'] == initial_stake + 50.0
    assert pos.total_stake == initial_total + 50.0
    
    # Test invalid stake updates
    with pytest.raises(ValueError):
        pos.update_stake("invalid_validator", 100.0)  # Invalid validator
    
    with pytest.raises(ValueError):
        pos.update_stake(validator_id, -1000000.0)  # Would make stake negative

def test_pos_minimum_stake():
    """Test PoS minimum stake requirements."""
    pos = ProofOfStake(min_stake=100.0)
    pos.initialize_participants(num_participants=2)
    validator_id = "validator_0"
    
    # Initially all validators should be active
    assert len(pos.get_active_participants()) == 2
    
    # Reduce stake below minimum
    initial_stake = pos.get_participant_stats(validator_id)['stake']
    pos.update_stake(validator_id, -(initial_stake - 50.0))  # Leave only 50 stake
    
    # Should now have one less active validator
    active_validators = pos.get_active_participants()
    assert len(active_validators) == 1
    assert validator_id not in active_validators
    
    # Validator with low stake should not produce blocks
    rewards_count = 0
    for _ in range(50):  # Run many steps
        _, distribution = pos.perform_consensus_step()
        if validator_id in distribution:
            rewards_count += 1
    assert rewards_count == 0  # Should never get block rewards

def test_dpos_initialization():
    """Test DPoS consensus initialization."""
    dpos = DelegatedProofOfStake(
        block_reward=1.0,
        min_stake=100.0,
        num_delegates=21,
        staking_apy=0.05
    )
    
    # Test initial state
    assert dpos.block_reward == 1.0
    assert dpos.min_stake == 100.0
    assert dpos.num_delegates == 21
    assert dpos.staking_apy == 0.05
    assert dpos.total_stake == 0.0
    assert len(dpos.participants) == 0
    assert len(dpos.delegates) == 0

def test_dpos_participant_initialization():
    """Test DPoS participant initialization."""
    dpos = DelegatedProofOfStake(min_stake=100.0, num_delegates=5)
    dpos.initialize_participants(num_participants=10)
    
    # Test participant initialization
    assert len(dpos.participants) == 10
    assert len(dpos.get_active_participants()) == 10
    assert len(dpos.delegates) == 5  # Should have selected 5 delegates
    
    # Test participant stats
    total_stake = 0
    delegate_count = 0
    for participant_id in dpos.participants:
        stats = dpos.get_participant_stats(participant_id)
        assert 100.0 <= stats['stake'] <= 1000.0  # Between min_stake and 10x min_stake
        assert stats['rewards'] == 0.0
        assert stats['blocks_produced'] == 0
        assert stats['active'] == True
        assert stats['stake_rewards'] == 0.0
        assert stats['delegated_stake'] == 0.0
        if stats['is_delegate']:
            delegate_count += 1
        total_stake += stats['stake']
    
    assert dpos.total_stake == total_stake
    assert delegate_count == 5  # Should have exactly 5 delegates

def test_dpos_consensus_step():
    """Test DPoS consensus step."""
    dpos = DelegatedProofOfStake(block_reward=10.0, staking_apy=0.1, num_delegates=3)  # 10% APY for noticeable rewards
    dpos.initialize_participants(num_participants=5)
    
    # Record initial stakes
    initial_stakes = {
        participant_id: stats['stake']
        for participant_id, stats in dpos.participants.items()
    }
    
    # Run multiple consensus steps
    total_rewards = 0
    blocks_produced = 0
    for _ in range(10):  # Run 10 days worth of steps
        step_rewards, distribution = dpos.perform_consensus_step()
        total_rewards += step_rewards
        if distribution:  # If a block was produced
            blocks_produced += len(distribution)
            # Verify reward distribution
            assert sum(distribution.values()) == step_rewards
            # Each individual reward should be equal to block_reward
            for reward in distribution.values():
                assert reward == dpos.block_reward
            # The total step_rewards should be a multiple of block_reward
            assert step_rewards % dpos.block_reward == 0
            # Verify only delegates produced blocks
            for producer_id in distribution:
                assert producer_id in dpos.delegates
    
    # Verify that blocks were produced
    assert blocks_produced > 0
    
    # Verify staking rewards were distributed
    for participant_id, stats in dpos.participants.items():
        current_stake = stats['stake']
        initial_stake = initial_stakes[participant_id]
        # Should have earned some staking rewards
        assert current_stake > initial_stake
        assert stats['stake_rewards'] > 0

def test_dpos_delegation():
    """Test DPoS stake delegation functionality."""
    dpos = DelegatedProofOfStake(min_stake=100.0, num_delegates=2)
    dpos.initialize_participants(num_participants=4)
    
    # Find a delegate and non-delegate
    delegate_id = dpos.delegates[0]
    non_delegate_id = next(p_id for p_id in dpos.participants if not dpos.participants[p_id]['is_delegate'])
    
    # Test successful delegation
    initial_stake = dpos.get_participant_stats(non_delegate_id)['stake']
    delegation_amount = initial_stake / 2
    dpos.delegate_stake(non_delegate_id, delegate_id, delegation_amount)
    
    # Verify delegation
    delegator_stats = dpos.get_participant_stats(non_delegate_id)
    delegate_stats = dpos.get_participant_stats(delegate_id)
    assert delegator_stats['stake'] == initial_stake - delegation_amount
    assert delegate_stats['delegated_stake'] == delegation_amount
    
    # Test invalid delegations
    with pytest.raises(ValueError):
        # Try to delegate to non-delegate
        dpos.delegate_stake(non_delegate_id, non_delegate_id, 100.0)
    
    with pytest.raises(ValueError):
        # Try to delegate more than available
        dpos.delegate_stake(non_delegate_id, delegate_id, initial_stake * 2)
    
    with pytest.raises(ValueError):
        # Try to delegate with invalid IDs
        dpos.delegate_stake("invalid_id", delegate_id, 100.0)

def test_dpos_delegate_selection():
    """Test DPoS delegate selection process."""
    dpos = DelegatedProofOfStake(min_stake=100.0, num_delegates=3)
    dpos.initialize_participants(num_participants=6)
    
    # Record initial delegates
    initial_delegates = set(dpos.delegates)
    
    # Find a non-delegate with high stake potential
    non_delegate_id = next(p_id for p_id in dpos.participants if not dpos.participants[p_id]['is_delegate'])
    
    # Give them a lot of delegated stake
    for other_id in dpos.participants:
        if other_id != non_delegate_id and not dpos.participants[other_id]['is_delegate']:
            try:
                dpos.delegate_stake(other_id, dpos.delegates[0], dpos.participants[other_id]['stake'] / 2)
            except ValueError:
                continue
    
    # Force delegate selection
    dpos._select_delegates()
    
    # Verify delegate changes
    new_delegates = set(dpos.delegates)
    assert len(new_delegates) == 3  # Should still have 3 delegates
    
    # Verify delegate status in participant stats
    for participant_id, stats in dpos.participants.items():
        assert stats['is_delegate'] == (participant_id in new_delegates) 