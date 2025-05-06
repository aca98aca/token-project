import pytest
from token_sim.agents.miner import Miner
from token_sim.agents.holder import Holder
from token_sim.agents.trader import Trader
from token_sim.agents.staker import Staker

def test_miner_initialization():
    """Test miner agent initialization."""
    miner = Miner(
        agent_id="test_miner",
        strategy="passive",
        initial_hashrate=50.0,
        electricity_cost=0.05,
        initial_balance=1000.0
    )
    
    # Test initial state
    state = miner.get_state()
    assert state['active'] == True
    assert state['hashrate'] == 50.0
    assert state['balance'] == 1000.0
    assert state['token_balance'] == 0.0
    assert state['total_rewards'] == 0.0
    assert state['total_costs'] == 0.0
    assert state['blocks_found'] == 0

def test_miner_act():
    """Test miner agent decision making."""
    miner = Miner(agent_id="test_miner", strategy="aggressive")
    
    # Test profitable scenario
    state = {
        'token_price': 100.0,
        'network_difficulty': 1.0,
        'block_reward': 10.0,
        'network_hashrate': 1000.0
    }
    actions = miner.act(state)
    assert isinstance(actions, dict)
    assert 'participate' in actions
    assert 'hashrate_adjustment' in actions
    
    # Test unprofitable scenario (very low price and high network hashrate)
    state['token_price'] = 0.01  # Make price very low
    state['network_hashrate'] = 10000.0  # Increase competition
    actions = miner.act(state)
    assert actions['participate'] == False  # Should stop participating when very unprofitable

def test_miner_update():
    """Test miner state updates."""
    miner = Miner(agent_id="test_miner")
    
    # Test reward processing
    new_state = {
        'token_price': 100.0,
        'hashrate_adjustment': 10.0,
        'network_hashrate': 1000.0
    }
    miner.update(reward=5.0, new_state=new_state)
    
    state = miner.get_state()
    assert state['token_balance'] == 5.0
    assert state['total_rewards'] == 500.0  # 5 tokens * $100
    assert state['blocks_found'] == 1

def test_miner_reset():
    """Test miner reset functionality."""
    miner = Miner(agent_id="test_miner")
    
    # Make some changes to state
    new_state = {
        'token_price': 100.0,
        'hashrate_adjustment': 10.0
    }
    miner.update(reward=5.0, new_state=new_state)
    
    # Reset
    miner.reset()
    state = miner.get_state()
    assert state['token_balance'] == 0.0
    assert state['total_rewards'] == 0.0
    assert state['blocks_found'] == 0

def test_miner_strategies():
    """Test different miner strategies."""
    strategies = ['passive', 'aggressive', 'opportunistic']
    
    for strategy in strategies:
        miner = Miner(agent_id=f"test_miner_{strategy}", strategy=strategy)
        
        # Test highly profitable scenario
        state = {
            'token_price': 1000.0,
            'network_difficulty': 1.0,
            'block_reward': 10.0,
            'network_hashrate': 1000.0
        }
        actions = miner.act(state)
        assert actions['participate'] == True
        assert actions['hashrate_adjustment'] > 0  # Should increase hashrate
        
        # Test unprofitable scenario
        state['token_price'] = 0.1
        actions = miner.act(state)
        assert actions['participate'] == False
        assert actions['hashrate_adjustment'] < 0  # Should decrease hashrate 

def test_holder_initialization():
    """Test holder agent initialization."""
    holder = Holder(
        agent_id="test_holder",
        strategy="long_term",
        initial_balance=1000.0
    )
    
    # Test initial state
    state = holder.get_state()
    assert state['active'] == True
    assert state['balance'] == 1000.0
    assert state['token_balance'] == 0.0
    assert state['total_profit'] == 0.0
    assert state['holding_period'] == 0
    assert state['last_trade_price'] == 0.0

def test_holder_strategies():
    """Test different holder strategies."""
    strategies = ['long_term', 'medium_term', 'short_term']
    
    for strategy in strategies:
        holder = Holder(agent_id=f"test_holder_{strategy}", strategy=strategy)
        
        # Test profitable scenario
        current_state = {
            'price': 150.0,  # 50% increase
            'last_trade_price': 100.0
        }
        holder.state['last_trade_price'] = 100.0
        holder.state['token_balance'] = 100.0  # Give some tokens to potentially sell
        
        # Run multiple periods to trigger strategy-specific behaviors
        actions = []
        for _ in range(50):  # Simulate 50 periods
            action = holder.act(current_state)
            actions.append(action)
            
            # Update price to trigger trades
            if strategy == 'medium_term' and _ % 30 == 0:  # Monthly check
                current_state['price'] = 200.0  # Significant price increase
            elif strategy == 'short_term' and _ % 7 == 0:  # Weekly check
                current_state['price'] = 180.0  # Moderate price increase
        
        # Check strategy-specific behaviors
        trade_actions = [a for a in actions if a['trade']]
        if strategy == 'long_term':
            # Long-term holders should trade less frequently
            assert len(trade_actions) <= 2  # Expect very few trades
        elif strategy == 'medium_term':
            # Medium-term holders should trade monthly when price increases 20%
            assert len(trade_actions) >= 1  # Should have at least one trade
        elif strategy == 'short_term':
            # Short-term holders should trade weekly when price increases 10%
            assert len(trade_actions) >= 1  # Should have the most trades

def test_holder_update():
    """Test holder state updates."""
    holder = Holder(agent_id="test_holder")
    
    # Test reward processing
    current_state = {
        'last_trade_price': 100.0
    }
    holder.update(reward=50.0, current_state=current_state)
    
    state = holder.get_state()
    assert state['balance'] == 1050.0  # Initial 1000 + 50 reward
    assert state['total_profit'] == 50.0
    assert state['last_trade_price'] == 100.0
    
    # Test inactivation on low balance
    holder.state['balance'] = 5.0
    holder.state['token_balance'] = 5.0
    holder.update(reward=0.0, current_state={})
    assert holder.state['active'] == False

def test_holder_reset():
    """Test holder reset functionality."""
    holder = Holder(agent_id="test_holder")
    
    # Make some changes to state
    holder.state['balance'] = 2000.0
    holder.state['token_balance'] = 100.0
    holder.state['total_profit'] = 1000.0
    
    # Reset
    holder.reset()
    state = holder.get_state()
    assert state['active'] == True
    assert state['balance'] == 1000.0  # Should reset to initial balance
    assert state['token_balance'] == 0.0 

def test_trader_initialization():
    """Test trader agent initialization."""
    trader = Trader(
        agent_id="test_trader",
        strategy="momentum",
        initial_balance=10000.0,
        risk_tolerance=0.5
    )
    
    # Test initial state
    state = trader.get_state()
    assert state['active'] == True
    assert state['balance'] == 10000.0
    assert state['token_balance'] == 0.0
    assert state['total_profit'] == 0.0
    assert state['trades'] == 0
    assert state['last_trade_price'] == 0.0

def test_trader_strategies():
    """Test different trader strategies."""
    strategies = ['momentum', 'mean_reversion', 'random']
    
    for strategy in strategies:
        trader = Trader(
            agent_id=f"test_trader_{strategy}",
            strategy=strategy,
            risk_tolerance=0.5
        )
        
        # Test with price history
        state = {
            'token_price': 100.0,
            'price_history': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0]  # Upward trend
        }
        
        # For mean reversion, need more price history
        if strategy == 'mean_reversion':
            state['price_history'] = [100.0] * 15 + [120.0] * 5  # Price above MA
        
        action = trader.act(state)
        assert isinstance(action, dict)
        assert 'trade' in action
        assert 'amount' in action
        assert 'type' in action
        
        if strategy == 'momentum':
            # Should detect upward momentum
            if action['trade']:
                assert action['type'] == 'buy'
        elif strategy == 'mean_reversion':
            # Should detect price above MA
            if action['trade']:
                assert action['type'] == 'sell'

def test_trader_update():
    """Test trader state updates."""
    trader = Trader(agent_id="test_trader")
    
    # Test buy trade
    new_state = {
        'trade_type': 'buy',
        'trade_amount': 10.0,
        'token_price': 100.0
    }
    trader.update(reward=0.0, new_state=new_state)
    
    state = trader.get_state()
    assert state['token_balance'] == 10.0
    assert state['balance'] == 9000.0  # 10000 - (10 * 100)
    assert state['trades'] == 1
    assert state['last_trade_price'] == 100.0
    
    # Test sell trade
    new_state = {
        'trade_type': 'sell',
        'trade_amount': 5.0,
        'token_price': 120.0
    }
    trader.update(reward=0.0, new_state=new_state)
    
    state = trader.get_state()
    assert state['token_balance'] == 5.0
    assert state['balance'] == 9600.0  # 9000 + (5 * 120)
    assert state['trades'] == 2
    assert state['last_trade_price'] == 120.0

def test_trader_reset():
    """Test trader reset functionality."""
    trader = Trader(agent_id="test_trader")
    
    # Make some trades
    new_state = {
        'trade_type': 'buy',
        'trade_amount': 10.0,
        'token_price': 100.0
    }
    trader.update(reward=0.0, new_state=new_state)
    
    # Reset
    trader.reset()
    state = trader.get_state()
    assert state['active'] == True
    assert state['balance'] == 10000.0  # Should reset to initial balance
    assert state['token_balance'] == 0.0
    assert state['total_profit'] == 0.0 

def test_staker_initialization():
    """Test staker agent initialization."""
    staker = Staker(
        agent_id="test_staker",
        strategy="long_term",
        initial_balance=10000.0,
        min_stake_duration=30
    )
    
    # Test initial state
    state = staker.get_state()
    assert state['active'] == True
    assert state['balance'] == 10000.0
    assert state['staked_amount'] == 0.0
    assert state['total_rewards'] == 0.0
    assert state['stake_duration'] == 0
    assert state['current_validator'] == None

def test_staker_strategies():
    """Test different staker strategies."""
    strategies = ['long_term', 'dynamic', 'validator_hopping']
    
    for strategy in strategies:
        staker = Staker(
            agent_id=f"test_staker_{strategy}",
            strategy=strategy,
            min_stake_duration=30
        )
        
        # Test with active validators and good APY
        state = {
            'staking_apy': 0.15,  # 15% APY
            'active_validators': ['validator1', 'validator2', 'validator3'],
            'validator_stats': {
                'validator1': {'rewards': 100},
                'validator2': {'rewards': 150},
                'validator3': {'rewards': 120}
            }
        }
        
        action = staker.act(state)
        assert isinstance(action, dict)
        assert 'stake' in action
        assert 'amount' in action
        assert 'validator' in action
        assert 'unstake' in action
        
        if strategy == 'long_term':
            # Should stake maximum amount
            if action['stake']:
                assert action['amount'] == staker.state['balance'] * 0.9
        elif strategy == 'dynamic':
            # Should stake with high APY
            if action['stake']:
                assert action['amount'] == staker.state['balance'] * 0.7
        elif strategy == 'validator_hopping':
            # Should not stake initially (need duration)
            assert not action['stake']

def test_staker_update():
    """Test staker state updates."""
    staker = Staker(agent_id="test_staker")
    
    # Test staking
    new_state = {
        'stake_action': 'stake',
        'stake_amount': 5000.0,
        'validator': 'validator1'
    }
    staker.update(reward=0.0, new_state=new_state)
    
    state = staker.get_state()
    assert state['staked_amount'] == 5000.0
    assert state['balance'] == 5000.0  # 10000 - 5000
    assert state['current_validator'] == 'validator1'
    assert state['stake_duration'] == 1  # Duration increases when staked
    
    # Test reward accumulation
    staker.update(reward=100.0, new_state={'stake_action': None})
    state = staker.get_state()  # Get updated state
    assert state['total_rewards'] == 100.0
    assert state['stake_duration'] == 2  # Duration increases each update
    
    # Test unstaking
    new_state = {
        'stake_action': 'unstake',
        'stake_amount': 5000.0
    }
    staker.update(reward=0.0, new_state=new_state)
    
    state = staker.get_state()
    assert state['staked_amount'] == 0.0
    assert state['balance'] == 10000.0  # Original balance restored
    assert state['current_validator'] == None

def test_validator_hopping():
    """Test validator hopping strategy specifically."""
    staker = Staker(
        agent_id="test_staker",
        strategy="validator_hopping",
        min_stake_duration=5
    )
    
    # Initial stake
    new_state = {
        'stake_action': 'stake',
        'stake_amount': 5000.0,
        'validator': 'validator1'
    }
    staker.update(reward=0.0, new_state=new_state)
    
    # Simulate passage of time and validator performance
    for _ in range(10):  # Pass minimum stake duration
        staker.update(reward=10.0, new_state={'stake_action': None})
    
    # Test hopping to better validator
    state = {
        'staking_apy': 0.15,
        'active_validators': ['validator1', 'validator2'],
        'validator_stats': {
            'validator1': {'rewards': 100},
            'validator2': {'rewards': 200}  # Better performer
        }
    }
    
    action = staker.act(state)
    assert action['unstake'] == True
    assert action['stake'] == True
    assert action['validator'] == 'validator2'  # Should choose better validator
    assert action['amount'] == staker.state['staked_amount']  # Should restake same amount 