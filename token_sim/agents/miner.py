from typing import Dict, Any
import random
from . import Agent

class Miner(Agent):
    """Miner agent that participates in the consensus mechanism."""
    
    def __init__(self, 
                 agent_id: str,
                 strategy: str = 'passive',
                 initial_hashrate: float = 50.0,
                 electricity_cost: float = 0.1,
                 initial_balance: float = 1000.0):  # Increased initial balance
        self.agent_id = agent_id
        self.id = agent_id  # Alias for backward compatibility
        self.strategy = strategy
        self.initial_hashrate = initial_hashrate
        self.electricity_cost = electricity_cost
        self.initial_balance = initial_balance
        self.state = {
            'active': True,
            'hashrate': initial_hashrate,
            'balance': initial_balance,
            'token_balance': 0.0,
            'total_profit': 0.0,
            'initial_balance': initial_balance,
            'electricity_cost': electricity_cost,
            'blocks_mined': 0,
            'total_rewards': 0.0,
            'total_costs': 0.0,
            'blocks_found': 0,
            'network_hashrate': 0.0
        }
    
    def initialize(self, initial_balance: float = 1000.0) -> None:
        """Initialize the miner with initial balance."""
        self.state['balance'] = initial_balance
        self.state['active'] = True
        self.state['hashrate'] = self.initial_hashrate
    
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Decide on mining actions based on current state and strategy."""
        actions = {
            'participate': True,
            'hashrate_adjustment': 0.0
        }
        
        # Get current state
        token_price = state.get('token_price', 1.0)
        network_difficulty = state.get('network_difficulty', 1.0)
        block_reward = state.get('block_reward', 1.0)
        network_hashrate = state.get('network_hashrate', self.initial_hashrate * 25)  # Default to 25 miners
        
        # Update network hashrate in state
        self.state['network_hashrate'] = network_hashrate
        
        # Calculate mining profitability
        # Probability of finding a block = miner_hashrate / total_hashrate
        block_probability = self.state['hashrate'] / (network_hashrate + 1e-6)  # Avoid division by zero
        
        # Expected revenue per block
        expected_revenue = block_probability * block_reward * token_price
        
        # Calculate costs per block (assuming 10 minute blocks)
        costs_per_block = (self.state['hashrate'] * self.electricity_cost) / 6  # 6 blocks per hour
        
        # Strategy-based decisions
        if self.strategy == 'aggressive':
            if expected_revenue > costs_per_block * 1.2:  # 20% profit margin
                actions['hashrate_adjustment'] = self.initial_hashrate * 0.5  # Increase by 50%
            elif expected_revenue < costs_per_block * 0.8:  # 20% loss threshold
                actions['participate'] = False
                actions['hashrate_adjustment'] = -self.state['hashrate'] * 0.2  # Reduce by 20%
        
        elif self.strategy == 'passive':
            if expected_revenue > costs_per_block * 1.1:  # 10% profit margin
                actions['hashrate_adjustment'] = self.initial_hashrate * 0.2  # Increase by 20%
            elif expected_revenue < costs_per_block * 0.9:  # 10% loss threshold
                actions['participate'] = False
                actions['hashrate_adjustment'] = -self.state['hashrate'] * 0.1  # Reduce by 10%
        
        elif self.strategy == 'opportunistic':
            if expected_revenue > costs_per_block * 1.5:  # 50% profit margin
                actions['hashrate_adjustment'] = self.initial_hashrate * 0.8  # Increase by 80%
            elif expected_revenue < costs_per_block * 0.95:  # 5% loss threshold
                actions['participate'] = False
                actions['hashrate_adjustment'] = -self.state['hashrate'] * 0.3  # Reduce by 30%
        
        # Ensure hashrate doesn't go below 10% of initial
        if self.state['hashrate'] + actions['hashrate_adjustment'] < self.initial_hashrate * 0.1:
            actions['hashrate_adjustment'] = self.initial_hashrate * 0.1 - self.state['hashrate']
        
        return actions
    
    def update(self, reward: float, new_state: Dict[str, Any]) -> None:
        """Update miner's state based on rewards and new state."""
        # Update token balance with block reward
        self.state['token_balance'] += reward
        
        # Convert tokens to fiat at current price
        token_price = new_state.get('token_price', 1.0)
        fiat_reward = reward * token_price
        self.state['balance'] += fiat_reward
        self.state['total_rewards'] += fiat_reward
        
        # Update hashrate if there was an adjustment
        if 'hashrate_adjustment' in new_state:
            self.state['hashrate'] = max(0.0, self.state['hashrate'] + new_state['hashrate_adjustment'])
        
        # Update costs (per block)
        costs = (self.state['hashrate'] * self.electricity_cost) / 6  # 6 blocks per hour
        self.state['total_costs'] += costs
        self.state['balance'] -= costs
        
        # Update blocks found
        if reward > 0:
            self.state['blocks_found'] += 1
        
        # Update network hashrate
        if 'network_hashrate' in new_state:
            self.state['network_hashrate'] = new_state['network_hashrate']
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the miner."""
        return self.state.copy()

    def reset(self):
        """Reset agent state."""
        self.state = {
            'active': True,
            'hashrate': self.initial_hashrate,
            'balance': self.initial_balance,
            'token_balance': 0.0,
            'total_profit': 0.0,
            'initial_balance': self.initial_balance,
            'electricity_cost': self.electricity_cost,
            'blocks_mined': 0,
            'total_rewards': 0.0,
            'total_costs': 0.0,
            'blocks_found': 0,
            'network_hashrate': 0.0
        } 

    def is_operational(self) -> bool:
        """Check if miner is operational."""
        # A miner is operational if:
        # 1. It is active
        # 2. Has positive balance for electricity costs
        # 3. Has positive hashrate
        return (
            self.state['active'] and 
            self.state['balance'] > 0 and 
            self.state['hashrate'] > 0
        ) 