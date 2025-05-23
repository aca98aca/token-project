from typing import Dict, Any, Optional
import random
import numpy as np
from . import Agent

class Miner(Agent):
    """Miner agent that participates in the consensus mechanism with improved strategies."""
    
    def __init__(self, 
                 agent_id: str,
                 strategy: str = 'profit_maximizer',
                 initial_balance: float = 10000.0,
                 initial_tokens: float = 0.0,
                 initial_hashrate: float = 1000.0,
                 electricity_cost: float = 0.1,  # Cost per unit of hashrate
                 efficiency: float = 0.8):  # Mining efficiency
        super().__init__(agent_id, strategy)
        self.maintenance_cost = electricity_cost
        self.efficiency = efficiency
        self.state.update({
            'balance': initial_balance,
            'token_balance': initial_tokens,
            'initial_balance': initial_balance,
            'hashrate': initial_hashrate,
            'base_hashrate': initial_hashrate,  # Store initial hashrate
            'maintenance_cost': electricity_cost * initial_hashrate,
            'efficiency': efficiency,
            'active': True,
            'rewards': 0.0,
            'costs': 0.0,
            'profit': 0.0,
            'last_adjustment': 0,
            'adjustment_cooldown': 10,  # Minimum steps between adjustments
            'participate': True,
            'hashrate_adjustment': 0.0
        })
    
    def initialize(self, initial_balance: float = 1000.0) -> None:
        """Initialize the miner with initial balance."""
        super().initialize(initial_balance)
        self.state.update({
            'hashrate': self.state['base_hashrate'],
            'blocks_mined': 0,
            'total_rewards': 0.0,
            'total_costs': 0.0,
            'blocks_found': 0,
            'network_hashrate': 0.0,
            'profitability_history': [],
            'hardware_status': 'active',
            'last_maintenance': 0,
            'maintenance_interval': 100
        })
    
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Decide mining actions based on strategy."""
        actions = {'adjust_hashrate': False, 'amount': 0.0}
        
        if not self.state['active']:
            return actions
            
        current_price = state.get('price', 0.0)
        if current_price == 0:
            return actions
            
        # Calculate mining profitability
        block_reward = state.get('block_reward', 0.0)
        network_hashrate = state.get('network_hashrate', 1.0)
        network_share = self.state['hashrate'] / network_hashrate
        
        # Calculate expected rewards and costs
        expected_reward = block_reward * network_share * current_price
        maintenance_cost = self.state['maintenance_cost'] * self.state['efficiency']
        
        # Strategy-specific actions
        if self.strategy == 'profit_maximizer':
            # Adjust hashrate based on profitability
            if self.state['last_adjustment'] >= self.state['adjustment_cooldown']:
                if expected_reward > maintenance_cost * 1.2:  # 20% profit margin
                    # Increase hashrate
                    actions['adjust_hashrate'] = True
                    actions['amount'] = self.state['base_hashrate'] * 0.1  # 10% increase
                elif expected_reward < maintenance_cost * 0.8:  # 20% loss margin
                    # Decrease hashrate
                    actions['adjust_hashrate'] = True
                    actions['amount'] = -self.state['base_hashrate'] * 0.1  # 10% decrease
                
                if actions['adjust_hashrate']:
                    self.state['last_adjustment'] = 0
        
        elif self.strategy == 'network_builder':
            # Focus on network growth
            if self.state['last_adjustment'] >= self.state['adjustment_cooldown']:
                if network_share < 0.1:  # Target 10% network share
                    actions['adjust_hashrate'] = True
                    actions['amount'] = self.state['base_hashrate'] * 0.2  # 20% increase
                elif network_share > 0.15:  # Cap at 15% network share
                    actions['adjust_hashrate'] = True
                    actions['amount'] = -self.state['base_hashrate'] * 0.1  # 10% decrease
                
                if actions['adjust_hashrate']:
                    self.state['last_adjustment'] = 0
        
        self.state['last_adjustment'] += 1
        return actions
    
    def update(self, reward: float, state: Dict[str, Any]) -> None:
        """Update miner state based on rewards and state."""
        # Update token balance with block reward
        self.state['token_balance'] += reward
        
        # Convert tokens to fiat at current price
        token_price = state.get('price', 1.0)
        fiat_reward = reward * token_price
        self.state['balance'] += fiat_reward
        self.state['rewards'] += fiat_reward
        
        # Update hashrate if there was an adjustment
        if state.get('hashrate_adjustment'):
            self.state['hashrate'] = max(0.0, self.state['hashrate'] + state['hashrate_adjustment'])
        
        # Update costs
        electricity_cost = (self.state['hashrate'] * self.state['maintenance_cost']) / 6  # 6 blocks per hour
        self.state['costs'] += electricity_cost
        self.state['balance'] -= electricity_cost
        
        # Update blocks found
        if reward > 0:
            self.state['blocks_found'] = self.state.get('blocks_found', 0) + 1
        
        # Update network hashrate
        if 'network_hashrate' in state:
            self.state['network_hashrate'] = state['network_hashrate']
        
        # Update efficiency if maintenance was performed
        if state.get('maintenance', False):
            self.state['efficiency'] = min(1.0, self.state['efficiency'] + 0.1)
        
        # Update profit
        self.state['profit'] = self.state['rewards'] - self.state['costs']
        
        # Check if miner should become inactive
        if self.state['balance'] < self.state['maintenance_cost'] * 10:  # Can't afford 10 steps of maintenance
            self.state['active'] = False
            self.state['hashrate'] = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the miner."""
        return self.state.copy()
    
    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self.state.update({
            'hashrate': self.state['base_hashrate'],
            'blocks_mined': 0,
            'total_rewards': 0.0,
            'total_costs': 0.0,
            'blocks_found': 0,
            'network_hashrate': 0.0,
            'profitability_history': [],
            'hardware_status': 'active',
            'last_maintenance': 0,
            'maintenance_interval': 100
        })

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