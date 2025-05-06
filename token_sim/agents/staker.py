from typing import Dict, Any
import random
from . import Agent

class Staker(Agent):
    """Staker agent that participates in staking activities."""
    
    def __init__(self, 
                 agent_id: str,
                 strategy: str = 'long_term',
                 initial_balance: float = 10000.0,
                 min_stake_duration: int = 30):  # minimum days to stake
        self.agent_id = agent_id
        self.strategy = strategy
        self.min_stake_duration = min_stake_duration
        self.state = {
            'balance': initial_balance,
            'staked_amount': 0.0,
            'active': True,
            'total_rewards': 0.0,
            'stake_duration': 0,
            'current_validator': None
        }
    
    def initialize(self, initial_balance: float = 10000.0) -> None:
        """Initialize the staker with initial balance."""
        self.state['balance'] = initial_balance
    
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Decide on staking actions based on current state and strategy."""
        actions = {
            'stake': False,
            'amount': 0.0,
            'validator': None,
            'unstake': False
        }
        
        if not self.state['active']:
            return actions
            
        # Get current APY and validator list
        current_apy = state.get('staking_apy', 0.0)
        validators = state.get('active_validators', [])
        
        if not validators:
            return actions
            
        # Strategy-based decisions
        if self.strategy == 'long_term':
            # Stake maximum amount for long periods
            if self.state['staked_amount'] == 0 and self.state['balance'] > 0:
                actions['stake'] = True
                actions['amount'] = self.state['balance'] * 0.9  # Keep 10% liquid
                actions['validator'] = random.choice(validators)
        
        elif self.strategy == 'dynamic':
            # Adjust stake based on APY and market conditions
            if current_apy > 0.1:  # 10% APY threshold
                if self.state['staked_amount'] == 0 and self.state['balance'] > 0:
                    actions['stake'] = True
                    actions['amount'] = self.state['balance'] * 0.7  # Keep 30% liquid
                    actions['validator'] = random.choice(validators)
            elif self.state['stake_duration'] > self.min_stake_duration:
                actions['unstake'] = True
        
        elif self.strategy == 'validator_hopping':
            # Switch validators to optimize rewards
            if self.state['stake_duration'] > self.min_stake_duration:
                validator_stats = state.get('validator_stats', {})
                if validator_stats:
                    # Find validator with highest rewards
                    best_validator = max(validator_stats.items(), 
                                      key=lambda x: x[1]['rewards'])
                    if (self.state['current_validator'] != best_validator[0] and
                        best_validator[1]['rewards'] > 0):
                        actions['unstake'] = True
                        actions['stake'] = True
                        actions['amount'] = self.state['staked_amount']
                        actions['validator'] = best_validator[0]
        
        return actions
    
    def update(self, reward: float, new_state: Dict[str, Any]) -> None:
        """Update staker's state based on rewards and new state."""
        self.state['total_rewards'] += reward
        
        # Update staked amount and balance
        if 'stake_amount' in new_state:
            if new_state.get('stake_action') == 'stake':
                self.state['staked_amount'] += new_state['stake_amount']
                self.state['balance'] -= new_state['stake_amount']
                self.state['current_validator'] = new_state.get('validator')
                self.state['stake_duration'] = 0
            elif new_state.get('stake_action') == 'unstake':
                self.state['balance'] += self.state['staked_amount']
                self.state['staked_amount'] = 0
                self.state['current_validator'] = None
        
        # Update stake duration
        if self.state['staked_amount'] > 0:
            self.state['stake_duration'] += 1
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the staker."""
        return self.state.copy() 