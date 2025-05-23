from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class Agent(ABC):
    """Base class for all agents in the simulation."""
    
    def __init__(self, agent_id: str, strategy: str = 'default'):
        self.agent_id = agent_id
        self.id = agent_id  # Alias for backward compatibility
        self.strategy = strategy
        self.state = {
            'active': True,
            'balance': 0.0,
            'token_balance': 0.0,
            'total_profit': 0.0,
            'initial_balance': 0.0,
            'trades': 0,
            'last_trade_price': 0.0,
            'performance_metrics': {}
        }
    
    @abstractmethod
    def initialize(self, initial_balance: float = 0.0) -> None:
        """Initialize the agent with its initial state."""
        self.state['balance'] = initial_balance
        self.state['initial_balance'] = initial_balance
        self.state['active'] = True
        self.state['total_profit'] = 0.0
        self.state['trades'] = 0
        self.state['last_trade_price'] = 0.0
        self.state['performance_metrics'] = {}
    
    @abstractmethod
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform an action based on the current state.
        
        Args:
            state: Current state of the simulation
            
        Returns:
            Dict containing the agent's actions
        """
        pass
    
    @abstractmethod
    def update(self, reward: float, new_state: Dict[str, Any]) -> None:
        """Update agent's state based on received reward and new state.
        
        Args:
            reward: Reward received from the last action
            new_state: New state of the simulation
        """
        self.state['total_profit'] += reward
        self._update_performance_metrics(new_state)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent."""
        return self.state.copy()
    
    def get_performance_metric(self) -> float:
        """Get a performance metric for the agent."""
        return self.state.get('total_profit', 0.0)
    
    def _update_performance_metrics(self, new_state: Dict[str, Any]) -> None:
        """Update performance metrics based on new state."""
        metrics = self.state['performance_metrics']
        
        # Calculate ROI
        initial_balance = self.state['initial_balance']
        current_balance = self.state['balance']
        if initial_balance > 0:
            metrics['roi'] = (current_balance - initial_balance) / initial_balance
        
        # Calculate win rate if trades exist
        if self.state['trades'] > 0:
            metrics['win_rate'] = metrics.get('winning_trades', 0) / self.state['trades']
        
        # Update Sharpe ratio if we have enough data
        if 'returns_history' not in metrics:
            metrics['returns_history'] = []
        metrics['returns_history'].append(self.state['total_profit'])
        if len(metrics['returns_history']) > 1:
            returns = np.array(metrics['returns_history'])
            metrics['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-6)
    
    def reset(self) -> None:
        """Reset agent state to initial values."""
        self.initialize(self.state['initial_balance'])
    
    def save_state(self, path: str) -> None:
        """Save agent state to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.state, f)
    
    def load_state(self, path: str) -> None:
        """Load agent state from disk."""
        import pickle
        with open(path, 'rb') as f:
            self.state = pickle.load(f) 