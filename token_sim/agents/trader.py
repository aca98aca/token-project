from typing import Dict, Any
import random
import numpy as np
from . import Agent

class Trader(Agent):
    """Trader agent that participates in market activities."""
    
    def __init__(self, 
                 agent_id: str,
                 strategy: str = 'momentum',
                 initial_balance: float = 10000.0,
                 risk_tolerance: float = 0.5,
                 initial_tokens: float = 0.0):
        self.agent_id = agent_id
        self.id = agent_id  # Alias for backward compatibility
        self.strategy = strategy
        self.risk_tolerance = risk_tolerance
        self.state = {
            'active': True,
            'balance': initial_balance,
            'token_balance': initial_tokens,
            'total_profit': 0.0,
            'initial_balance': initial_balance,
            'trades': 0,
            'last_trade_price': 0.0
        }
    
    def initialize(self, initial_balance: float = 10000.0) -> None:
        """Initialize the trader with initial balance."""
        self.state['balance'] = initial_balance
    
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Decide on trading actions based on current state and strategy."""
        actions = {
            'trade': False,
            'amount': 0.0,
            'type': 'buy'  # or 'sell'
        }
        
        if not self.state['active']:
            return actions
            
        current_price = state.get('token_price', 0.0)
        if current_price == 0:
            return actions
            
        # Strategy-based decisions
        if self.strategy == 'momentum':
            # Look at price history for momentum
            price_history = state.get('price_history', [])
            if len(price_history) >= 2:
                price_change = (price_history[-1] - price_history[-2]) / price_history[-2]
                if abs(price_change) > 0.02:  # 2% threshold
                    actions['trade'] = True
                    actions['type'] = 'buy' if price_change > 0 else 'sell'
                    # Amount based on risk tolerance and price change
                    actions['amount'] = self.state['balance'] * self.risk_tolerance * abs(price_change)
        
        elif self.strategy == 'mean_reversion':
            # Look for price deviations from moving average
            price_history = state.get('price_history', [])
            if len(price_history) >= 20:
                ma = np.mean(price_history[-20:])
                deviation = (current_price - ma) / ma
                if abs(deviation) > 0.05:  # 5% threshold
                    actions['trade'] = True
                    actions['type'] = 'sell' if deviation > 0 else 'buy'
                    actions['amount'] = self.state['balance'] * self.risk_tolerance * abs(deviation)
        
        elif self.strategy == 'random':
            # Random trading decisions
            if random.random() < 0.1:  # 10% chance to trade
                actions['trade'] = True
                actions['type'] = 'buy' if random.random() < 0.5 else 'sell'
                actions['amount'] = self.state['balance'] * random.uniform(0.1, 0.3)
        
        return actions
    
    def update(self, reward: float, new_state: Dict[str, Any]) -> None:
        """Update trader's state based on rewards and new state."""
        self.state['balance'] += reward
        self.state['total_profit'] += reward
        
        # Update token balance if there was a trade
        if 'trade_amount' in new_state:
            if new_state['trade_type'] == 'buy':
                self.state['token_balance'] += new_state['trade_amount']
                self.state['balance'] -= new_state['trade_amount'] * new_state['token_price']
            else:  # sell
                self.state['token_balance'] -= new_state['trade_amount']
                self.state['balance'] += new_state['trade_amount'] * new_state['token_price']
            
            self.state['trades'] += 1
            self.state['last_trade_price'] = new_state['token_price']
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the trader."""
        return self.state.copy()
    
    def reset(self):
        """Reset agent state."""
        self.state = {
            'active': True,
            'balance': self.state['initial_balance'],  # Reset to initial balance
            'token_balance': self.state['token_balance'],
            'total_profit': 0.0,
            'initial_balance': self.state['initial_balance']  # Keep initial balance
        } 
    
    def save_model(self, path: str) -> None:
        """Save the agent's model to disk (for AI agents)."""
        if hasattr(self, 'model') and self.model is not None:
            try:
                import pickle
                with open(path, 'wb') as f:
                    pickle.dump(self.model, f)
            except Exception as e:
                print(f"Error saving model: {e}")
    
    def load_model(self, path: str) -> None:
        """Load the agent's model from disk (for AI agents)."""
        try:
            import pickle
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}") 