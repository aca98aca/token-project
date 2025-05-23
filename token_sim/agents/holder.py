from typing import Dict, Any
import random
import numpy as np
from token_sim.agents import Agent

class Holder(Agent):
    """Agent that holds tokens with different time horizons."""
    
    def __init__(self,
                 agent_id: str,
                 strategy: str = 'long_term',
                 initial_balance: float = 10000.0,
                 initial_tokens: float = 1000.0,
                 holding_period: int = 0,
                 min_holding_period: int = 10,
                 profit_target: float = 0.2,  # Decreased from 0.5
                 loss_threshold: float = 0.1):  # Decreased from 0.2
        super().__init__(agent_id, strategy)
        self.min_holding_period = min_holding_period
        self.profit_target = profit_target
        self.loss_threshold = loss_threshold
        self.initial_tokens = initial_tokens
        self.state = {
            'active': True,
            'balance': initial_balance,
            'token_balance': initial_tokens,
            'initial_balance': initial_balance,
            'total_profit': 0.0,
            'holding_period': holding_period,
            'last_trade_price': 0.0,
            'entry_price': 0.0,
            'trades': 0,
            'trade_history': []
        }
    
    def initialize(self) -> None:
        """Initialize the holder's state."""
        self.state.update({
            'active': True,
            'balance': self.state['initial_balance'],
            'token_balance': self.initial_tokens,
            'total_profit': 0.0,
            'holding_period': 0,
            'last_trade_price': 0.0,
            'entry_price': 0.0,
            'trades': 0,
            'trade_history': []
        })
    
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Decide holding actions based on strategy."""
        actions = {'trade': False, 'amount': 0.0, 'type': 'buy'}
        
        if not self.state['active']:
            return actions
            
        current_price = state.get('price', 0.0)
        if current_price == 0:
            return actions
            
        # Update holding period
        self.state['holding_period'] += 1
        
        # Check if we should trade based on strategy
        if self.strategy == 'long_term':
            # Long-term holders trade more frequently (5% chance)
            if random.random() < 0.05:
                # Consider market conditions
                price_history = state.get('price_history', [])
                if len(price_history) >= 20:
                    short_ma = np.mean(price_history[-5:])
                    long_ma = np.mean(price_history[-20:])
                    trend = (short_ma - long_ma) / long_ma
                    
                    # Sell if trend is negative and we're in profit
                    if trend < -0.02 and current_price > self.state['last_trade_price'] * 1.1:
                        actions['trade'] = True
                        actions['type'] = 'sell'
                        actions['amount'] = self.state['token_balance'] * 0.5  # Sell half
                    # Buy if trend is positive and we have enough balance
                    elif trend > 0.02 and self.state['balance'] > current_price * 100:
                        actions['trade'] = True
                        actions['type'] = 'buy'
                        actions['amount'] = self.state['balance'] * 0.3  # Use 30% of balance
        else:
            # Short-term holders are more active
            if self.state['holding_period'] >= self.min_holding_period:
                # Check profit/loss conditions
                if self.state['last_trade_price'] > 0:
                    price_change = (current_price - self.state['last_trade_price']) / self.state['last_trade_price']
                    
                    if price_change >= self.profit_target:
                        actions['trade'] = True
                        actions['type'] = 'sell'
                        actions['amount'] = self.state['token_balance'] * 0.7  # Sell 70%
                    elif price_change <= -self.loss_threshold:
                        actions['trade'] = True
                        actions['type'] = 'sell'
                        actions['amount'] = self.state['token_balance'] * 0.3  # Sell 30% to cut losses
        
        return actions
    
    def update(self, state: Dict[str, Any], action_result: Dict[str, Any]) -> None:
        """Update holder state based on action results."""
        if action_result.get('success', False):
            self.state['trades'] += 1
            self.state['trade_history'].append(action_result)
            
            if action_result['type'] == 'buy':
                self.state['balance'] -= action_result['cost']
                self.state['token_balance'] += action_result['amount']
                self.state['last_trade_price'] = action_result['price']
                self.state['entry_price'] = action_result['price']
                self.state['holding_period'] = 0
            else:  # sell
                self.state['balance'] += action_result['value']
                self.state['token_balance'] -= action_result['amount']
                self.state['last_trade_price'] = action_result['price']
                self.state['holding_period'] = 0
                
            # Check if holder should become inactive
            if self.state['balance'] < 100 and self.state['token_balance'] < 100:
                self.state['active'] = False
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the holder."""
        return self.state.copy()
    
    def reset(self):
        """Reset agent state."""
        self.state = {
            'active': True,
            'balance': self.state['initial_balance'],  # Reset to initial balance
            'token_balance': self.state['initial_tokens'],
            'total_profit': 0.0,
            'initial_balance': self.state['initial_balance'],  # Keep initial balance
            'holding_period': 0,
            'last_trade_price': 0.0,
            'entry_price': 0.0,
            'trades': 0,
            'trade_history': []
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