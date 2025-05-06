from typing import Dict, Any
import random
from token_sim.agents import Agent

class Holder(Agent):
    """Agent that holds tokens with different time horizons."""
    
    def __init__(self,
                 agent_id: str,
                 strategy: str = 'long_term',
                 initial_balance: float = 1000.0,
                 initial_tokens: float = 0.0):
        self.agent_id = agent_id
        self.id = agent_id  # Alias for backward compatibility
        self.strategy = strategy
        self.initial_tokens = initial_tokens
        self.state = {
            'active': True,
            'balance': initial_balance,
            'token_balance': initial_tokens,
            'initial_balance': initial_balance,
            'total_profit': 0.0,
            'holding_period': 0,
            'last_trade_price': 0.0
        }
    
    def initialize(self) -> None:
        """Initialize the holder's state."""
        self.state['balance'] = self.state['initial_balance']
        self.state['token_balance'] = self.initial_tokens
        self.state['active'] = True
        self.state['total_profit'] = 0.0
        self.state['holding_period'] = 0
        self.state['last_trade_price'] = 0.0
    
    def act(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Determine holding action based on strategy."""
        current_price = current_state['price']
        
        # Skip if not active
        if not self.state['active']:
            return {'trade': False}
        
        # Update holding period
        self.state['holding_period'] += 1
        
        # Strategy-specific actions
        if self.strategy == 'long_term':
            # Long-term holders rarely trade
            if random.random() < 0.01:  # 1% chance to trade
                if current_price > self.state['last_trade_price'] * 1.5:
                    return {
                        'trade': True,
                        'type': 'sell',
                        'amount': self.state['token_balance'] * 0.1  # Sell 10% of holdings
                    }
        
        elif self.strategy == 'medium_term':
            # Medium-term holders trade based on price movements
            if self.state['holding_period'] % 30 == 0:  # Check monthly
                if current_price > self.state['last_trade_price'] * 1.2:
                    return {
                        'trade': True,
                        'type': 'sell',
                        'amount': self.state['token_balance'] * 0.2  # Sell 20% of holdings
                    }
                elif current_price < self.state['last_trade_price'] * 0.8:
                    return {
                        'trade': True,
                        'type': 'buy',
                        'amount': self.state['balance'] * 0.2  # Use 20% of balance
                    }
        
        else:  # short_term
            # Short-term holders trade more frequently
            if self.state['holding_period'] % 7 == 0:  # Check weekly
                if current_price > self.state['last_trade_price'] * 1.1:
                    return {
                        'trade': True,
                        'type': 'sell',
                        'amount': self.state['token_balance'] * 0.3  # Sell 30% of holdings
                    }
                elif current_price < self.state['last_trade_price'] * 0.9:
                    return {
                        'trade': True,
                        'type': 'buy',
                        'amount': self.state['balance'] * 0.3  # Use 30% of balance
                    }
        
        return {'trade': False}
    
    def update(self, reward: float, current_state: Dict[str, Any]) -> None:
        """Update holder's state based on rewards and market conditions."""
        if not self.state['active']:
            return
        
        # Update balances
        self.state['balance'] += reward
        self.state['total_profit'] += reward
        
        # Update last trade price if we traded
        if current_state.get('last_trade_price'):
            self.state['last_trade_price'] = current_state['last_trade_price']
        
        # Check if holder should become inactive
        if self.state['balance'] < 10.0 and self.state['token_balance'] < 10.0:
            self.state['active'] = False
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the holder."""
        return self.state.copy()
    
    def reset(self):
        """Reset agent state."""
        self.state = {
            'active': True,
            'balance': self.state['initial_balance'],  # Reset to initial balance
            'token_balance': self.initial_tokens,
            'total_profit': 0.0,
            'initial_balance': self.state['initial_balance'],  # Keep initial balance
            'holding_period': 0,
            'last_trade_price': 0.0
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