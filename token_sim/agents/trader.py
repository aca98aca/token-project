from typing import Dict, Any, List, Optional
import random
import numpy as np
from . import Agent

class Trader(Agent):
    """Trader agent that participates in market activities with improved strategies."""
    
    def __init__(self, 
                 agent_id: str,
                 strategy: str = 'momentum',
                 initial_balance: float = 10000.0,
                 risk_tolerance: float = 0.7,  # Increased from 0.5
                 initial_tokens: float = 0.0,
                 max_position_size: float = 0.4,  # Increased from 0.2
                 stop_loss: float = 0.05,  # Decreased from 0.1
                 take_profit: float = 0.1):  # Decreased from 0.2
        super().__init__(agent_id, strategy)
        self.risk_tolerance = risk_tolerance
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.state.update({
            'balance': initial_balance,
            'token_balance': initial_tokens,
            'initial_balance': initial_balance,
            'trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'last_trade_price': 0.0,
            'entry_price': 0.0,
            'position_size': 0.0,
            'trade_history': []
        })
    
    def initialize(self, initial_balance: float = 10000.0) -> None:
        """Initialize the trader with initial balance."""
        super().initialize(initial_balance)
        self.state.update({
            'trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'last_trade_price': 0.0,
            'entry_price': 0.0,
            'position_size': 0.0,
            'trade_history': []
        })
    
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Decide on trading actions based on current state and strategy."""
        actions = {
            'trade': False,
            'amount': 0.0,
            'type': 'buy'
        }
        
        if not self.state['active']:
            return actions
            
        current_price = state.get('token_price', 0.0)
        if current_price == 0:
            return actions
        
        # Check stop loss and take profit
        if self.state['position_size'] > 0:
            price_change = (current_price - self.state['entry_price']) / self.state['entry_price']
            if price_change <= -self.stop_loss:
                actions['trade'] = True
                actions['type'] = 'sell'
                actions['amount'] = self.state['position_size']
                self.state['losing_trades'] += 1
            elif price_change >= self.take_profit:
                actions['trade'] = True
                actions['type'] = 'sell'
                actions['amount'] = self.state['position_size']
                self.state['winning_trades'] += 1
        
        # Strategy-based decisions
        if self.strategy == 'momentum':
            actions.update(self._momentum_strategy(state))
        elif self.strategy == 'mean_reversion':
            actions.update(self._mean_reversion_strategy(state))
        elif self.strategy == 'technical':
            actions.update(self._technical_strategy(state))
        elif self.strategy == 'random':
            actions.update(self._random_strategy())
        
        # Apply risk management
        if actions['trade']:
            actions['amount'] = self._apply_risk_management(actions['amount'], actions['type'])
        
        return actions
    
    def _momentum_strategy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Momentum trading strategy."""
        actions = {'trade': False, 'amount': 0.0, 'type': 'buy'}
        
        price_history = state.get('price_history', [])
        if len(price_history) >= 2:
            # Calculate momentum indicators
            short_ma = np.mean(price_history[-5:]) if len(price_history) >= 5 else price_history[-1]  # Shorter window
            long_ma = np.mean(price_history[-20:]) if len(price_history) >= 20 else short_ma  # Shorter window
            
            # Calculate momentum
            momentum = (short_ma - long_ma) / long_ma
            
            if abs(momentum) > 0.01:  # Lower threshold to 1%
                actions['trade'] = True
                actions['type'] = 'buy' if momentum > 0 else 'sell'
                actions['amount'] = self.state['balance'] * self.risk_tolerance * abs(momentum) * 2  # Double the impact
        
        return actions
    
    def _mean_reversion_strategy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Mean reversion trading strategy."""
        actions = {'trade': False, 'amount': 0.0, 'type': 'buy'}
        
        price_history = state.get('price_history', [])
        if len(price_history) >= 20:
            # Calculate Bollinger Bands
            ma = np.mean(price_history[-20:])
            std = np.std(price_history[-20:])
            upper_band = ma + 2 * std
            lower_band = ma - 2 * std
            
            current_price = price_history[-1]
            deviation = (current_price - ma) / ma
            
            if abs(deviation) > 0.05:  # 5% threshold
                actions['trade'] = True
                actions['type'] = 'sell' if deviation > 0 else 'buy'
                actions['amount'] = self.state['balance'] * self.risk_tolerance * abs(deviation)
        
        return actions
    
    def _technical_strategy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Technical analysis based trading strategy."""
        actions = {'trade': False, 'amount': 0.0, 'type': 'buy'}
        
        price_history = state.get('price_history', [])
        if len(price_history) < 14:  # Minimum required for RSI
            return actions
            
        # Calculate RSI
        prices = np.array(price_history)
        delta = np.diff(prices)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        rs = avg_gain / (avg_loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema12 = np.mean(prices[-12:])
        ema26 = np.mean(prices[-26:]) if len(prices) >= 26 else ema12
        macd = ema12 - ema26
        signal = np.mean(prices[-9:])  # Signal line
        
        # RSI strategy
        if rsi < 30:
            actions['trade'] = True
            actions['type'] = 'buy'
            actions['amount'] = self.state['balance'] * self.risk_tolerance * (30 - rsi) / 30
        elif rsi > 70:
            actions['trade'] = True
            actions['type'] = 'sell'
            actions['amount'] = self.state['token_balance'] * self.risk_tolerance * (rsi - 70) / 30
        
        # MACD strategy
        if macd > signal and not actions['trade']:
            actions['trade'] = True
            actions['type'] = 'buy'
            actions['amount'] = self.state['balance'] * self.risk_tolerance * 0.5
        elif macd < signal and not actions['trade']:
            actions['trade'] = True
            actions['type'] = 'sell'
            actions['amount'] = self.state['token_balance'] * self.risk_tolerance * 0.5
        
        return actions
    
    def _random_strategy(self) -> Dict[str, Any]:
        """Random trading strategy."""
        actions = {'trade': False, 'amount': 0.0, 'type': 'buy'}
        
        if random.random() < 0.1:  # 10% chance to trade
            actions['trade'] = True
            actions['type'] = 'buy' if random.random() < 0.5 else 'sell'
            actions['amount'] = self.state['balance'] * random.uniform(0.1, 0.3)
        
        return actions
    
    def _apply_risk_management(self, amount: float, trade_type: str) -> float:
        """Apply risk management rules to trade amount."""
        if trade_type == 'buy':
            # Limit position size
            max_amount = self.state['balance'] * self.max_position_size
            amount = min(amount, max_amount)
            
            # Ensure sufficient balance
            amount = min(amount, self.state['balance'])
        else:  # sell
            # Limit position size
            max_amount = self.state['token_balance'] * self.max_position_size
            amount = min(amount, max_amount)
            
            # Ensure sufficient tokens
            amount = min(amount, self.state['token_balance'])
        
        return amount
    
    def update(self, reward: float, new_state: Dict[str, Any]) -> None:
        """Update trader's state based on rewards and new state."""
        super().update(reward, new_state)
        
        # Update token balance if there was a trade
        if 'trade_amount' in new_state:
            if new_state['trade_type'] == 'buy':
                self.state['token_balance'] += new_state['trade_amount']
                self.state['balance'] -= new_state['trade_amount'] * new_state['token_price']
                self.state['position_size'] = new_state['trade_amount']
                self.state['entry_price'] = new_state['token_price']
            else:  # sell
                self.state['token_balance'] -= new_state['trade_amount']
                self.state['balance'] += new_state['trade_amount'] * new_state['token_price']
                self.state['position_size'] = 0
                self.state['entry_price'] = 0
            
            self.state['trades'] += 1
            self.state['last_trade_price'] = new_state['token_price']
            
            # Record trade
            self.state['trade_history'].append({
                'type': new_state['trade_type'],
                'amount': new_state['trade_amount'],
                'price': new_state['token_price'],
                'timestamp': new_state.get('current_step', 0)
            })
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the trader."""
        return self.state.copy()
    
    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self.state.update({
            'trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'last_trade_price': 0.0,
            'entry_price': 0.0,
            'position_size': 0.0,
            'trade_history': []
        })
    
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