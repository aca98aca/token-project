import random
from typing import Dict, Any
from token_sim.agents import Agent
from token_sim.agents.trader import Trader
from token_sim.agents.holder import Holder

class MarketMaker:
    """Provides liquidity and price quotes in the market."""
    
    def __init__(self, 
                 unique_id: str,
                 model: Any,
                 liquidity_fiat: float = 50000.0,
                 liquidity_tokens: float = 50000.0,
                 fee_rate: float = 0.01,
                 price_volatility: float = 0.05):
        self.unique_id = unique_id
        self.model = model
        self.liquidity_fiat = liquidity_fiat
        self.liquidity_tokens = liquidity_tokens
        self.fee_rate = fee_rate
        self.price_volatility = price_volatility
        self.initial_liquidity_fiat = liquidity_fiat
        self.initial_liquidity_tokens = liquidity_tokens
        self.initial_price = self.provide_price_quote()
        self.trade_history = []
        self.current_volume = 0.0
    
    def provide_price_quote(self) -> float:
        """Provide a dynamic token price quote based on liquidity."""
        fiat = self.liquidity_fiat
        tokens = self.liquidity_tokens
        
        if tokens <= 0 or fiat <= 0:
            return max(self.model.price_discovery.current_price * (1 + random.uniform(-0.1, 0.1)), 0.001)
        
        base_price = fiat / tokens
        
        # Apply volatility
        volatility_factor = random.uniform(-self.price_volatility, self.price_volatility)
        adjusted_price = base_price * (1 + volatility_factor)
        
        # Clamp price within bounds
        min_price = self.model.price_discovery.current_price * 0.5
        max_price = self.model.price_discovery.current_price * 2.0
        final_price = max(min(adjusted_price, max_price), min_price)
        
        return round(final_price, 4)
    
    def _calculate_price_impact(self, amount: float, trade_type: str) -> float:
        """Calculate the price impact of a trade."""
        # Base impact is proportional to trade size relative to liquidity
        base_impact = amount / self.liquidity_tokens if trade_type == 'buy' else amount / self.liquidity_fiat
        
        # Add volatility factor
        volatility_factor = self.price_volatility * (1 + self.current_volume / self.liquidity_fiat)
        
        # Calculate final impact
        impact = base_impact * volatility_factor
        
        # Cap the impact
        return min(impact, 0.1)  # Maximum 10% price impact

    def execute_trade(self, agent: Agent, trade_type: str, amount: float) -> Dict[str, Any]:
        """Execute a trade with the market maker."""
        if trade_type == 'buy':
            # Calculate price impact
            price_impact = self._calculate_price_impact(amount, 'buy')
            execution_price = self.initial_price * (1 + price_impact)
            
            # Calculate total cost
            total_cost = amount * execution_price
            
            # Check if agent has enough balance
            if agent.state['fiat_balance'] < total_cost:
                return None
            
            # Execute trade
            agent.state['fiat_balance'] -= total_cost
            agent.state['token_balance'] += amount
            
            # Update market maker state
            self.liquidity_fiat += total_cost
            self.liquidity_tokens -= amount
            
            # Calculate profit/return
            profit = 0.0
            if isinstance(agent, Trader):
                # For traders, profit is based on price movement
                profit = -price_impact * amount * self.initial_price
            
            return {
                'type': 'buy',
                'amount': amount,
                'price': execution_price,
                'volume': total_cost,
                'profit': profit
            }
            
        elif trade_type == 'sell':
            # Calculate price impact
            price_impact = self._calculate_price_impact(amount, 'sell')
            execution_price = self.initial_price * (1 - price_impact)
            
            # Calculate total value
            total_value = amount * execution_price
            
            # Check if agent has enough tokens
            if agent.state['token_balance'] < amount:
                return None
            
            # Execute trade
            agent.state['token_balance'] -= amount
            agent.state['fiat_balance'] += total_value
            
            # Update market maker state
            self.liquidity_fiat -= total_value
            self.liquidity_tokens += amount
            
            # Calculate profit/return
            profit = 0.0
            return_value = 0.0
            
            if isinstance(agent, Trader):
                # For traders, profit is based on price movement
                profit = price_impact * amount * self.initial_price
            elif isinstance(agent, Holder):
                # For holders, return is based on holding period
                holding_period = self.model.current_step - agent.state.get('last_trade_step', 0)
                return_value = (execution_price - agent.state.get('last_trade_price', execution_price)) * amount
                return_value *= (1 + 0.01 * holding_period)  # Add small bonus for holding
            
            return {
                'type': 'sell',
                'amount': amount,
                'price': execution_price,
                'volume': total_value,
                'profit': profit,
                'return': return_value
            }
        
        return None

    def reset(self):
        """Reset market maker state."""
        self.liquidity_fiat = self.initial_liquidity_fiat
        self.liquidity_tokens = self.initial_liquidity_tokens
        self.current_price = self.initial_price
        self.trade_history = []
        self.current_volume = 0.0

    def get_parameter(self, parameter_name: str) -> Any:
        """Get a parameter value from the market maker."""
        if hasattr(self, parameter_name):
            return getattr(self, parameter_name)
        raise ValueError(f"Unknown parameter: {parameter_name}") 