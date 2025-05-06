import random
from typing import Dict, Any
from token_sim.agents import Agent

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
    
    def execute_trade(self, agent: Agent, trade_type: str, amount: float) -> Dict[str, Any]:
        """Execute a token buy/sell at the current price."""
        price = self.provide_price_quote()
        
        # Safety checks
        if price <= 0 or amount <= 0:
            return None
        
        trade_result = {
            'type': trade_type,
            'price': price,
            'amount': amount,
            'volume': 0.0,
            'success': False,
            'timestamp': self.model.current_step
        }
        
        if trade_type == "buy":
            tokens_bought = amount / price
            if self.liquidity_tokens >= tokens_bought and agent.state['balance'] >= amount:
                self.liquidity_tokens -= tokens_bought
                self.liquidity_fiat += amount * (1 - self.fee_rate)
                
                agent.state['token_balance'] += tokens_bought
                agent.state['balance'] -= amount
                
                trade_result['volume'] = amount
                trade_result['success'] = True
                trade_result['tokens_bought'] = tokens_bought
                self.current_volume += amount
        
        elif trade_type == "sell":
            fiat_returned = amount * price
            if self.liquidity_fiat >= fiat_returned and agent.state['token_balance'] >= amount:
                self.liquidity_tokens += amount
                self.liquidity_fiat -= fiat_returned * (1 + self.fee_rate)
                
                agent.state['token_balance'] -= amount
                agent.state['balance'] += fiat_returned
                
                trade_result['volume'] = fiat_returned
                trade_result['success'] = True
                trade_result['fiat_returned'] = fiat_returned
                self.current_volume += fiat_returned
        
        if trade_result['success']:
            self.trade_history.append(trade_result)
            # Update price discovery
            self.model.price_discovery.update_price(
                volume=trade_result['volume'],
                market_sentiment=0.0,  # TODO: Implement market sentiment
                time_step=self.model.current_step
            )
            
        return trade_result if trade_result['success'] else None

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