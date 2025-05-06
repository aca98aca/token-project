from typing import Dict, List, Tuple
import numpy as np

class PriceDiscovery:
    """Handles token price discovery and market dynamics."""
    
    def __init__(self,
                 initial_price: float = 1.0,
                 volatility: float = 0.1,
                 market_depth: float = 1000000.0,
                 price_impact_factor: float = 0.0001):
        self.initial_price = initial_price
        self.current_price = initial_price
        self.volatility = volatility
        self.market_depth = market_depth
        self.price_impact_factor = price_impact_factor
        self.current_volume = 0.0
        self.price_history: List[float] = [initial_price]
        self.volume_history: List[float] = [0.0]
        self.last_update_time = 0
        self.order_book = []
        self.trade_history = []
        self.liquidity = market_depth
        self.fee_rate = 0.001
        self._last_block_height = 0
        self._balances = {}  # Track agent balances
        
    def update_price(self, 
                    volume: float = 0.0,
                    market_sentiment: float = 0.0,
                    time_step: int = 1) -> float:
        """Update token price based on trading volume and market conditions.
        
        Args:
            volume: Trading volume in the current period
            market_sentiment: Market sentiment factor (-1 to 1)
            time_step: Current time step
            
        Returns:
            Updated token price
        """
        # Update current volume
        self.current_volume = volume
        
        # Calculate price impact from volume
        volume_impact = (volume / self.market_depth) * self.price_impact_factor
        
        # Add random market noise (reduced when volume is high)
        noise_factor = max(0.1, 1.0 - (volume / self.market_depth))
        market_noise = np.random.normal(0, self.volatility * noise_factor)
        
        # Calculate sentiment impact (weighted by volume)
        sentiment_weight = min(1.0, volume / self.market_depth)
        sentiment_impact = market_sentiment * self.volatility * 0.5 * sentiment_weight
        
        # Calculate price change
        price_change = volume_impact + market_noise + sentiment_impact
        
        # Update price with bounds to prevent extreme values
        self.current_price *= (1 + price_change)
        self.current_price = max(0.001, min(self.current_price, self.initial_price * 100))
        
        # Update volume history
        self.volume_history.append(volume)
        if len(self.volume_history) > 24:  # Keep last 24 periods
            self.volume_history.pop(0)
        
        # Update price history
        self.price_history.append(self.current_price)
        if len(self.price_history) > 24:  # Keep last 24 periods
            self.price_history.pop(0)
        
        # Update liquidity based on volume
        self.liquidity = max(self.market_depth * 0.1, self.liquidity * (1 + price_change * 0.1))
        
        # Update order book
        self._update_order_book()
        
        self.last_update_time = time_step
        return self.current_price
    
    def _update_order_book(self) -> None:
        """Generate a realistic order book based on current price and liquidity."""
        # Clear existing order book
        self.order_book = []
        
        # Number of orders on each side
        num_orders = 10
        
        # Generate buy orders (bids)
        for i in range(num_orders):
            price_discount = 0.005 * (i + 1)  # 0.5% steps down
            price = self.current_price * (1 - price_discount)
            size = self.liquidity * 0.01 * (num_orders - i) / num_orders  # Decreasing size
            
            self.order_book.append({
                'type': 'buy',
                'price': price,
                'amount': size,
                'total': price * size
            })
        
        # Generate sell orders (asks)
        for i in range(num_orders):
            price_premium = 0.005 * (i + 1)  # 0.5% steps up
            price = self.current_price * (1 + price_premium)
            size = self.liquidity * 0.01 * (num_orders - i) / num_orders  # Decreasing size
            
            self.order_book.append({
                'type': 'sell',
                'price': price,
                'amount': size,
                'total': price * size
            })
        
        # Add a trade to the trade history
        self.trade_history.append({
            'price': self.current_price,
            'volume': self.current_volume,
            'timestamp': self.last_update_time
        })
        
        # Limit trade history size
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def get_price_stats(self) -> Dict[str, float]:
        """Get current price statistics."""
        if not self.price_history:
            return {
                'mean': self.initial_price,
                'std': self.volatility,
                'min': self.initial_price,
                'max': self.initial_price,
                'price_change_24h': 0.0,
                'volatility': self.volatility,
                'volume_24h': 0.0
            }
        
        # Calculate basic statistics
        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history)
        
        # Calculate price momentum
        if len(prices) > 1:
            price_momentum = (prices[-1] - prices[-2]) / prices[-2]
        else:
            price_momentum = 0.0
        
        return {
            'mean': float(np.mean(prices)),
            'std': float(np.std(prices)),
            'min': float(np.min(prices)),
            'max': float(np.max(prices)),
            'price_change_24h': float((prices[-1] - prices[0]) / prices[0]),
            'volatility': float(np.std(prices) / np.mean(prices)),
            'volume_24h': float(np.sum(volumes)),
            'price_momentum': float(price_momentum),
            'current_volume': float(self.current_volume)
        }
    
    def get_price_history(self) -> List[float]:
        """Get historical price data."""
        return self.price_history.copy()
    
    def get_volume_history(self) -> List[float]:
        """Get historical volume data."""
        return self.volume_history.copy()
    
    def reset(self):
        """Reset price discovery state."""
        self.current_price = self.initial_price
        self.current_volume = 0.0
        self.price_history = [self.initial_price]
        self.volume_history = [0.0]
        self.last_update_time = 0

    def is_operational(self) -> bool:
        """Check if the market is operational."""
        return (
            self.current_price > 0 and
            self.liquidity > 0 and
            len(self.order_book) > 0
        )

    def force_price_crash(self, crash_factor: float, governance=None) -> None:
        """Force a price crash for testing purposes."""
        if not 0 < crash_factor < 1:
            raise ValueError("Crash factor must be between 0 and 1")
        
        # Reduce price
        self.current_price *= crash_factor
        
        # Clear order book
        self.order_book = []
        
        # Reduce liquidity
        self.liquidity *= crash_factor
        
        # Add emergency orders to stabilize
        self.order_book.append({
            'type': 'buy',
            'price': self.current_price * 0.9,
            'amount': self.liquidity * 0.1
        })
        
        # Create emergency governance proposals
        if governance:
            # Emergency proposal to increase market depth
            emergency_proposal = {
                'type': 'parameter_update',
                'parameter': 'market_depth',
                'new_value': self.market_depth * 1.5,
                'description': 'Emergency increase in market depth to stabilize price'
            }
            governance.submit_proposal(emergency_proposal)

    def get_balance(self, agent_id: str) -> float:
        """Get the token balance for an agent."""
        # If we're tracking this agent's balance, return it
        return self._balances.get(agent_id, 0.0)

    @property
    def last_block_height(self) -> int:
        """Get the last processed block height."""
        return self._last_block_height

    @last_block_height.setter
    def last_block_height(self, height: int) -> None:
        """Set the last processed block height."""
        self._last_block_height = height

    def is_healthy(self) -> bool:
        """Check if market is healthy based on conditions."""
        # Check if price is positive and not too volatile
        price_history = np.array(self.price_history)
        if len(price_history) < 2:
            return self.current_price > 0
        
        # Calculate volatility
        price_volatility = np.std(price_history) / np.mean(price_history)
        
        # Market is healthy if:
        # 1. Price is positive
        # 2. Volatility is not extreme
        # 3. Liquidity is available
        return (
            self.current_price > 0 and
            price_volatility < 0.5 and
            self.liquidity > 0
        )

    def update_balances(self, agents: List) -> None:
        """Update tracked balances from agent states."""
        for agent in agents:
            self._balances[agent.id] = agent.state['token_balance']

class SimplePriceDiscovery(PriceDiscovery):
    """A simplified price discovery model for optimization."""
    
    def __init__(self,
                 initial_price: float = 1.0,
                 market_depth: float = 1000000.0,
                 volatility: float = 0.1):
        super().__init__(
            initial_price=initial_price,
            volatility=volatility,
            market_depth=market_depth,
            price_impact_factor=0.001
        )
    
    def update_price(self, 
                    volume: float = 0.0,
                    market_sentiment: float = 0.0,
                    time_step: int = 1) -> float:
        """Simplified update method that considers volume and market sentiment."""
        # Calculate market sentiment based on volume trend if not provided
        if market_sentiment == 0.0 and len(self.volume_history) > 1:
            volume_trend = (volume - self.volume_history[-1]) / (self.volume_history[-1] + 1e-6)
            market_sentiment = np.clip(volume_trend, -1.0, 1.0)
        
        return super().update_price(
            volume=volume,
            market_sentiment=market_sentiment,
            time_step=time_step
        ) 