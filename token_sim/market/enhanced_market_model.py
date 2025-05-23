from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class OrderBookLevel:
    """Represents a price level in the order book."""
    price: float
    quantity: float
    orders: int

class OrderBook:
    """Implements a limit order book with price-time priority."""
    
    def __init__(self, initial_price: float, tick_size: float = 0.01):
        """Initialize the order book.
        
        Args:
            initial_price: Initial price of the asset
            tick_size: Minimum price movement
            
        Raises:
            ValueError: If initial_price or tick_size are invalid
        """
        if initial_price <= 0:
            raise ValueError("Initial price must be positive")
        if tick_size <= 0:
            raise ValueError("Tick size must be positive")
            
        self.tick_size = tick_size
        self.bids = {}  # price -> OrderBookLevel
        self.asks = {}  # price -> OrderBookLevel
        self.last_price = initial_price
        self.last_trade_price = initial_price
        self.last_trade_quantity = 0.0
        self.volume = 0.0
        
        # Performance optimization: Cache best bid/ask
        self._best_bid = None
        self._best_ask = None
        
        # Initialize with some liquidity
        self._initialize_liquidity(initial_price)
    
    def _initialize_liquidity(self, price: float):
        """Initialize the order book with some liquidity."""
        try:
            # Add some bids
            for i in range(10):
                bid_price = price * (1 - (i + 1) * 0.001)  # 0.1% steps
                bid_price = round(bid_price / self.tick_size) * self.tick_size
                self.bids[bid_price] = OrderBookLevel(
                    price=bid_price,
                    quantity=1000.0 * (1 - i * 0.1),  # Decreasing quantities
                    orders=1
                )
            
            # Add some asks
            for i in range(10):
                ask_price = price * (1 + (i + 1) * 0.001)  # 0.1% steps
                ask_price = round(ask_price / self.tick_size) * self.tick_size
                self.asks[ask_price] = OrderBookLevel(
                    price=ask_price,
                    quantity=1000.0 * (1 - i * 0.1),  # Decreasing quantities
                    orders=1
                )
                
            # Initialize best bid/ask cache
            self._update_best_prices()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize order book liquidity: {str(e)}")
    
    def _update_best_prices(self):
        """Update cached best bid and ask prices."""
        self._best_bid = max(self.bids.keys()) if self.bids else None
        self._best_ask = min(self.asks.keys()) if self.asks else None
    
    def add_order(self, side: str, price: float, quantity: float) -> bool:
        """Add a new order to the book.
        
        Args:
            side: 'bid' or 'ask'
            price: Order price
            quantity: Order quantity
            
        Returns:
            True if order was added, False if it was matched
            
        Raises:
            ValueError: If side, price, or quantity are invalid
        """
        # Input validation
        if side not in ['bid', 'ask']:
            raise ValueError("Side must be 'bid' or 'ask'")
        if price <= 0:
            raise ValueError("Price must be positive")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
            
        try:
            price = round(price / self.tick_size) * self.tick_size
            
            if side == 'bid':
                # Check if we can match against asks
                if self._best_ask is not None and price >= self._best_ask:
                    return self._match_order(side, price, quantity)
                
                # Add to bids
                if price in self.bids:
                    self.bids[price].quantity += quantity
                    self.bids[price].orders += 1
                else:
                    self.bids[price] = OrderBookLevel(price=price, quantity=quantity, orders=1)
                    
                # Update best bid if necessary
                if self._best_bid is None or price > self._best_bid:
                    self._best_bid = price
                    
            else:  # ask
                # Check if we can match against bids
                if self._best_bid is not None and price <= self._best_bid:
                    return self._match_order(side, price, quantity)
                
                # Add to asks
                if price in self.asks:
                    self.asks[price].quantity += quantity
                    self.asks[price].orders += 1
                else:
                    self.asks[price] = OrderBookLevel(price=price, quantity=quantity, orders=1)
                    
                # Update best ask if necessary
                if self._best_ask is None or price < self._best_ask:
                    self._best_ask = price
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to add order: {str(e)}")
    
    def _match_order(self, side: str, price: float, quantity: float) -> bool:
        """Match an order against the opposite side of the book.
        
        Args:
            side: 'bid' or 'ask'
            price: Order price
            quantity: Order quantity
            
        Returns:
            False to indicate the order was matched
            
        Raises:
            RuntimeError: If matching fails
        """
        try:
            remaining = quantity
            
            if side == 'bid':
                while remaining > 0 and self._best_ask is not None:
                    if price < self._best_ask:
                        break
                        
                    level = self.asks[self._best_ask]
                    matched = min(remaining, level.quantity)
                    
                    # Update the book
                    level.quantity -= matched
                    remaining -= matched
                    self.volume += matched
                    self.last_trade_price = self._best_ask
                    self.last_trade_quantity = matched
                    
                    if level.quantity == 0:
                        del self.asks[self._best_ask]
                        self._update_best_prices()
                    
            else:  # ask
                while remaining > 0 and self._best_bid is not None:
                    if price > self._best_bid:
                        break
                        
                    level = self.bids[self._best_bid]
                    matched = min(remaining, level.quantity)
                    
                    # Update the book
                    level.quantity -= matched
                    remaining -= matched
                    self.volume += matched
                    self.last_trade_price = self._best_bid
                    self.last_trade_quantity = matched
                    
                    if level.quantity == 0:
                        del self.bids[self._best_bid]
                        self._update_best_prices()
            
            return False
            
        except Exception as e:
            raise RuntimeError(f"Failed to match order: {str(e)}")
    
    def get_market_depth(self, levels: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        """Get the current market depth.
        
        Args:
            levels: Number of price levels to return
            
        Returns:
            Dictionary with bid and ask depth
            
        Raises:
            ValueError: If levels is invalid
        """
        if levels <= 0:
            raise ValueError("Levels must be positive")
            
        try:
            bids = sorted(self.bids.items(), reverse=True)[:levels]
            asks = sorted(self.asks.items())[:levels]
            
            return {
                'bids': [(price, level.quantity) for price, level in bids],
                'asks': [(price, level.quantity) for price, level in asks]
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get market depth: {str(e)}")
    
    def get_spread(self) -> float:
        """Get the current bid-ask spread."""
        if self._best_bid is None or self._best_ask is None:
            return float('inf')
        return self._best_ask - self._best_bid
    
    def get_mid_price(self) -> float:
        """Get the current mid price."""
        if self._best_bid is None or self._best_ask is None:
            return self.last_price
        return (self._best_ask + self._best_bid) / 2

class EnhancedMarketModel:
    """Enhanced market model with sophisticated order book dynamics."""
    
    def __init__(self, initial_price: float, market_depth: float, volatility: float, max_history_size: int = 10000):
        """Initialize the market model.
        
        Args:
            initial_price: Initial price of the asset
            market_depth: Initial market depth
            volatility: Price volatility
            max_history_size: Maximum number of historical data points to keep
            
        Raises:
            ValueError: If any input parameters are invalid
        """
        # Validate input parameters
        if initial_price <= 0:
            raise ValueError("Initial price must be positive")
        if market_depth <= 0:
            raise ValueError("Market depth must be positive")
        if volatility < 0 or volatility > 1:
            raise ValueError("Volatility must be between 0 and 1")
        if max_history_size <= 0:
            raise ValueError("Max history size must be positive")
            
        self.order_book = OrderBook(initial_price)
        self.initial_price = initial_price
        self.market_depth = market_depth
        self.volatility = volatility
        self.max_history_size = max_history_size
        
        # Use numpy arrays for better performance
        self.price_history = np.zeros(max_history_size)
        self.volume_history = np.zeros(max_history_size)
        self.spread_history = np.zeros(max_history_size)
        self.history_index = 0
        
        # Initialize first data point
        self.price_history[0] = initial_price
        self.volume_history[0] = 0.0
        self.spread_history[0] = self.order_book.get_spread()
        
        # Market making parameters
        self.min_spread = initial_price * 0.001  # 0.1% minimum spread
        self.max_spread = initial_price * 0.01   # 1% maximum spread
        self.depth_factor = 0.1  # 10% of volume as depth
        
        # Initialize market makers
        self._initialize_market_makers()
        
        # Add validation state
        self.last_update_time = 0
        self.max_price_change = initial_price * 0.1  # 10% maximum price change per update
        
        # Performance monitoring
        self.performance_metrics = {
            'order_count': 0,
            'match_count': 0,
            'update_time': 0.0
        }
    
    def _initialize_market_makers(self):
        """Initialize market makers with different strategies."""
        self.market_makers = []
        
        # Conservative market maker
        self.market_makers.append({
            'spread_multiplier': 1.5,
            'depth_multiplier': 0.8,
            'update_frequency': 0.1
        })
        
        # Aggressive market maker
        self.market_makers.append({
            'spread_multiplier': 0.8,
            'depth_multiplier': 1.2,
            'update_frequency': 0.2
        })
        
        # Adaptive market maker
        self.market_makers.append({
            'spread_multiplier': 1.0,
            'depth_multiplier': 1.0,
            'update_frequency': 0.15
        })
    
    def _add_to_history(self, price: float, volume: float, spread: float):
        """Add data point to history with circular buffer behavior."""
        self.price_history[self.history_index] = price
        self.volume_history[self.history_index] = volume
        self.spread_history[self.history_index] = spread
        
        self.history_index = (self.history_index + 1) % self.max_history_size
    
    def get_history(self) -> Dict[str, np.ndarray]:
        """Get historical data with proper ordering."""
        if self.history_index == 0:
            return {
                'price': self.price_history,
                'volume': self.volume_history,
                'spread': self.spread_history
            }
        
        # Reorder data to maintain chronological order
        return {
            'price': np.concatenate([self.price_history[self.history_index:], 
                                   self.price_history[:self.history_index]]),
            'volume': np.concatenate([self.volume_history[self.history_index:], 
                                    self.volume_history[:self.history_index]]),
            'spread': np.concatenate([self.spread_history[self.history_index:], 
                                    self.spread_history[:self.history_index]])
        }
    
    def update(self, time_step: int, market_conditions: Dict[str, Any]):
        """Update the market state.
        
        Args:
            time_step: Current time step
            market_conditions: Current market conditions
            
        Raises:
            ValueError: If time_step is invalid or market_conditions are missing required fields
            RuntimeError: If price change exceeds maximum allowed
        """
        import time
        start_time = time.time()
        
        try:
            # Validate time step
            if time_step < self.last_update_time:
                raise ValueError("Time step must be greater than last update time")
            self.last_update_time = time_step
            
            # Validate market conditions
            required_fields = ['hashrate', 'block_time', 'participant_count']
            missing_fields = [field for field in required_fields if field not in market_conditions]
            if missing_fields:
                raise ValueError(f"Missing required market conditions: {missing_fields}")
                
            # Update market makers
            self._update_market_makers(market_conditions)
            
            # Simulate trading activity
            self._simulate_trading_activity(market_conditions)
            
            # Validate price change
            current_price = self.order_book.get_mid_price()
            prev_price = self.price_history[self.history_index - 1]
            if prev_price == 0:
                price_change = 0.0
            else:
                price_change = abs(current_price - prev_price) / prev_price
            if prev_price != 0 and price_change > self.max_price_change:
                raise RuntimeError(f"Price change {price_change:.2%} exceeds maximum allowed {self.max_price_change:.2%}")
            
            # Update history
            self._add_to_history(
                current_price,
                self.order_book.volume,
                self.order_book.get_spread()
            )
            
            # Clean up old orders
            self._cleanup_old_orders()
            
            # Update performance metrics
            self.performance_metrics['update_time'] = time.time() - start_time
            
        except Exception as e:
            raise RuntimeError(f"Failed to update market state: {str(e)}")
    
    def _update_market_makers(self, market_conditions: Dict[str, Any]):
        """Update market maker orders based on current conditions."""
        mid_price = self.order_book.get_mid_price()
        volatility = market_conditions.get('volatility', self.volatility)
        
        for maker in self.market_makers:
            # Calculate spread and depth
            spread = max(
                self.min_spread,
                min(self.max_spread, volatility * maker['spread_multiplier'])
            )
            depth = self.market_depth * maker['depth_multiplier']
            
            # Place new orders
            bid_price = mid_price - spread / 2
            ask_price = mid_price + spread / 2
            
            self.order_book.add_order('bid', bid_price, depth)
            self.order_book.add_order('ask', ask_price, depth)
    
    def _simulate_trading_activity(self, market_conditions: Dict[str, Any]):
        """Simulate trading activity based on market conditions."""
        # Simulate random trades
        num_trades = np.random.poisson(10)  # Average 10 trades per update
        
        for _ in range(num_trades):
            # Randomly choose side and size
            side = np.random.choice(['bid', 'ask'])
            size = np.random.lognormal(mean=5, sigma=1)
            
            # Calculate price based on current spread
            mid_price = self.order_book.get_mid_price()
            spread = self.order_book.get_spread()
            
            if side == 'bid':
                price = mid_price - spread * np.random.random()
            else:
                price = mid_price + spread * np.random.random()
            
            # Place the order
            self.order_book.add_order(side, price, size)
    
    def _cleanup_old_orders(self):
        """Clean up old orders from the book."""
        # Remove orders that are too far from the mid price
        mid_price = self.order_book.get_mid_price()
        max_distance = mid_price * 0.05  # 5% maximum distance
        
        # Clean bids
        for price in list(self.order_book.bids.keys()):
            if mid_price - price > max_distance:
                del self.order_book.bids[price]
        
        # Clean asks
        for price in list(self.order_book.asks.keys()):
            if price - mid_price > max_distance:
                del self.order_book.asks[price]
    
    def get_market_metrics(self) -> Dict[str, Any]:
        """Get current market metrics."""
        return {
            'price': self.order_book.get_mid_price(),
            'spread': self.order_book.get_spread(),
            'volume': self.order_book.volume,
            'depth': self.order_book.get_market_depth(),
            'history': self.get_history(),
            'performance': self.performance_metrics
        } 