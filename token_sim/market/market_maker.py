import random
import numpy as np
from Base_Agent import TokenAgent

# @title Market Maker
class MarketMaker(TokenAgent):
    """Provides liquidity and price quotes in the market with improved order book management"""

    def __init__(self, unique_id, model,
                 liquidity_fiat=50000.0,
                 liquidity_tokens=50000.0,
                 fee_rate=0.01,
                 price_volatility=0.05,
                 max_slippage=0.02,
                 order_book_depth=10):
        super().__init__(unique_id, model)
        self.agent_type = "market_maker"
        self.liquidity_fiat = liquidity_fiat
        self.liquidity_tokens = liquidity_tokens
        self.fee_rate = fee_rate
        self.price_volatility = price_volatility
        self.max_slippage = max_slippage
        self.order_book_depth = order_book_depth
        self.order_book = {
            'bids': [],  # List of (price, size) tuples
            'asks': []   # List of (price, size) tuples
        }
        self.initialize_order_book()

    def initialize_order_book(self):
        """Initialize the order book with spread orders"""
        base_price = self.liquidity_fiat / self.liquidity_tokens
        spread = base_price * self.price_volatility
        
        # Initialize bids (buy orders)
        for i in range(self.order_book_depth):
            price = base_price * (1 - spread * (i + 1) / self.order_book_depth)
            size = self.liquidity_tokens / (self.order_book_depth * 2)
            self.order_book['bids'].append((price, size))
        
        # Initialize asks (sell orders)
        for i in range(self.order_book_depth):
            price = base_price * (1 + spread * (i + 1) / self.order_book_depth)
            size = self.liquidity_tokens / (self.order_book_depth * 2)
            self.order_book['asks'].append((price, size))

    def calculate_slippage(self, amount, order_type):
        """Calculate price slippage based on order size and order book depth"""
        total_volume = sum(size for _, size in self.order_book['bids' if order_type == 'sell' else 'asks'])
        if total_volume == 0:
            return self.max_slippage
        
        # Calculate slippage based on order size relative to available liquidity
        slippage = min(self.max_slippage, (amount / total_volume) * self.max_slippage)
        return slippage

    def provide_price_quote(self, amount=None, order_type=None):
        """Provide a dynamic token price quote based on liquidity and order book depth"""
        fiat = self.liquidity_fiat
        tokens = self.liquidity_tokens

        if tokens <= 0 or fiat <= 0:
            return max(self.model.token_price * (1 + random.uniform(-0.1, 0.1)), 0.001)

        base_price = fiat / tokens

        # Apply volatility
        volatility_factor = random.uniform(-self.price_volatility, self.price_volatility)
        adjusted_price = base_price * (1 + volatility_factor)

        # Calculate slippage if amount and order type are provided
        if amount and order_type:
            slippage = self.calculate_slippage(amount, order_type)
            if order_type == 'buy':
                adjusted_price *= (1 + slippage)
            else:
                adjusted_price *= (1 - slippage)

        # Clamp price within bounds
        min_price = self.model.token_price * 0.5
        max_price = self.model.token_price * 2.0
        final_price = max(min(adjusted_price, max_price), min_price)

        return round(final_price, 4)

    def execute_trade(self, agent, trade_type, amount):
        """Executes a token buy/sell with improved price impact handling"""
        # Calculate effective price with slippage
        effective_price = self.provide_price_quote(amount, trade_type)
        
        if trade_type == "buy":
            tokens_bought = amount / effective_price
            if self.liquidity_tokens >= tokens_bought:
                # Update order book
                self.update_order_book('asks', effective_price, tokens_bought)
                
                self.liquidity_tokens -= tokens_bought
                self.liquidity_fiat += amount * (1 - self.fee_rate)

                agent.tokens += tokens_bought
                agent.fiat_balance -= amount
                self.model.record_transaction("buy", amount, tokens_bought, agent.unique_id)
                return True

        elif trade_type == "sell":
            fiat_returned = amount * effective_price
            if self.liquidity_fiat >= fiat_returned:
                # Update order book
                self.update_order_book('bids', effective_price, amount)
                
                self.liquidity_tokens += amount
                self.liquidity_fiat -= fiat_returned * (1 + self.fee_rate)

                agent.tokens -= amount
                agent.fiat_balance += fiat_returned
                self.model.record_transaction("sell", fiat_returned, amount, agent.unique_id)
                return True
        
        return False

    def update_order_book(self, side, price, size):
        """Update the order book after a trade"""
        orders = self.order_book[side]
        remaining_size = size
        
        # Match orders starting from the best price
        while remaining_size > 0 and orders:
            order_price, order_size = orders[0]
            if (side == 'bids' and price <= order_price) or (side == 'asks' and price >= order_price):
                matched_size = min(remaining_size, order_size)
                orders[0] = (order_price, order_size - matched_size)
                remaining_size -= matched_size
                
                # Remove empty orders
                if orders[0][1] <= 0:
                    orders.pop(0)
            else:
                break
        
        # Add remaining size as a new order
        if remaining_size > 0:
            orders.append((price, remaining_size))
            orders.sort(key=lambda x: x[0], reverse=(side == 'bids'))

    def step(self):
        """Update order book and adjust prices based on market conditions"""
        # Rebalance order book based on current price
        self.rebalance_order_book()
        
        # Adjust prices based on market conditions
        self.adjust_prices()

    def rebalance_order_book(self):
        """Rebalance the order book to maintain desired depth and spread"""
        base_price = self.liquidity_fiat / self.liquidity_tokens
        spread = base_price * self.price_volatility
        
        # Rebalance bids
        self.order_book['bids'] = []
        for i in range(self.order_book_depth):
            price = base_price * (1 - spread * (i + 1) / self.order_book_depth)
            size = self.liquidity_tokens / (self.order_book_depth * 2)
            self.order_book['bids'].append((price, size))
        
        # Rebalance asks
        self.order_book['asks'] = []
        for i in range(self.order_book_depth):
            price = base_price * (1 + spread * (i + 1) / self.order_book_depth)
            size = self.liquidity_tokens / (self.order_book_depth * 2)
            self.order_book['asks'].append((price, size))

    def adjust_prices(self):
        """Adjust prices based on market conditions and inventory"""
        # Calculate inventory imbalance
        inventory_ratio = self.liquidity_tokens / (self.liquidity_tokens + self.liquidity_fiat / self.model.token_price)
        
        # Adjust spread based on inventory imbalance
        if inventory_ratio > 0.6:  # Too many tokens
            self.price_volatility *= 1.1  # Increase spread to encourage selling
        elif inventory_ratio < 0.4:  # Too many fiat
            self.price_volatility *= 1.1  # Increase spread to encourage buying
        else:
            self.price_volatility *= 0.95  # Gradually decrease spread
        
        # Clamp volatility
        self.price_volatility = max(0.01, min(0.1, self.price_volatility))
