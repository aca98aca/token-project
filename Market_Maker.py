import random
from Base_Agent import TokenAgent

# @title Market Maker
class MarketMaker(TokenAgent):
    """Provides liquidity and price quotes in the market"""

    def __init__(self, unique_id, model,
                 liquidity_fiat=50000.0,
                 liquidity_tokens=50000.0,
                 fee_rate=0.01,
                 price_volatility=0.05):
        super().__init__(unique_id, model)
        self.agent_type = "market_maker"
        self.liquidity_fiat = liquidity_fiat
        self.liquidity_tokens = liquidity_tokens
        self.fee_rate = fee_rate
        self.price_volatility = price_volatility

    def provide_price_quote(self):
        """Provide a dynamic token price quote based on liquidity"""
        fiat = self.liquidity_fiat
        tokens = self.liquidity_tokens

        if tokens <= 0 or fiat <= 0:
            return max(self.model.token_price * (1 + random.uniform(-0.1, 0.1)), 0.001)

        base_price = fiat / tokens

        # Apply volatility
        volatility_factor = random.uniform(-self.price_volatility, self.price_volatility)
        adjusted_price = base_price * (1 + volatility_factor)

        # Clamp price within bounds
        min_price = self.model.token_price * 0.5
        max_price = self.model.token_price * 2.0
        final_price = max(min(adjusted_price, max_price), min_price)

        return round(final_price, 4)

    def execute_trade(self, agent, trade_type, amount):
        """Executes a token buy/sell at the current price"""
        price = self.model.token_price

        if trade_type == "buy":
            tokens_bought = amount / price
            if self.liquidity_tokens >= tokens_bought:
                self.liquidity_tokens -= tokens_bought
                self.liquidity_fiat += amount * (1 - self.fee_rate)

                agent.tokens += tokens_bought
                agent.fiat_balance -= amount
                self.model.record_transaction("buy", amount, tokens_bought, agent.unique_id)

        elif trade_type == "sell":
            fiat_returned = amount * price
            if self.liquidity_fiat >= fiat_returned:
                self.liquidity_tokens += amount
                self.liquidity_fiat -= fiat_returned * (1 + self.fee_rate)

                agent.tokens -= amount
                agent.fiat_balance += fiat_returned
                self.model.record_transaction("sell", fiat_returned, amount, agent.unique_id)

    def step(self):
        """Market maker does not trade actively in this version"""
        pass
