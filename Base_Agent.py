import mesa

# @title Base Agent
class TokenAgent(mesa.Agent):
    """Base class for all agents in the token economy"""

    def __init__(self, unique_id, model, initial_tokens=0):
        super().__init__(unique_id, model)
        self.tokens = initial_tokens  # Token balance
        self.fiat_balance = 0  # USD or other fiat currency
        self.joined_at = model.schedule.steps  # When agent joined
        self.profit_history = []  # Historical record of profits/losses
        self.active = True  # Whether agent is active in the network

    def record_profit(self, amount):
        """Record a profit or loss event"""
        self.profit_history.append(amount)

    def calculate_profit(self, period=30):
        """Calculate profit over the last period"""
        if len(self.profit_history) < period:
            return sum(self.profit_history)
        return sum(self.profit_history[-period:])

    def buy_tokens(self, amount_fiat):
        """Buy tokens using MarketMaker execution"""
        if amount_fiat <= 0 or amount_fiat > self.fiat_balance:
            return
        self.model.market_maker.execute_trade(self, "buy", amount_fiat)

    def sell_tokens(self, amount_tokens):
        """Sell tokens using MarketMaker execution"""
        if amount_tokens <= 0 or amount_tokens > self.tokens:
            return
        self.model.market_maker.execute_trade(self, "sell", amount_tokens)

    def step(self):
        """Base step method to be overridden by subclasses"""
        pass
