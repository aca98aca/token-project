import random
from Base_Agent import TokenAgent

# @title Token Holders
class TokenHolder(TokenAgent):
    """An agent that holds tokens for investment"""

    def __init__(self, unique_id, model, investment_strategy='hold',
                 risk_tolerance=0.5, initial_tokens=0, initial_fiat=500,price_belief=None):
        super().__init__(unique_id, model, initial_tokens)
        self.agent_type = "holder"
        self.fiat_balance = initial_fiat
        self.investment_strategy = investment_strategy
        self.risk_tolerance = risk_tolerance
        self.price_belief = price_belief if price_belief is not None else model.token_price
        self.price_history = []

        # Emergent behavioral thresholds
        self.sell_threshold = random.uniform(1.2, 1.6)
        self.buy_threshold = random.uniform(0.7, 0.95)
        self.sell_fraction = random.uniform(0.1, 0.4)
        self.buy_fraction = random.uniform(0.1, 0.4)

    def update_price_belief(self):
        """Update belief about token price based on market and history"""
        price = self.model.token_price
        self.price_history.append(price)

        if len(self.price_history) > 30:
            self.price_history = self.price_history[-30:]

        if len(self.price_history) >= 7:
            short_ma = sum(self.price_history[-7:]) / 7
            long_ma = sum(self.price_history) / len(self.price_history)

            if short_ma > long_ma:
                self.price_belief *= 1.01
            elif short_ma < long_ma:
                self.price_belief *= 0.99
            else:
                self.price_belief *= random.uniform(0.99, 1.01)
        else:
            self.price_belief = self.model.token_price * random.uniform(0.95, 1.05)

    def make_trade_decision(self):
        """Decide whether to buy, sell, or hold tokens based on strategy"""
        price = self.model.token_price

        if self.investment_strategy == 'hold':
            if price > self.price_belief * self.sell_threshold and self.tokens > 0:
                amount_to_sell = self.tokens * self.sell_fraction
                self.sell_tokens(amount_to_sell)
                return ('sell', amount_to_sell, price * amount_to_sell)

        elif self.investment_strategy == 'trade':
            if price < self.price_belief * self.buy_threshold and self.fiat_balance > 0:
                amount_to_spend = self.fiat_balance * self.buy_fraction
                self.buy_tokens(amount_to_spend)
                return ('buy', amount_to_spend / price, amount_to_spend)
            elif price > self.price_belief * self.sell_threshold and self.tokens > 0:
                amount_to_sell = self.tokens * self.sell_fraction
                self.sell_tokens(amount_to_sell)
                return ('sell', amount_to_sell, price * amount_to_sell)

        elif self.investment_strategy == 'stake':
            if price < self.price_belief * self.buy_threshold and self.fiat_balance > 0:
                amount_to_spend = self.fiat_balance * (self.risk_tolerance + 0.1)
                self.buy_tokens(amount_to_spend)
                return ('buy', amount_to_spend / price, amount_to_spend)

        return ('hold', 0, 0)

    def step(self):
        """Perform holder activities"""
        self.update_price_belief()
        action, amount, value = self.make_trade_decision()

        if len(self.price_history) >= 2:
            token_value_last = self.tokens * self.price_history[-2]
            token_value_now = self.tokens * self.price_history[-1]
            profit = token_value_now - token_value_last
            self.record_profit(profit)

        if self.model.verbose:
            print(
                f"Holder {self.unique_id} | strategy: {self.investment_strategy} | action: {action} | "
                f"tokens: {self.tokens:.2f} | fiat: {self.fiat_balance:.2f}"
            )
