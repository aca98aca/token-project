import random
from Base_Agent import TokenAgent

# @title Miner Agent
class StorageMiner(TokenAgent):
    """An agent that mines tokens by providing storage to the network"""

    def __init__(self, unique_id, model, storage=1.0, 
                 operating_cost=1.0, initial_tokens=0, initial_fiat=100,efficiency=1.0):
        super().__init__(unique_id, model, initial_tokens)
        self.agent_type = "miner"
        self.active = True
        self.storage = storage  # Storage contribution in TB
        self.efficiency = efficiency  # Mining efficiency factor
        self.operating_cost = operating_cost  # Monthly cost to operate in USD
        self.fiat_balance = initial_fiat
        self.cumulative_rewards = 0
        self.last_reward = 0
        self.profit_history = []
        self.profitability_threshold = 0.1  # 10% profit margin to stay active

        self.base_operating_cost = 5.0  # Base cost: electricity, internet, etc.
        self.per_tb_cost = 1.0  # Cost per TB of storage
        self.operating_cost = self.base_operating_cost + (self.storage * self.per_tb_cost)

    def mine_tokens(self):
        """Mine tokens based on storage contribution relative to network"""
        if not self.active:
            return 0

        total_active_storage = self.model.get_total_active_storage()
        if total_active_storage == 0:
            return 0

        share = self.storage / total_active_storage
        reward = self.model.monthly_token_distribution * share * self.efficiency
        self.tokens += reward
        self.last_reward = reward
        self.cumulative_rewards += reward

        self.model.record_transaction("mining_reward", reward, self.model.token_price, self.unique_id)
        return reward

    def pay_costs(self):
        """Deduct monthly operating costs from fiat balance"""
        self.fiat_balance -= self.operating_cost
        self.model.record_transaction("power_cost", -self.operating_cost, 1, self.unique_id)

    def evaluate_profitability(self):
        """Deactivate if not profitable over threshold"""
        reward_value = self.last_reward * self.model.token_price
        profit = reward_value - self.operating_cost
        roi = profit / self.operating_cost if self.operating_cost > 0 else float('inf')
        self.record_profit(profit)

        if roi < self.profitability_threshold and self.model.schedule.steps - self.joined_at > 3:
            self.active = False
            if self in self.model.active_miners:
                self.model.active_miners.remove(self)
            self.model.inactive_miners.append(self)

    def step(self):
        """Monthly simulation step"""
        if not self.active:
            return

        self.mine_tokens()
        self.pay_costs()
        self.evaluate_profitability()

        if self.tokens > 0 and random.random() < 0.3:
            amount_to_sell = self.tokens * random.uniform(0.05, 0.2)
            self.sell_tokens(amount_to_sell)

        if self.model.verbose:
            print(
                f"Miner {self.unique_id} | active: {self.active} | reward: {self.last_reward:.2f} | "
                f"tokens: {self.tokens:.2f} | fiat: {self.fiat_balance:.2f}"
            )
